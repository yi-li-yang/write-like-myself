#!/usr/bin/env python3
"""Generate texts for the test tasks, build blind review files and score Yili's verdicts.

    python -X utf8 tools/rounds.py split                      freeze the task pool and assign tasks to sets
    python -X utf8 tools/rounds.py generate --round 0         write the texts for a review round
    python -X utf8 tools/rounds.py generate --set dev --condition skill --samples 3
                                                              write texts for automated checks
    python -X utf8 tools/rounds.py make --round 0             build the blind review file
    python -X utf8 tools/rounds.py pairs --round 1            build a blind file of the two texts for each task
    python -X utf8 tools/rounds.py score --round 0            score the verdicts in the review file, and the pairs
    python -X utf8 tools/rounds.py check --label dev          mechanical checks of generated texts
    python -X utf8 tools/rounds.py calibrate                  test whether an AI grader agrees with Yili's past verdicts

Every run is a fresh, headless `claude -p` session in its own folder, with no user plugins, hooks or MCP
servers. The baseline condition gets Yili's current Writing Rules block as its CLAUDE.md; the skill condition
gets this repository as a plugin and nothing else. Texts, keys and verdicts live in the private folder.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUBLIC_TASKS = REPO / "evals" / "tasks.json"
MODEL = "claude-opus-5-5"


def find_claude():
    """The newest Claude Code CLI: the one bundled with the VS Code extension if present, since a global
    npm install can lag behind and refuse newer models."""
    bundled = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe"),
                     key=lambda p: [int(x) for x in re.findall(r"\d+", p.parts[-4])])
    return str(bundled[-1]) if bundled else shutil.which("claude") or "claude"


CLAUDE = find_claude()
FAMILIES = ["papers", "letters", "web", "everyday"]
SET_SIZES = {"dev": 4, "round1": 3, "round2": 3}   # per family; the first two dev tasks form round 0
TIMEOUT = 900


def private_dir():
    local = tomllib.loads((REPO / "tools" / "local.toml").read_text(encoding="utf-8"))
    return Path(local["private_dir"]).expanduser()


PRIVATE = private_dir()
PRIVATE_TASKS = PRIVATE / "tasks_private" / "tasks.json"
LOCK = PRIVATE / "tasks_private" / "lock.json"


def load_tasks():
    tasks = []
    for path in (PUBLIC_TASKS, PRIVATE_TASKS):
        for task in json.loads(path.read_text(encoding="utf-8"))["tasks"]:
            task["private"] = path == PRIVATE_TASKS
            tasks.append(task)
    return {t["id"]: t for t in tasks}


def sha256(path):
    # Line endings are normalised, since git on Windows may rewrite them on checkout.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_lock():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for name, path in (("public", PUBLIC_TASKS), ("private", PRIVATE_TASKS)):
        if sha256(path) != lock["sha256"][name]:
            sys.exit(f"rounds: the {name} task file changed after the pool was frozen. Restore it, "
                     "or add new tasks to a new pool.")
    return lock


# ---------------------------------------------------------------- split

def writing_rules_block():
    """The Writing Rules and style-guide sections of Yili's RTSV2 CLAUDE.md, verbatim."""
    config = tomllib.loads((PRIVATE / "config.toml").read_text(encoding="utf-8"))
    text = (Path(config["sources"]["rtsv2_repo"]) / "CLAUDE.md").read_text(encoding="utf-8")
    start, end = text.index("## Writing Rules:"), text.index("## Other rules:")
    return text[start:end].strip() + "\n"


def split(args):
    if LOCK.exists() and not args.force:
        sys.exit("rounds: the pool is already frozen (see the lock file). Use --force only to start a new pool.")
    tasks = load_tasks()
    rng = random.Random(args.seed)
    sets = {name: [] for name in SET_SIZES}
    for family in FAMILIES:
        ids = sorted(t for t in tasks if tasks[t]["family"] == family)
        rng.shuffle(ids)
        start = 0
        for name, size in SET_SIZES.items():
            sets[name] += ids[start:start + size]
            start += size
        if start != len(ids):
            sys.exit(f"rounds: family {family} has {len(ids)} tasks, expected {start}")
    round0 = [t for family in FAMILIES for t in [i for i in sets["dev"] if tasks[i]["family"] == family][:2]]
    baseline_rules = writing_rules_block()
    (PRIVATE / "tasks_private" / "baseline_CLAUDE.md").write_text(baseline_rules, encoding="utf-8")
    lock = {
        "frozen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "sha256": {"public": sha256(PUBLIC_TASKS), "private": sha256(PRIVATE_TASKS)},
        "sets": sets,
        "round0": round0,
        "model": MODEL,
        "baseline_rules_sha256": hashlib.sha256(baseline_rules.encode()).hexdigest(),
    }
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"rounds: froze {len(tasks)} tasks. Round 0: {', '.join(round0)}")
    for name, ids in sets.items():
        print(f"  {name}: {', '.join(ids)}")


# ---------------------------------------------------------------- generate

def notes_for(task):
    lines = ["# Notes", "", f"Audience: {task['audience']}"]
    if task.get("limit_words"):
        lines.append(f"Length: at most {task['limit_words']} words.")
    lines += ["", "Facts:"] + [f"- {fact}" for fact in task["facts"]]
    return "\n".join(lines) + "\n"


def prompt_for(task):
    files = ["notes.md"] + list(task.get("materials", {}))
    named = files[0] if len(files) == 1 else ", ".join(files[:-1]) + " and " + files[-1]
    return f"{task['prompt']}\n\n(Everything you need is in {named}. Put the finished text in draft.md.)"


def run_one(task, condition, run_dir, plugin=None):
    """One headless session; returns its metadata. Skips runs that already finished. The skill condition
    uses this repository's plugin unless another plugin folder is given."""
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "notes.md").write_text(notes_for(task), encoding="utf-8")
    for name, content in task.get("materials", {}).items():
        (run_dir / name).write_text(content, encoding="utf-8")
    # The prompt goes in on stdin: on Windows `claude` is a .cmd wrapper, and cmd.exe would split a
    # multi-line argument.
    command = [CLAUDE, "-p", "--model", MODEL, "--setting-sources", "project,local",
               "--strict-mcp-config", "--permission-mode", "acceptEdits", "--no-session-persistence",
               "--output-format", "stream-json", "--verbose"]
    if condition == "baseline":
        rules = (PRIVATE / "tasks_private" / "baseline_CLAUDE.md").read_text(encoding="utf-8")
        (run_dir / "CLAUDE.md").write_text(rules, encoding="utf-8")
    elif condition == "skill":
        command += ["--plugin-dir", str(plugin or REPO / "plugins" / "write-like-myself")]
    else:
        sys.exit(f"rounds: unknown condition {condition}")
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    started = datetime.now(timezone.utc)
    with open(run_dir / "stream.jsonl", "w", encoding="utf-8") as out, \
            open(run_dir / "stderr.txt", "w", encoding="utf-8") as err:
        code = subprocess.run(command, cwd=run_dir, env=env, input=prompt_for(task), text=True,
                              encoding="utf-8", stdout=out, stderr=err, timeout=TIMEOUT).returncode
    meta = {"task": task["id"], "condition": condition, "exit": code, "started": started.isoformat(timespec="seconds"),
            "tools": [], "skills": [], "plugin": str(plugin) if plugin else None}
    final = ""
    for line in (run_dir / "stream.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            meta["model"] = event.get("model")
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    meta["tools"].append(block.get("name"))
                    if block.get("name") == "Skill":
                        meta["skills"].append(block.get("input", {}).get("skill"))
        if event.get("type") == "result":
            final = event.get("result") or ""
            meta.update(cost_usd=event.get("total_cost_usd"), duration_ms=event.get("duration_ms"),
                        outcome=event.get("subtype"))
    draft = run_dir / "draft.md"
    text = draft.read_text(encoding="utf-8").strip() if draft.exists() else final.strip()
    meta["text_source"] = "draft.md" if draft.exists() else "final message"
    if code != 0 or meta.get("outcome") != "success" or not text or final.startswith("API Error"):
        # Keep the evidence but leave no meta.json, so the next generate call retries this run.
        meta["error"] = final[:500]
        (run_dir / "failed.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        draft.unlink(missing_ok=True)
        return meta
    (run_dir / "text.md").write_text(text + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def round_plan(lock, number):
    """(task, condition) pairs for a review round. Round 0 tested the current rules alone. Later rounds take
    the first two tasks of each family from the round's set and write each one with the skill and with the rules."""
    if number == 0:
        return [(t, "baseline") for t in lock["round0"]]
    tasks = load_tasks()
    ids = lock["sets"][f"round{number}"]
    chosen = [t for family in FAMILIES for t in [i for i in ids if tasks[i]["family"] == family][:2]]
    return [(t, c) for t in chosen for c in ("skill", "baseline")]


def generate(args):
    lock = load_lock()
    tasks = load_tasks()
    if args.round is not None:
        jobs = [(t, c, 1, PRIVATE / "results" / f"round-{args.round}" / f"{t}-{c}") for t, c in round_plan(lock, args.round)]
    else:
        ids = args.tasks.split(",") if args.tasks else lock["sets"][args.set]
        jobs = [(t, args.condition, s, PRIVATE / "results" / args.label / f"{t}-{args.condition}-{s}")
                for t in ids for s in range(1, args.samples + 1)]
    plugin = Path(args.plugin).expanduser() if args.plugin else None
    version = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"rounds: {len(jobs)} runs, {args.parallel} at a time, Claude Code {version}, model {MODEL}")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, tasks[t], c, d, plugin): (t, c, s) for t, c, s, d in jobs}
        for future, (t, c, s) in futures.items():
            meta = future.result()
            status = f"FAILED: {meta['error'][:120]}" if meta.get("error") else f"text from {meta['text_source']}"
            print(f"  {t} {c} #{s}: exit {meta['exit']}, ${meta.get('cost_usd') or 0:.2f}, "
                  f"skills {meta['skills'] or '-'}, {status}")


# ---------------------------------------------------------------- make

INSTRUCTIONS = """For each item, read the task and then the text, and fill in three lines.

- Verdict: 1 if you would use the text as it is, "1 with edits" if you would use it after small edits,
  or 0 if you would not use it.
- Reason: what is wrong, whenever the verdict is not a plain 1.
- AI: 1 if a reader would suspect a machine wrote it, 0 if not, or "not sure".

You never need to edit the text. Save the file when you are done and tell Claude.
"""


def reviewer_notes(folder):
    """Facts the reviewer needs that the texts alone do not show, such as an error in a task, by task id."""
    path = folder / "notes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def task_lines(task, note=None):
    """The task as the reviewer sees it: the request, the audience, the notes and any materials."""
    lines = [f"**Task:** {task['prompt']}", "", f"**Audience:** {task['audience']}", "",
             "**Notes given to the writer:**", ""] + [f"- {fact}" for fact in task["facts"]]
    for name, content in task.get("materials", {}).items():
        lines += ["", f"**{name}:**", ""] + [f"> {line}" if line else ">" for line in content.splitlines()]
    if note:
        lines += ["", f"**Note for the reviewer:** {note}"]
    return lines


def round_text(number, task_id, condition):
    path = PRIVATE / "results" / f"round-{number}" / f"{task_id}-{condition}" / "text.md"
    if not path.exists():
        sys.exit(f"rounds: no text yet for {task_id} ({condition}). Run generate first.")
    return path.read_text(encoding="utf-8").strip()


def make(args):
    lock = load_lock()
    tasks = load_tasks()
    plan = round_plan(lock, args.round)
    folder = PRIVATE / "rounds" / f"round-{args.round}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "review.md").exists() and any(
            v["verdict"] for v in parse_review((folder / "review.md").read_text(encoding="utf-8")).values()):
        sys.exit(f"rounds: {folder / 'review.md'} already holds verdicts. Score it, or move it away first.")
    # A new --order draws a new item order, for example when the key has been seen.
    rng = random.Random(f"review-{args.round}-{lock['seed']}-{args.order}")
    order = list(range(len(plan)))
    rng.shuffle(order)
    notes = reviewer_notes(folder)
    review = [f"# Review, round {args.round}", "", INSTRUCTIONS]
    if len({t for t, _ in plan}) < len(plan):
        review += ["Each task appears twice, with two different texts. Judge each text on its own, as if it were "
                   "the only one.", ""]
    key = {"round": args.round, "made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "order": args.order,
           "items": []}
    for number, index in enumerate(order, 1):
        task_id, condition = plan[index]
        task = tasks[task_id]
        text = round_text(args.round, task_id, condition)
        item = f"{number:02d}"
        review += ["---", "", f"## Item {item}", ""] + task_lines(task, notes.get(task_id))
        review += ["", f"**Text {item}:**", "", text, "", f"**Your verdict on {item}:**", "",
                   "Verdict: ", "Reason: ", "AI: ", ""]
        key["items"].append({"item": item, "task": task_id, "condition": condition, "family": task["family"],
                             "text_sha1": hashlib.sha1(text.encode()).hexdigest()})
    (folder / "review.md").write_text("\n".join(review), encoding="utf-8")
    (folder / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"rounds: wrote {len(plan)} items to {folder / 'review.md'}")


PAIR_INSTRUCTIONS = """These are the texts from the review, shown in pairs. Both texts in a pair answer the same task,
and which one is A was decided at random. For each pair, fill in two lines.

- Better: A or B, or "same" if neither is better.
- Reason: why, in a few words. You may leave it blank.

Save the file when you are done and tell Claude.
"""


def pairs(args):
    """A second review file that shows the two texts for each task side by side."""
    lock = load_lock()
    tasks = load_tasks()
    plan = round_plan(lock, args.round)
    ids = list(dict.fromkeys(t for t, _ in plan))
    conditions = sorted({c for _, c in plan})
    if len(conditions) != 2 or len(plan) != 2 * len(ids):
        sys.exit(f"rounds: round {args.round} does not write each task in two conditions.")
    folder = PRIVATE / "rounds" / f"round-{args.round}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "pairs.md"
    if path.exists() and any(c["better"] for c in parse_pairs(path.read_text(encoding="utf-8")).values()):
        sys.exit(f"rounds: {path} already holds choices. Score it, or move it away first.")
    rng = random.Random(f"pairs-{args.round}-{lock['seed']}-{args.order}")
    rng.shuffle(ids)
    # Each condition is text A in half of the pairs, so a habit of preferring A or B favours neither.
    first = [conditions[i % 2] for i in range(len(ids))]
    rng.shuffle(first)
    notes = reviewer_notes(folder)
    review = [f"# Pairs, round {args.round}", "", PAIR_INSTRUCTIONS]
    key = {"round": args.round, "made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "order": args.order,
           "pairs": []}
    for number, (task_id, a) in enumerate(zip(ids, first), 1):
        b = conditions[1 - conditions.index(a)]
        review += ["---", "", f"## Pair {number}", ""] + task_lines(tasks[task_id], notes.get(task_id))
        for label, condition in (("A", a), ("B", b)):
            review += ["", f"**Text {label}:**", "", round_text(args.round, task_id, condition)]
        review += ["", f"**Your choice for pair {number}:**", "", "Better: ", "Reason: ", ""]
        key["pairs"].append({"pair": number, "task": task_id, "family": tasks[task_id]["family"], "A": a, "B": b})
    path.write_text("\n".join(review), encoding="utf-8")
    (folder / "pairs_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"rounds: wrote {len(ids)} pairs to {path}")


# ---------------------------------------------------------------- score

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_verdict(text):
    """'approve', 'approve_with_edits', 'reject' or None, from the way Yili writes verdicts:
    1 / 0, approve / reject, or a sentence such as 'will use but with some edits, ...'."""
    t = text.lower().strip()
    if not t:
        return None
    edits = "edit" in t
    if t.startswith(("1", "approve", "yes", "will use", "use")) or edits:
        return "approve_with_edits" if edits else "approve"
    if t.startswith(("0", "reject", "no", "would not", "won't")):
        return "reject"
    return None


def read_ai(text):
    """True, False, or None when blank or unsure."""
    t = text.lower().strip()
    if not t or "sure" in t:
        return None
    if t.startswith(("1", "y")):
        return True
    if t.startswith(("0", "n")):
        return False
    return None


def parse_review(text):
    verdicts = {}
    parts = re.split(r"^## Item (\d+)\s*$", text, flags=re.M)
    for item, body in zip(parts[1::2], parts[2::2]):
        def field(name):
            # The verdict lines follow the text, so the last match is Yili's, whatever the text contains.
            matches = re.findall(rf"^{name}:[ \t]*(.*)$", body, flags=re.M)
            return matches[-1].strip() if matches else ""
        verdict_text, reason = field("Verdict"), field("Reason")
        # A verdict written as a sentence carries its own reason, as in "will use but ..., too many commas".
        extra = re.sub(r"^(1|0|approve|reject)\b\s*", "", verdict_text, flags=re.I).strip()
        if extra and len(extra.split()) > 3:
            reason = f"{extra} {reason}".strip()
        verdicts[item] = {"verdict": read_verdict(verdict_text), "reason": reason,
                          "ai": read_ai(field("AI")), "ai_answer": field("AI")}
    return verdicts


def parse_pairs(text):
    """For each pair, 'A', 'B', 'same' or None, with the reason. A choice may carry its reason, as in 'A, shorter'."""
    choices = {}
    parts = re.split(r"^## Pair (\d+)\s*$", text, flags=re.M)
    for pair, body in zip(parts[1::2], parts[2::2]):
        answers = re.findall(r"^Better:[ \t]*(.*)$", body, flags=re.M)
        reasons = re.findall(r"^Reason:[ \t]*(.*)$", body, flags=re.M)
        answer = answers[-1].strip() if answers else ""
        reason = reasons[-1].strip() if reasons else ""
        letter = re.match(r"(?i)(?:text\s+)?([ab])\b[\s,.:;-]*(.*)", answer)
        if letter:
            better, extra = letter.group(1).upper(), letter.group(2).strip()
        elif re.search(r"(?i)same|equal|neither|tie|both", answer):
            better, extra = "same", ""
        else:
            better, extra = None, ""
        choices[int(pair)] = {"better": better, "answer": answer, "reason": f"{extra} {reason}".strip()}
    return choices


def score_pairs(folder):
    """Lines for the summary: how often each condition's text was preferred, and why."""
    key = json.loads((folder / "pairs_key.json").read_text(encoding="utf-8"))
    choices = parse_pairs((folder / "pairs.md").read_text(encoding="utf-8"))
    counts, details, rows = {}, [], []
    for p in key["pairs"]:
        choice = choices.get(p["pair"], {})
        preferred = p[choice["better"]] if choice.get("better") in ("A", "B") else choice.get("better") or "no answer"
        counts[preferred] = counts.get(preferred, 0) + 1
        rows.append({**p, **choice, "preferred": preferred})
        details.append(f"- pair {p['pair']} {p['task']}: {preferred}" + (f". {choice['reason']}" if choice.get("reason") else ""))
    with open(folder / "pairs_scores.jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    return ["", "## Pairs", "", "Preferred: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), ""] + details


def rate(rows, test):
    k = sum(1 for r in rows if test(r))
    n = len(rows)
    low, high = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% CI {low:.0%} to {high:.0%})" if n else "none"


def score(args):
    folder = PRIVATE / "rounds" / f"round-{args.round}"
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    verdicts = parse_review((folder / "review.md").read_text(encoding="utf-8"))
    coverage_path = PRIVATE / "tasks_private" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    rows, missing = [], []
    for item in key["items"]:
        v = verdicts.get(item["item"], {})
        if v.get("verdict") is None:
            missing.append(item["item"])
        rows.append({**item, **v, "covered": coverage.get(item["task"])})
    with open(folder / "scores.jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    done = [r for r in rows if r.get("verdict")]
    lines = [f"# Round {args.round} scores", ""]
    if missing:
        lines += [f"Items without a verdict: {', '.join(missing)}", ""]
    groups = [("all", done)]
    groups += [(f"condition {c}", [r for r in done if r["condition"] == c]) for c in ("skill", "baseline")]
    groups += [(f"family {f}", [r for r in done if r["family"] == f]) for f in FAMILIES]
    groups += [(f"covered {c}", [r for r in done if r["covered"] == c]) for c in (True, False)]
    for name, group in groups:
        if group:
            answered = [r for r in group if r.get("ai") is not None]
            lines.append(f"- {name}: used as is {rate(group, lambda r: r['verdict'] == 'approve')}; "
                         f"used at all {rate(group, lambda r: r['verdict'] != 'reject')}; "
                         f"AI suspected {rate(answered, lambda r: r['ai'])} of those answered "
                         f"({len(group) - len(answered)} blank or unsure)")
    lines += ["", "## Reasons", ""]
    for r in rows:
        if r.get("reason"):
            lines.append(f"- {r['item']} {r['task']} {r['condition']} {r['verdict']}: {r['reason']}")
    if (folder / "pairs.md").exists():
        lines += score_pairs(folder)
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- mechanical checks

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Asides: dashes, semicolons, colons that introduce an explanation, and brackets other than years,
# panel labels such as (a) or (b-d), and references to tables, figures or sections.
ASIDE = re.compile(r"—|\s--\s|;|:\s(?=[a-z])|\s\((?!\d{4}\)|[a-z](?:[–-][a-z])?\)|"
                   r"(?:Table|Figure|Fig\.|Section|Supplementary|Appendix)\b)")
BOOST = re.compile(r"\b(crucial|robust|significant(?:ly)?|novel|comprehensive|notably|pivotal|seamless(?:ly)?|"
                   r"cutting-edge|state-of-the-art|unprecedented|groundbreaking|transformative|innovative|"
                   r"extremely|highly|remarkabl[ey]|vital|key insights?|leverag\w*|delv\w*|showcas\w*|underscor\w*)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|possibly|potentially|arguably|perhaps|somewhat|relatively)\b", re.I)


def shape(text):
    """Counts that describe how a text reads, per sentence or per 100 words."""
    body = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", text, flags=re.M)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", " ".join(body.split())) if len(s.split()) > 2]
    words = max(1, len(body.split()))
    lengths = [len(s.split()) for s in sentences] or [0]
    mean = sum(lengths) / len(lengths)
    sd = math.sqrt(sum((x - mean) ** 2 for x in lengths) / len(lengths))
    return {"words": words, "sentences": len(sentences), "words_per_sentence": round(mean, 1), "sd": round(sd, 1),
            "commas_per_sentence": round(body.count(",") / max(1, len(sentences)), 2),
            "asides_per_100w": round(100 * len(ASIDE.findall(body)) / words, 2),
            "boosters_per_100w": round(100 * len(BOOST.findall(body)) / words, 2),
            "hedges_per_100w": round(100 * len(HEDGE.findall(body)) / words, 2)}


def check(args):
    tasks = load_tasks()
    rows = []
    for run in sorted((PRIVATE / "results" / args.label).glob("*/meta.json")):
        meta = json.loads(run.read_text(encoding="utf-8"))
        task = tasks[meta["task"]]
        text = (run.parent / "text.md").read_text(encoding="utf-8")
        source = " ".join([task["prompt"], *task["facts"], *task.get("materials", {}).values()])
        source_numbers = set(NUMBER.findall(source))
        unsourced = sorted({n for n in NUMBER.findall(text) if n not in source_numbers and not re.fullmatch(r"[1-9]", n)})
        limit = task.get("limit_words")
        fired = any("write-like-myself" in (s or "") for s in meta.get("skills", []))
        rows.append({"task": meta["task"], "condition": meta["condition"], "fired": fired,
                     "over_limit": bool(limit and len(text.split()) > limit), "unsourced_numbers": unsourced,
                     **shape(text)})
    (PRIVATE / "results" / args.label / "checks.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    approved = [(PRIVATE / "results" / "round-0" / f"{t}-baseline" / "text.md").read_text(encoding="utf-8")
                for t in args.approved.split(",")]
    reference = shape("\n\n".join(approved))
    print(f"rounds: {len(rows)} texts checked. Reference, texts Yili used as is: {reference}")
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        mean = lambda k: round(sum(r[k] for r in group) / len(group), 2)
        print(f"- {condition} ({len(group)} texts): skill fired {sum(r['fired'] for r in group)}, "
              f"over the word limit {sum(r['over_limit'] for r in group)}, "
              f"with unsourced numbers {sum(bool(r['unsourced_numbers']) for r in group)}; "
              f"words/sentence {mean('words_per_sentence')}, commas/sentence {mean('commas_per_sentence')}, "
              f"asides/100w {mean('asides_per_100w')}, boosters/100w {mean('boosters_per_100w')}, "
              f"hedges/100w {mean('hedges_per_100w')}")
    for r in rows:
        if r["unsourced_numbers"] or r["over_limit"]:
            print(f"  {r['task']} {r['condition']}: over limit {r['over_limit']}, unsourced {r['unsourced_numbers']}")


# ---------------------------------------------------------------- calibrate

GRADER = "claude-sonnet-5-5"
WORKFLOW = ("ui problem", "vscode", "vs code", "auto approved", "auto-approved")


def plain(text):
    """Manuscript text without LaTeX markup, comments or citations."""
    text = re.sub(r"(?<!\\)%[^\n]*", " ", text or "")
    text = re.sub(r"\\(?:cite[pt]?|ref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text).replace("{", "").replace("}", "").replace("~", " ")
    return " ".join(text.split())


def calibration_pairs(limit):
    """(approved, rejected) pairs from training chains not used for the shipped examples."""
    records = [json.loads(line) for line in open(PRIVATE / "corpus" / "records.jsonl", encoding="utf-8")]
    excluded = set(json.loads((PRIVATE / "corpus" / "illustration_chains.json").read_text(encoding="utf-8")))
    chains = {}
    for r in records:
        if r.get("chain"):
            chains.setdefault(r["chain"], []).append(r)
    pairs, seen = [], set()
    for chain in sorted(chains):
        group = sorted(chains[chain], key=lambda r: r.get("time") or "")
        if group[0]["split"] != "train" or chain in excluded:
            continue
        for i, r in enumerate(group):
            reason = (r.get("reason") or "").lower()
            if r["kind"] != "edit_rejected" or not reason or any(w in reason for w in WORKFLOW):
                continue
            if "keep" in reason or "original" in reason or "old version" in reason:
                approved = r.get("before")
            else:
                later = [x for x in group[i + 1:] if x["kind"] == "edit_accepted"]
                approved = later[0]["after"] if later else None
            a, b = plain(approved), plain(r.get("after"))
            if a and b and a != b and 6 <= len(a.split()) <= 250 and len(b.split()) >= 6 and (a, b) not in seen:
                seen.add((a, b))
                pairs.append({"chain": chain, "genre": r.get("genre"), "approved": a, "rejected": b})
    return pairs[:limit]


def calibrate(args):
    pairs = calibration_pairs(args.pairs)
    rng = random.Random(args.seed)
    lines, key = [], []
    for n, pair in enumerate(pairs, 1):
        approved_is_a = rng.random() < 0.5
        a, b = (pair["approved"], pair["rejected"]) if approved_is_a else (pair["rejected"], pair["approved"])
        key.append("A" if approved_is_a else "B")
        lines += [f"Pair {n} ({pair['genre']})", f"A: {a}", f"B: {b}", ""]
    prompt = ("An author, a scientist, reviewed proposed edits to a journal manuscript. In each pair below, the "
              "author approved one version and rejected the other. Decide which version the author approved.\n\n"
              "The author's own writing rules, for reference:\n\n"
              + (PRIVATE / "tasks_private" / "baseline_CLAUDE.md").read_text(encoding="utf-8")
              + "\nAnswer with JSON only: a list of objects such as {\"pair\": 1, \"approved\": \"A\"}.\n\n"
              + "\n".join(lines))
    folder = PRIVATE / "results" / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    result = subprocess.run([CLAUDE, "-p", "--model", GRADER, "--setting-sources", "project,local",
                             "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"],
                            cwd=folder, env=env, input=prompt, text=True, encoding="utf-8", capture_output=True,
                            timeout=TIMEOUT)
    answer = json.loads(result.stdout).get("result", "")
    choices = {item["pair"]: item["approved"] for item in json.loads(re.search(r"\[.*\]", answer, re.S).group())}
    right = sum(1 for n, k in enumerate(key, 1) if choices.get(n) == k)
    trusted = right >= args.threshold * len(key) / 20
    (folder / "calibration.json").write_text(json.dumps(
        {"grader": GRADER, "pairs": pairs, "key": key, "choices": choices, "right": right, "of": len(key),
         "trusted": trusted}, indent=2), encoding="utf-8")
    print(f"rounds: grader {GRADER} picked Yili's choice in {right} of {len(key)} pairs. "
          f"{'Trusted' if trusted else 'Not trusted'} (threshold {args.threshold} of 20).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--label", default="dev")
    p.add_argument("--approved", default="C10,D03,B05,C05,B10,D10",
                   help="round-0 tasks whose baseline text Yili used, as the reference for the counts")
    p = sub.add_parser("calibrate")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--threshold", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("split")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("--round", type=int)
    p.add_argument("--set", default="dev")
    p.add_argument("--tasks")
    p.add_argument("--condition", default="baseline")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--label", default="dev")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--plugin", help="a plugin folder for the skill condition, in place of this repository's")
    p = sub.add_parser("make")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--order", type=int, default=0, help="draws a different item order")
    p = sub.add_parser("pairs")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--order", type=int, default=0, help="draws a different pair order")
    p = sub.add_parser("score")
    p.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    {"split": split, "generate": generate, "make": make, "pairs": pairs, "score": score, "check": check,
     "calibrate": calibrate}[args.command](args)


if __name__ == "__main__":
    main()
