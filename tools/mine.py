#!/usr/bin/env python3
"""Build the private corpus of Yili's writing judgements.

    python -X utf8 tools/mine.py                 mine every source named in the private config
    python -X utf8 tools/mine.py --report        counts, plus spot checks of random records against their sources
    python -X utf8 tools/mine.py --fetch-own     download Yili's published abstracts and thesis into own/public
    python -X utf8 tools/mine.py --live --since 2026-11-01
                                                 approval of prose edits in newer sessions where the skill fired

Sources, all named in the private config:
- Claude Code transcripts: each proposed edit with Yili's verdict, typed reason and review time; Yili's typed
  messages, plan comments and answers to questions.
- Annotated PDFs: each mark with Yili's comment and the text under it.
- Git history: Yili's own edits, paired sentence by sentence.
- Yili's own texts: published abstracts, the thesis, older CVs, sent emails and Drive drafts.

Records go to corpus/records.jsonl in the private folder. Edits are grouped into chains (an edit and its retries
on the same passage); about a quarter of the chains that hold a reasoned rejection are held back from the skill,
chosen by a fixed hash, and the split is frozen in corpus/split.json.
"""
import argparse
import glob
import hashlib
import json
import random
import re
import subprocess
import sys
import tomllib
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = Path(tomllib.loads((REPO / "tools" / "local.toml").read_text(encoding="utf-8"))["private_dir"]).expanduser()
CONFIG = tomllib.loads((PRIVATE / "config.toml").read_text(encoding="utf-8"))
CORPUS = PRIVATE / "corpus"

REJECTED = "doesn't want to proceed with this tool use"
REASON = "reason for the rejection:"
STOPPED = "STOP what you are doing"
SYSTEM_TAGS = re.compile(r"<(ide_opened_file|ide_selection|system-reminder|command-name|command-message|command-args|"
                         r"local-command-stdout|local-command-stderr|user-prompt-submit-hook)>.*?</\1>", re.S)
SKIP_TYPED = ("[Request interrupted", "Base directory for this skill:", "This session is being continued",
              "Caveat:", "<")
WORKFLOW_WORDS = ("ui problem", "vs code", "vscode", "auto approv", "auto-approv", "group code", "batch", "one by one",
                  "manual", "approve", "allow all", "interrupt")
GENRES = {"00_abstract": "paper.abstract", "01_introduction": "paper.introduction", "02_study_area": "paper.data",
          "03_methods": "paper.methods", "04_results": "paper.results", "05_discussion": "paper.discussion",
          "06_conclusions": "paper.conclusions", "07_backmatter": "paper.backmatter", "main.tex": "paper.front",
          "supplementary": "paper.supplement", "cover_letter": "letter.cover", "author_statement": "paper.statement",
          "docs/": "working_doc", "README": "working_doc", "CLAUDE.md": "working_doc"}


# ---------------------------------------------------------------- helpers

def tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def overlap(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def same_passage(a, b):
    """True when two texts are versions of the same passage: broadly similar, or one long text mostly inside
    the other. Short snippets never link on containment alone, or whole files would merge into one chain."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    contained = len(ta & tb) / min(len(ta), len(tb))
    small, large = sorted((len(ta), len(tb)))
    # Containment links only texts of comparable size: a rewrite of a whole subsection contains many
    # separate sentences, and must not tie them all into one passage.
    return jaccard >= 0.5 or (contained >= 0.8 and small >= 12 and large <= 2 * small)


def genre_of(path):
    path = (path or "").replace("\\", "/")
    for key, genre in GENRES.items():
        if key in path:
            return genre
    return "other"


def relative(path):
    """A transcript or git path relative to the manuscript repository."""
    path = (path or "").replace("\\", "/")
    marker = Path(CONFIG["sources"]["rtsv2_repo"]).name + "/"
    return path.split(marker, 1)[1] if marker in path else path


def phase_of(time):
    """The writing phase of RTSV2 at a given time, since cutting to a word limit shapes many judgements."""
    day = (time or "")[:10]
    if day >= "2026-10-13":
        return "length_cut"
    if day >= "2026-09-19":
        return "journal_audit"
    return "drafting"


def seconds_between(start, end):
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return round((b - a).total_seconds(), 1)
    except (AttributeError, ValueError):
        return None


def result_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content or [] if isinstance(block, dict))


def expand_to_lines(original, old, new):
    """Widen an edit to whole lines, using the file as it was before the edit (prose is one sentence per line)."""
    if not original or not old or old not in original:
        return old, new
    start = original.rfind("\n", 0, original.index(old)) + 1
    end_index = original.index(old) + len(old)
    end = original.find("\n", end_index)
    end = len(original) if end == -1 else end
    prefix, suffix = original[start:original.index(old)], original[end_index:end]
    return prefix + old + suffix, prefix + new + suffix


def git(repo, *args):
    repo = Path(repo)
    result = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
                            capture_output=True)
    if result.returncode != 0:
        sys.exit(f"mine: git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8", errors="replace")


def expand_globs(patterns):
    files = []
    for pattern in patterns:
        files += sorted(glob.glob(str(Path(pattern).expanduser())))
    return [Path(f) for f in files]


# ---------------------------------------------------------------- transcripts

def read_jsonl(path):
    """Records of one transcript, tolerating a last line that is still being written."""
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def mine_transcript(path, cutoff):
    session = path.stem
    uses, out, seen_typed = {}, [], set()
    mode = None
    for rec in read_jsonl(path):
        if rec.get("isSidechain") or (rec.get("timestamp") or "") > cutoff:
            continue
        mode = rec.get("permissionMode") or mode
        message = rec.get("message") or {}
        content = message.get("content")
        if rec.get("type") == "assistant":
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    uses[block["id"]] = {"name": block.get("name"), "input": block.get("input") or {},
                                         "time": rec.get("timestamp")}
            continue
        if rec.get("type") != "user":
            continue
        if isinstance(content, str) or (isinstance(content, list) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content)):
            text = content if isinstance(content, str) else result_text(content)
            text = SYSTEM_TAGS.sub("", text).strip()
            if text and not text.startswith(SKIP_TYPED) and not rec.get("isMeta") and rec.get("uuid") not in seen_typed:
                seen_typed.add(rec.get("uuid"))
                out.append({"id": f"rtsv2/typed/{rec.get('uuid')}", "kind": "typed", "tier": 1, "text": text,
                            "time": rec.get("timestamp"), "provenance": {"session": session}})
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            use = uses.get(block.get("tool_use_id"))
            if use is None:
                continue
            text = result_text(block.get("content"))
            out += records_for_result(session, block["tool_use_id"], use, text, rec, mode)
    return out


def records_for_result(session, use_id, use, text, rec, mode):
    name, given = use["name"], use["input"]
    base = {"time": rec.get("timestamp"), "provenance": {"session": session, "tool_use_id": use_id},
            "review": {"mode": mode, "latency_s": seconds_between(use["time"], rec.get("timestamp"))}}
    rejected = REJECTED in text
    reason = text.split(REASON, 1)[1].strip() if REASON in text else None
    if name == "AskUserQuestion" and ("questions have been answered" in text or "The user answered" in text):
        return [{**base, "id": f"rtsv2/answer/{use_id}", "kind": "answer", "tier": 1, "text": text,
                 "question": json.dumps(given.get("questions", []))[:4000]}]
    if name == "ExitPlanMode" and (rejected or "Comments on the plan" in text):
        return [{**base, "id": f"rtsv2/plan/{use_id}", "kind": "plan_comment", "tier": 1,
                 "text": reason or text}]
    if name not in ("Edit", "MultiEdit", "Write"):
        if rejected and reason:
            return [{**base, "id": f"rtsv2/other/{use_id}", "kind": "rejection_other", "tier": 1, "tool": name,
                     "reason": reason}]
        return []
    path = relative(given.get("file_path"))
    # Only files of the manuscript repository: Claude's own plan files are not prose for readers.
    if not path.endswith((".tex", ".md")) or re.match(r"^[A-Za-z]:|^/|\.claude", path):
        return []
    result = rec.get("toolUseResult") if isinstance(rec.get("toolUseResult"), dict) else {}
    edits = given.get("edits") if name == "MultiEdit" else [given]
    out = []
    for number, edit in enumerate(edits):
        before = edit.get("old_string") if name != "Write" else None
        after = edit.get("new_string") if name != "Write" else given.get("content")
        if result.get("originalFile") and name == "Edit":
            before, after = expand_to_lines(result["originalFile"], before, after)
        kind = "edit_rejected" if rejected or STOPPED in text else "edit_accepted"
        out.append({**base, "id": f"rtsv2/edit/{use_id}" + (f"/{number}" if len(edits) > 1 else ""),
                    "kind": kind, "tier": 3 if kind == "edit_rejected" else 2, "file": path,
                    "genre": genre_of(path), "phase": phase_of(rec.get("timestamp")), "before": before,
                    "after": after, "new_core": (edit.get("new_string") or given.get("content") or "")[:80],
                    "reason": reason if kind == "edit_rejected" else None,
                    "bare_rejection": kind == "edit_rejected" and not reason})
    return out


# ---------------------------------------------------------------- annotated PDFs

def mine_pdfs(paths):
    import fitz
    out = []
    for pdf in paths:
        doc = fitz.open(pdf)
        for page in doc:
            spans = [(fitz.Rect(span["bbox"]), span["text"], span["color"])
                     for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])
                     for span in line["spans"]]
            for annot in page.annots() or []:
                info = annot.info
                vertices = annot.vertices or []
                rects = [fitz.Quad(vertices[i:i + 4]).rect for i in range(0, len(vertices) - 3, 4)] or [annot.rect]
                kept, deleted = [], []
                for rect, text, colour in spans:
                    if any(rect.intersects(r) and (rect & r).get_area() > 0.3 * rect.get_area() for r in rects):
                        (deleted if colour == 0xFF0000 else kept).append(text)
                out.append({"id": f"pdf/{Path(pdf).stem}/{page.number + 1}/{annot.xref}", "kind": "pdf_mark",
                            "tier": 1, "mark": annot.type[1], "author": info.get("title"),
                            "comment": (info.get("content") or "").strip() or None,
                            "marked_text": " ".join(kept).strip() or None,
                            "deleted_text": " ".join(deleted).strip() or None,
                            "time": info.get("modDate") or info.get("creationDate"),
                            "provenance": {"pdf": Path(pdf).name, "page": page.number + 1}})
    return out


# ---------------------------------------------------------------- git

def blame_authors(repo, commit, path):
    """Line number -> 'claude', 'yili' or 'unknown' for a file as it was at a commit."""
    try:
        porcelain = git(repo, "blame", "--line-porcelain", commit, "--", path)
    except SystemExit:
        return {}
    authors, commits_seen, line_no, current = {}, {}, 0, None
    for line in porcelain.splitlines():
        header = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if header:
            current, line_no = header.group(1), int(header.group(2))
        elif line.startswith("author ") and current:
            commits_seen.setdefault(current, line[7:])
        elif line.startswith("\t") and current:
            authors[line_no] = current
    kinds = {}
    for sha in set(authors.values()):
        message = git(repo, "log", "-1", "--format=%an%n%B", sha)
        name = message.splitlines()[0] if message else ""
        kinds[sha] = "claude" if "Co-Authored-By: Claude" in message else "yili" if name == "YILI YANG" else "unknown"
    return {n: kinds[sha] for n, sha in authors.items()}


def hunks(diff):
    """(file, removed [(old line no, text)], added [text]) for each hunk of a zero-context diff."""
    path, old_no = None, 0
    removed, added, out = [], [], []
    for line in diff.splitlines() + ["@@ end"]:
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else path
        elif line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else path
        elif line.startswith("@@"):
            if removed or added:
                out.append((path, removed, added))
            removed, added = [], []
            match = re.match(r"@@ -(\d+)", line)
            old_no = int(match.group(1)) if match else 0
        elif line.startswith("-"):
            removed.append((old_no, line[1:]))
            old_no += 1
        elif line.startswith("+"):
            added.append(line[1:])
    return out


def pair_lines(removed, added):
    """Pair removed and added sentences by word overlap; leftovers become pure deletions or additions."""
    pairs, free = [], list(range(len(added)))
    for old_no, old in removed:
        best = max(free, key=lambda i: overlap(old, added[i]), default=None)
        if best is not None and overlap(old, added[best]) >= 0.3:
            pairs.append((old_no, old, added[best]))
            free.remove(best)
        else:
            pairs.append((old_no, old, None))
    pairs += [(None, None, added[i]) for i in free]
    return pairs


def mine_git(repo, paths, prompt_to_think):
    out = []
    log = git(repo, "log", "--format=%H%x09%an%x09%aI%x09%s", "--", *paths)
    for entry in log.splitlines():
        sha, author, date, subject = entry.split("\t", 3)
        body = git(repo, "log", "-1", "--format=%B", sha)
        own = author == "YILI YANG" or "Yili's own" in body or "Yili's wording" in subject
        if not own:
            continue
        # Co-authors edited in Overleaf too (Yili confirmed), so a sync may carry their edits as well as
        # Yili's. Named hand-edit commits are Yili's own.
        synced = subject.startswith("Updates from Overleaf")
        assist = "grammarly" if "grammarly" in body.lower() else "overleaf_sync" if synced else None
        parents = git(repo, "log", "-1", "--format=%P", sha).split()
        if len(parents) != 1:
            continue
        diff = git(repo, "show", sha, "-U0", "--format=", "--no-color", "--", *paths)
        blame_cache = {}
        for path, removed, added in hunks(diff):
            if not path or not path.endswith((".tex", ".md")):
                continue
            if path not in blame_cache:
                blame_cache[path] = blame_authors(repo, parents[0], path)
            for number, (old_no, before, after) in enumerate(pair_lines(removed, added)):
                if (before or "").strip() == (after or "").strip():
                    continue
                out.append({"id": f"git/{sha[:10]}/{path}/{old_no or 'add'}/{number}", "kind": "hand_edit",
                            "tier": 1, "file": path, "genre": genre_of(path), "phase": phase_of(date),
                            "time": date, "before": before, "after": after,
                            "before_author": blame_cache[path].get(old_no, "unknown") if old_no else None,
                            "assist": assist, "author_certain": not synced,
                            "provenance": {"commit": sha, "subject": subject}})
    if prompt_to_think:
        diff = git(prompt_to_think, "show", "3c6352c", "-U0", "--format=", "--no-color")
        for path, removed, added in hunks(diff):
            for number, (old_no, before, after) in enumerate(pair_lines(removed, added)):
                if before and after and before.strip() != after.strip():
                    out.append({"id": f"git/prompt-to-think/3c6352c/{old_no}/{number}", "kind": "hand_edit",
                                "tier": 1, "file": path, "genre": "skill_text", "time": None, "before": before,
                                "after": after, "before_author": "unknown", "assist": None,
                                "provenance": {"commit": "3c6352c", "repo": "prompt-to-think"}})
    return out


# ---------------------------------------------------------------- own texts

OWN_PUBLIC = {
    "rse2023_abstract": "10.1016/j.rse.2023.113495",
    "scidata2025_abstract": "10.1038/s41597-025-04372-7",
    "jag2026_abstract": "10.1016/j.jag.2026.105187",
    "egu2024_abstract": "10.5194/egusphere-egu24-1365",
    "essoar2021_abstract": "10.1002/essoar.10507074.1",
}
OWN_DATACITE = {"arts2024_dataset_abstract": "10.18739/A2PK0738B"}
THESIS_URL = "https://era.ed.ac.uk/bitstreams/edacdc18-8cd2-4305-b99f-0b74cb52d2e9/download"


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "write-like-myself corpus builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_own():
    folder = PRIVATE / "own" / "public"
    for name, doi in OWN_PUBLIC.items():
        try:
            work = fetch_json(f"https://api.openalex.org/works/https://doi.org/{doi}")
            index = work.get("abstract_inverted_index") or {}
            positions = sorted((p, word) for word, ps in index.items() for p in ps)
            text = " ".join(word for _, word in positions)
            year = work.get("publication_year")
        except Exception as error:  # a missing abstract is reported, not fatal
            text, year = "", None
            print(f"mine: no abstract for {name}: {error}")
        if text:
            (folder / f"{name}.txt").write_text(f"source: doi:{doi}\nyear: {year}\n\n{text}\n", encoding="utf-8")
            print(f"mine: saved {name} ({len(text.split())} words)")
    for name, doi in OWN_DATACITE.items():
        try:
            record = fetch_json(f"https://api.datacite.org/dois/{doi}")["data"]["attributes"]
            text = "\n\n".join(d.get("description", "") for d in record.get("descriptions", []))
            (folder / f"{name}.txt").write_text(f"source: doi:{doi}\nyear: {record.get('publicationYear')}\n\n"
                                                f"{text}\n", encoding="utf-8")
            print(f"mine: saved {name} ({len(text.split())} words)")
        except Exception as error:
            print(f"mine: no description for {name}: {error}")
    thesis = PRIVATE / "own" / "thesis.pdf"
    if not thesis.exists():
        request = urllib.request.Request(THESIS_URL, headers={"User-Agent": "write-like-myself corpus builder"})
        with urllib.request.urlopen(request, timeout=600) as response:
            thesis.write_bytes(response.read())
    import fitz
    doc = fitz.open(thesis)
    text = "\n".join(page.get_text() for page in doc)
    (folder / "thesis2021_fulltext.txt").write_text(f"source: doi:10.7488/era/1530\nyear: 2021\n\n{text}",
                                                    encoding="utf-8")
    print(f"mine: saved thesis text ({len(text.split())} words, {doc.page_count} pages)")


def mine_own():
    out = []
    for folder, public in ((PRIVATE / "own" / "public", True), (PRIVATE / "own" / "private", False),
                           (PRIVATE / "mail", False), (PRIVATE / "drive", False)):
        for path in sorted(folder.glob("*.txt")):
            raw = path.read_text(encoding="utf-8", errors="replace")
            header, _, body = raw.partition("\n\n")
            meta = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
            genre = "email" if folder.name == "mail" else meta.get("genre", "own")
            out.append({"id": f"own/{folder.name}/{path.stem}", "kind": "own_text", "tier": 1, "genre": genre,
                        "public": public, "time": meta.get("date") or meta.get("year"), "text": body.strip(),
                        "provenance": {"file": f"{folder.name}/{path.name}", **meta}})
    return out


# ---------------------------------------------------------------- chains and split

def chain_and_split(records):
    edits = [r for r in records if r["kind"] in ("edit_rejected", "edit_accepted", "hand_edit")]
    parent = {r["id"]: r["id"] for r in edits}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_file = defaultdict(list)
    for r in edits:
        by_file[r.get("file")].append(r)
    for group in by_file.values():
        group.sort(key=lambda r: r.get("time") or "")
        for j, later in enumerate(group):
            for earlier in group[max(0, j - 400):j]:
                if same_passage(later.get("before"), earlier.get("before")) or \
                        same_passage(later.get("before"), earlier.get("after")):
                    parent[find(later["id"])] = find(earlier["id"])
    members = defaultdict(list)
    for r in edits:
        members[find(r["id"])].append(r)
    split_path = CORPUS / "split.json"
    frozen = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    for root, group in members.items():
        earliest = min(group, key=lambda r: (r.get("time") or "", r["id"]))
        chain = hashlib.sha1(earliest["id"].encode()).hexdigest()[:12]
        reasoned = any(r["kind"] == "edit_rejected" and r.get("reason")
                       and not any(w in r["reason"].lower() for w in WORKFLOW_WORDS) for r in group)
        if chain not in frozen:
            frozen[chain] = "holdout" if reasoned and int(hashlib.sha1(chain.encode()).hexdigest(), 16) % 4 == 0 \
                else "train"
        for r in group:
            r["chain"], r["split"] = chain, frozen[chain]
    split_path.write_text(json.dumps(frozen, indent=1, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------- denylist numbers

def seed_denylist_numbers():
    """Add the unpublished decimal numbers of the manuscript to the private denylist."""
    repo = Path(CONFIG["sources"]["rtsv2_repo"])
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in list((repo / "sections").glob("*.tex")) + [repo / "supplementary.tex"])
    # Numbers already published (own abstracts, the website and its CV data) are not secrets.
    allowed = []
    for pattern in CONFIG.get("index", {}).get("allow_dirs", []):
        root = Path(pattern).expanduser()
        root = root if root.is_absolute() else PRIVATE / pattern
        allowed += [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else \
            [Path(p) for p in glob.glob(str(root))]
    public = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in allowed)
    numbers = {n for n in re.findall(r"(?<![\d.])\d+\.\d+(?![\d.])", text)
               if len(n.replace(".", "").lstrip("0")) >= 3 and n not in public}
    path = PRIVATE / "denylist.txt"
    marker = "# Unpublished decimal numbers (seeded by tools/mine.py)"
    kept = path.read_text(encoding="utf-8").split(marker)[0].rstrip("\n")
    path.write_text(kept + "\n" + marker + "\n" + "\n".join(sorted(numbers)) + "\n", encoding="utf-8")
    return len(numbers)


# ---------------------------------------------------------------- main commands

def mine():
    sources = CONFIG["sources"]
    cutoff = sources.get("cutoff", "9999") + "T23:59:59Z"
    records = []
    for path in expand_globs(sources["transcripts"]):
        records += mine_transcript(path, cutoff)
    records += mine_pdfs(expand_globs(sources["annotated_pdfs"]))
    repo = sources["rtsv2_repo"]
    records += mine_git(repo, ["sections", "main.tex", "supplementary.tex", "jag", "isprs", "rse"],
                        sources.get("prompt_to_think_repo"))
    records += mine_own()
    unique = {}
    for r in records:
        unique.setdefault(r["id"], r)
    records = list(unique.values())
    chain_and_split(records)
    CORPUS.mkdir(exist_ok=True)
    with open(CORPUS / "records.jsonl", "w", encoding="utf-8") as out:
        for r in sorted(records, key=lambda r: (r.get("time") or "", r["id"])):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    seeded = seed_denylist_numbers()
    print(f"mine: wrote {len(records)} records; seeded {seeded} unpublished numbers into the denylist.")
    report()


def load_records():
    return [json.loads(line) for line in open(CORPUS / "records.jsonl", encoding="utf-8")]


def report():
    records = load_records()
    kinds = Counter(r["kind"] for r in records)
    section_edits = [r for r in records if r["kind"] in ("edit_accepted", "edit_rejected")
                     and (r.get("file") or "").startswith("sections/")]
    rejected = [r for r in section_edits if r["kind"] == "edit_rejected"]
    marks = [r for r in records if r["kind"] == "pdf_mark"]
    manual = [r for r in section_edits if (r["review"].get("latency_s") or 0) >= 10]
    chains = Counter(r["split"] for r in {r["chain"]: r for r in records if r.get("chain")}.values())
    print("mine: records by kind:", dict(kinds))
    print(f"mine: edits proposed to sections/*.tex: {len(section_edits)}; rejected {len(rejected)} "
          f"({sum(1 for r in rejected if r.get('reason'))} with a reason); accepted after at least 10 s of review: "
          f"{sum(1 for r in manual if r['kind'] == 'edit_accepted')}")
    print(f"mine: PDF marks {len(marks)}, with a comment {sum(1 for r in marks if r.get('comment'))}")
    print(f"mine: chains: {dict(chains)}")
    print(f"mine: own texts: {Counter(r['provenance']['file'].split('/')[0] for r in records if r['kind'] == 'own_text')}")
    ids = [r.get("provenance", {}).get("tool_use_id") for r in records if r["kind"].startswith("edit")]
    print(f"mine: duplicate tool_use ids: {len(ids) - len(set(ids))}")
    spot_check(records)


def spot_check(records, n=10):
    """Re-find ten random records in their raw sources."""
    rng = random.Random(7)
    checks = rng.sample([r for r in records if r["kind"] in ("edit_rejected", "edit_accepted", "typed",
                                                              "pdf_mark", "hand_edit")], n)
    transcripts = {p.stem: p for p in expand_globs(CONFIG["sources"]["transcripts"])}
    passed = 0
    for r in checks:
        prov = r.get("provenance", {})
        if r["kind"].startswith("edit") or r["kind"] == "typed":
            raw = transcripts[prov["session"]].read_text(encoding="utf-8", errors="replace")
            probe = (r.get("reason") or r.get("text") or r.get("new_core") or "")[:60]
            found = json.dumps(probe, ensure_ascii=False)[1:-1][:40] in raw and \
                (r["kind"] == "typed" or prov["tool_use_id"] in raw)
        elif r["kind"] == "hand_edit":
            repo = CONFIG["sources"]["prompt_to_think_repo"] if prov.get("repo") else CONFIG["sources"]["rtsv2_repo"]
            found = (r.get("after") or r.get("before") or "")[:40] in git(repo, "show", prov["commit"], "--format=")
        else:
            found = bool(r.get("comment") or r.get("marked_text"))
        passed += found
        print(f"  spot check {r['id'][:60]:60s} {'found' if found else 'NOT FOUND'}")
    print(f"mine: spot checks found {passed} of {n} records in their sources.")


def live(since):
    """Approval of prose edits in sessions after a date where the write-like-myself skill fired."""
    proposals = []
    for path in expand_globs(["~/.claude/projects/*/*.jsonl"]):
        records = read_jsonl(path)
        fired = any(isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Skill"
                    and "write-like-myself" in json.dumps(b.get("input"))
                    for rec in records if rec.get("type") == "assistant"
                    for b in (rec.get("message") or {}).get("content") or [])
        if not fired:
            continue
        for r in mine_transcript(path, "9999"):
            if r["kind"] in ("edit_accepted", "edit_rejected") and (r.get("time") or "") >= since:
                proposals.append(r)
    reviewed = [r for r in proposals if (r["review"].get("latency_s") or 0) >= 10]
    accepted = sum(1 for r in reviewed if r["kind"] == "edit_accepted")
    print(f"mine: {len(proposals)} prose edits since {since} in sessions where the skill fired; "
          f"{accepted} of {len(reviewed)} reviewed for at least 10 s were accepted.")
    for r in proposals:
        if r.get("reason"):
            print(f"  rejected ({r.get('file')}): {r['reason']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--fetch-own", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--since", default="2026-10-21")
    args = parser.parse_args()
    if args.fetch_own:
        fetch_own()
    elif args.report:
        report()
    elif args.live:
        live(args.since)
    else:
        mine()


if __name__ == "__main__":
    main()
