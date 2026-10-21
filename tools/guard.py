#!/usr/bin/env python3
"""Keep private writing material out of this public repository.

The skill in this repository is distilled from private sources: an unpublished
manuscript and its review history, emails and drafts. This script blocks any
commit or push that would publish them. It runs as the git hooks in .githooks/
and on demand:

    python -X utf8 tools/guard.py --staged       check staged changes (pre-commit)
    python -X utf8 tools/guard.py --message F    check a commit message (commit-msg)
    python -X utf8 tools/guard.py --push         check commits being pushed (pre-push)
    python -X utf8 tools/guard.py --tree         check every tracked file
    python -X utf8 tools/guard.py --history      check every commit reachable from HEAD
    python -X utf8 tools/guard.py --build-index  index the private sources
    python -X utf8 tools/guard.py --selftest     check that every rule fires

A finding names the file, line and rule, never the private text that matched.
The private folder is named in tools/local.toml, which is not committed.
"""
import argparse
import glob
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "tools" / "local.toml"
ALLOW = REPO / "tools" / "allow.txt"

MAX_BYTES = 200_000
BLOCKED_DIRS = ("corpus/", "private/", "rounds/", "results/", "workspace/",
                "evals/private/", "evals/results/")
BLOCKED_EXT = {".jsonl", ".pdf", ".tex", ".bib", ".doc", ".docx", ".eml", ".mbox",
               ".msg", ".sqlite", ".db", ".pkl"}
INDEXED_EXT = {".tex", ".md", ".txt", ".json", ".jsonl"}
PROSE_EXT = {".md", ".txt", ""}  # "" covers commit messages
N = 7      # a shared run of this many words counts as a match
LONG = 10  # one shared run this long blocks on its own; shorter runs block in pairs

# Patterns are written so that their own source text does not match them.
PATTERNS = {
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "phone number": re.compile(r"\+\d{1,3}[ .-]?\(?\d{1,5}\)?(?:[ .-]?\d{2,5}){2,4}"
                               r"|\b07\d{3} ?\d{6}\b|\(\d{3}\) ?\d{3}-\d{4}"),
    "API key or secret": re.compile(r"sk-ant-[\w-]{10,}|ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}"
                                    r"|AIza[\w-]{30,}|xox[abprs]-[\w-]{10,}"
                                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "local path": re.compile(r"\b[A-Za-z]:[\\/](?:Users|GitHub)[\\/]|/[c]/Users/|(?<![\w.])/Users/\w"
                             r"|g[s]://", re.I),
    "manuscript id": re.compile(r"\b[A-Z]{2,5}-S-\d{2}-\d{4,6}\b"),
}
# The owner's address may appear in the plugin manifests, as in other public plugins.
EMAIL_OK_FILES = {".claude-plugin/marketplace.json", "plugins/write-like-myself/.claude-plugin/plugin.json"}

LATEX_COMMAND = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")


# ---------------------------------------------------------------- text

def words(text):
    """Lower-case word tokens, with LaTeX commands dropped but their arguments kept."""
    text = LATEX_COMMAND.sub(" ", text)
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def grams(tokens):
    """64-bit hashes of every run of N consecutive tokens."""
    for i in range(len(tokens) - N + 1):
        digest = hashlib.blake2b(" ".join(tokens[i:i + N]).encode(), digest_size=8).digest()
        yield int.from_bytes(digest, "big", signed=True)


def shared_runs(tokens, is_private):
    """Lengths, in words, of the maximal runs this text shares with private text."""
    runs, length = [], 0
    for h in grams(tokens):
        if is_private(h):
            length += 1
        elif length:
            runs.append(length + N - 1)
            length = 0
    if length:
        runs.append(length + N - 1)
    return runs


def copies_private(tokens, is_private):
    runs = shared_runs(tokens, is_private)
    return any(r >= LONG for r in runs) or len(runs) >= 2


def strings_in(value):
    """Every string inside a parsed JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings_in(v)


def read_text(path):
    """Text of a file for indexing; JSON and JSONL contribute their string values."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        out = []
        for line in raw.splitlines():
            try:
                out.extend(strings_in(json.loads(line)))
            except json.JSONDecodeError:
                out.append(line)
        return "\n".join(out)
    if suffix == ".json":
        try:
            return "\n".join(strings_in(json.loads(raw)))
        except json.JSONDecodeError:
            return raw
    return raw


# ---------------------------------------------------------------- config and index

def git(*args, repo=REPO, data=None):
    repo = Path(repo)
    result = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-c", "core.quotePath=false",
                             "-C", str(repo), *args], capture_output=True, input=data)
    if result.returncode != 0:
        sys.exit(f"guard: git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def private_dir():
    if not LOCAL.exists():
        return None
    return Path(tomllib.loads(LOCAL.read_text(encoding="utf-8"))["private_dir"]).expanduser()


def private_config(folder):
    return tomllib.loads((folder / "config.toml").read_text(encoding="utf-8"))


class Index:
    """Hashes of private text, minus hashes of text that may be published."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)

    def is_private(self, h):
        cur = self.db.execute("SELECT 1 FROM private WHERE h = ? AND h NOT IN (SELECT h FROM allowed)", (h,))
        return cur.fetchone() is not None

    def size(self):
        return self.db.execute("SELECT COUNT(*) FROM private").fetchone()[0]


def load_index(folder):
    """The private index, or None when this checkout has no private folder."""
    if folder is None:
        return None
    path = folder / "index.sqlite"
    if not path.exists():
        sys.exit("guard: the private index is missing, so nothing can be checked. "
                 "Run: python -X utf8 tools/guard.py --build-index")
    index = Index(path)
    if index.size() == 0:
        sys.exit("guard: the private index is empty. Run: python -X utf8 tools/guard.py --build-index")
    return index


def load_denylist(folder):
    if folder is None or not (folder / "denylist.txt").exists():
        return []
    entries = []
    for number, line in enumerate((folder / "denylist.txt").read_text(encoding="utf-8").splitlines(), 1):
        term = line.strip()
        if term and not term.startswith("#"):
            entries.append((number, term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)))
    return entries


def load_exceptions(folder):
    """Denylisted terms that the owner has allowed in one named file, kept in the private config
    because a public list of exceptions would itself publish the terms."""
    if folder is None:
        return {}
    exceptions = private_config(folder).get("guard", {}).get("denylist_exceptions", {})
    return {path: {term.lower() for term in terms} for path, terms in exceptions.items()}


def git_blobs(repo, paths):
    """Contents of every version of the text files under paths in a repository's history."""
    listing = git("rev-list", "--all", "--objects", "--", *paths, repo=repo).decode(errors="replace")
    shas = []
    for line in listing.splitlines():
        sha, _, name = line.partition(" ")
        if name and Path(name).suffix.lower() in INDEXED_EXT:
            shas.append(sha)
    shas = list(dict.fromkeys(shas))
    if not shas:
        return
    out = git("cat-file", "--batch", repo=repo, data=("\n".join(shas) + "\n").encode())
    pos = 0
    while pos < len(out):
        end = out.index(b"\n", pos)
        header = out[pos:end].split()
        pos = end + 1
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        blob = out[pos:pos + size]
        pos += size + 1
        if b"\0" not in blob:
            yield blob.decode("utf-8", errors="replace")


def files_under(pattern, base):
    """Indexable files matched by a glob, a file or a folder (relative patterns are under base)."""
    pattern = str(Path(pattern).expanduser())
    root = Path(pattern) if Path(pattern).is_absolute() else base / pattern
    if root.is_dir():
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in INDEXED_EXT)
    return sorted(Path(p) for p in glob.glob(str(root), recursive=True) if Path(p).is_file())


def build_index(folder):
    config = private_config(folder).get("index", {})
    db = sqlite3.connect(folder / "index.sqlite")
    db.executescript("""
        DROP TABLE IF EXISTS private; DROP TABLE IF EXISTS allowed; DROP TABLE IF EXISTS meta;
        CREATE TABLE private (h INTEGER PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE allowed (h INTEGER PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
    """)
    counts = {"git versions": 0, "files": 0, "allowed files": 0}

    def add(table, text):
        db.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?)", ((h,) for h in grams(words(text))))

    for source in config.get("git", []):
        for text in git_blobs(source["repo"], source.get("paths", ["."])):
            add("private", text)
            counts["git versions"] += 1
    for pattern in config.get("files", []) + config.get("private_dirs", []):
        for path in files_under(pattern, folder):
            add("private", read_text(path))
            counts["files"] += 1
    allowed = [p for pattern in config.get("allow_dirs", []) for p in files_under(pattern, folder)]
    allowed += [ALLOW] if ALLOW.exists() else []
    for path in allowed:
        add("allowed", read_text(path))
        counts["allowed files"] += 1
    total = db.execute("SELECT COUNT(*) FROM private").fetchone()[0]
    db.executemany("INSERT INTO meta VALUES (?, ?)",
                   [("built", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    ("counts", json.dumps(counts)), ("private hashes", str(total))])
    db.commit()
    print(f"guard: indexed {counts['git versions']} file versions from git and {counts['files']} private files "
          f"({total:,} distinct {N}-word runs); {counts['allowed files']} files may be quoted.")


# ---------------------------------------------------------------- checks

class Checker:
    def __init__(self, index, denylist, owner_email, exceptions=None):
        self.index, self.denylist, self.owner_email = index, denylist, owner_email
        self.exceptions = exceptions or {}
        self.findings = []

    def exempt(self, name, term):
        # name may carry a commit label before the path, as in "abc12345 CLAUDE.md"
        return any(name == path or name.endswith(" " + path) for path, terms in self.exceptions.items()
                   if term.lower() in terms)

    def flag(self, where, rule):
        self.findings.append(f"{where}  {rule}")

    def path(self, name, size, where=None):
        where = where or name
        if name.startswith(BLOCKED_DIRS):
            self.flag(where, "file in a private folder")
        if Path(name).suffix.lower() in BLOCKED_EXT:
            self.flag(where, f"{Path(name).suffix} files stay private")
        if size is not None and size > MAX_BYTES:
            self.flag(where, f"file larger than {MAX_BYTES // 1000} KB")

    def text(self, name, first_line, lines):
        """Check a block of consecutive lines that starts at first_line."""
        for offset, line in enumerate(lines):
            where = f"{name}:{first_line + offset}"
            for rule, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    if rule == "email address" and ("noreply" in match.group().lower() or (
                            name in EMAIL_OK_FILES and match.group().lower() == (self.owner_email or "").lower())):
                        continue  # public no-reply addresses, such as commit trailers, and the owner in manifests
                    self.flag(where, rule)
                    break
            for number, term, pattern in self.denylist:
                if pattern.search(line) and not self.exempt(name, term):
                    self.flag(where, f"denylisted term (entry {number} of the private denylist)")
        if self.index is not None:
            # Prose is checked a paragraph at a time, since a sentence may wrap across lines; data and
            # code a line at a time, so that chance matches across a whole file do not add up.
            units = paragraphs(first_line, lines) if Path(name).suffix.lower() in PROSE_EXT \
                else ((first_line + i, line) for i, line in enumerate(lines))
            for start, unit in units:
                if copies_private(words(unit), self.index.is_private):
                    self.flag(f"{name}:{start}", "copies private text")


def paragraphs(first_line, lines):
    """(first line number, text) for each run of non-blank lines."""
    start, buf = None, []
    for offset, line in enumerate(lines + [""]):
        if line.strip():
            if start is None:
                start = first_line + offset
            buf.append(line)
        elif buf:
            yield start, "\n".join(buf)
            start, buf = None, []


def added_blocks(diff):
    """(path, first line, lines) for each run of added lines in a unified diff with no context."""
    path, line, start, buf = None, 0, 0, []
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            if buf:
                yield path, start, buf
                buf = []
            path = raw[6:] if raw.startswith("+++ b/") else None
        elif raw.startswith("@@"):
            if buf:
                yield path, start, buf
                buf = []
            line = int(re.search(r"\+(\d+)", raw).group(1))
        elif raw.startswith("+") and path is not None:
            if not buf:
                start = line
            buf.append(raw[1:])
            line += 1
        elif buf:
            yield path, start, buf
            buf = []
    if buf:
        yield path, start, buf


def check_changes(checker, names, diff, size_of, label=""):
    for name in filter(None, names):
        checker.path(name, size_of(name), where=f"{label}{name}")
    for name, first, lines in added_blocks(diff.decode("utf-8", errors="replace")):
        if name and Path(name).suffix.lower() not in BLOCKED_EXT:
            checker.text(f"{label}{name}", first, lines)


def check_staged(checker):
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").decode(errors="replace").split("\n")
    diff = git("diff", "--cached", "-U0", "--no-color", "--diff-filter=ACMR")
    check_changes(checker, names, diff, lambda name: int(git("cat-file", "-s", f":{name}").decode().strip()))


def check_commits(checker, commits):
    for commit in commits:
        short = commit[:8]
        names = git("show", commit, "--name-only", "--format=", "--diff-filter=ACMR").decode(errors="replace")
        diff = git("show", commit, "-U0", "--no-color", "--format=", "--diff-filter=ACMR")
        check_changes(checker, names.split("\n"), diff,
                      lambda name: int(git("cat-file", "-s", f"{commit}:{name}").decode().strip()),
                      label=f"{short} ")
        message = git("log", "-1", "--format=%B", commit).decode("utf-8", errors="replace")
        checker.text(f"{short} (commit message)", 1, message.splitlines())


def pushed_commits(stdin):
    zero = "0" * 40
    commits = []
    for line in stdin.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == zero:
            continue
        local_sha, remote_sha = parts[1], parts[3]
        span = [local_sha, "--not", "--remotes"] if remote_sha == zero else [f"{remote_sha}..{local_sha}"]
        commits += git("rev-list", *span).decode().split()
    return list(dict.fromkeys(commits))


def report(checker, what):
    if checker.findings:
        print(f"guard: blocked {what}. Each line names a place and the rule it broke:", file=sys.stderr)
        for finding in dict.fromkeys(checker.findings):
            print(f"  {finding}", file=sys.stderr)
        print("Remove or rewrite the text, or, for a quote that may be published, add it to tools/allow.txt "
              "and rebuild the index.", file=sys.stderr)
        return 1
    print(f"guard: {what} clean.")
    return 0


# ---------------------------------------------------------------- self-test

def selftest():
    failures = []

    def expect(label, got, want):
        if got != want:
            failures.append(f"{label}: expected {want}, got {got}")

    checker = Checker(None, [], "owner" + "@" + "example.org")
    samples = {
        "email address": "write to someone" + "@" + "example.org today",
        "phone number": "call " + "+44 " + "20 7946 0958",
        "API key or secret": "key " + "sk-" + "ant-" + "a" * 20,
        "local path": "see " + "C:" + "\\" + "Users" + "\\" + "someone",
        "manuscript id": "ref " + "ABC" + "-S-" + "26" + "-" + "01234",
    }
    for rule, sample in samples.items():
        checker.findings = []
        checker.text("sample.md", 1, [sample])
        expect(rule, any(rule in f for f in checker.findings), True)
    checker.findings = []
    checker.text("sample.md", 1, ["Readers take the opening of a paragraph as its point, so say it first."])
    expect("clean prose", checker.findings, [])
    checker.findings = []
    checker.text(".claude-plugin/marketplace.json", 1, ['"email": "owner' + "@" + 'example.org"'])
    expect("owner email in manifest", checker.findings, [])

    for name, blocked in [("corpus/records.md", True), ("notes/a.jsonl", True), ("evals/private/t.json", True),
                          ("draft.tex", True), ("skills/write-like-myself/SKILL.md", False)]:
        checker.findings = []
        checker.path(name, 10)
        expect(f"path {name}", bool(checker.findings), blocked)
    checker.findings = []
    checker.path("big.md", MAX_BYTES + 1)
    expect("size", bool(checker.findings), True)

    canary = ("the river carried fine sediment past the old mill and settled it in a broad shallow bay "
              "below the bridge where herons fed at dawn")
    private = set(grams(words(canary)))
    is_private = private.__contains__
    copied = "As noted, the river carried fine sediment past the old mill and settled it somewhere."
    two_runs = "the river carried fine sediment past the old barn, and later a broad shallow bay below the bridge."
    one_short = "Then the river carried fine sediment past the town, as rivers do."
    reworded = "Fine silt moved downstream beyond the mill and came to rest in a wide inlet under the bridge."
    expect("ten-word copy", copies_private(words(copied), is_private), True)
    expect("two seven-word runs", copies_private(words(two_runs), is_private), True)
    expect("one short run", copies_private(words(one_short), is_private), False)
    expect("reworded", copies_private(words(reworded), is_private), False)

    deny = [(1, "Jane Doe", re.compile(r"(?<!\w)" + re.escape("Jane Doe") + r"(?!\w)", re.I))]
    checker = Checker(None, deny, None, {"NOTES.md": {"jane doe"}})
    checker.text("x.md", 1, ["Thanks to jane doe for comments."])
    expect("denylist", bool(checker.findings), True)
    checker.findings = []
    checker.text("abc12345 NOTES.md", 1, ["Thanks to jane doe for comments."])
    expect("denylist exception in its own file", checker.findings, [])
    checker.findings = []
    checker.text("commit message", 1, ["Co-Authored-By: A Bot <" + "noreply" + "@" + "example.com>"])
    expect("no-reply address", checker.findings, [])

    blocks = list(added_blocks("+++ b/a.md\n@@ -0,0 +3,2 @@\n+first\n+second\n"))
    expect("diff parsing", blocks, [("a.md", 3, ["first", "second"])])

    folder = private_dir()
    if folder is None:
        print("guard: no tools/local.toml, so this checkout runs the path and pattern rules only.")
    elif (folder / "index.sqlite").exists():
        print(f"guard: private index has {Index(folder / 'index.sqlite').size():,} runs.")
    else:
        failures.append("private folder is configured but has no index (run --build-index)")

    for failure in failures:
        print(f"guard selftest FAILED  {failure}", file=sys.stderr)
    if not failures:
        print("guard: selftest passed.")
    return 1 if failures else 0


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--message", metavar="FILE")
    mode.add_argument("--push", action="store_true")
    mode.add_argument("--tree", action="store_true")
    mode.add_argument("--history", action="store_true")
    mode.add_argument("--build-index", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    args, _ = parser.parse_known_args()

    if args.selftest:
        return selftest()
    folder = private_dir()
    if args.build_index:
        if folder is None:
            sys.exit("guard: tools/local.toml names no private folder.")
        build_index(folder)
        return 0

    owner = git("config", "user.email").decode().strip() or None
    checker = Checker(load_index(folder), load_denylist(folder), owner, load_exceptions(folder))
    if args.staged:
        check_staged(checker)
        return report(checker, "staged changes")
    if args.message:
        message = Path(args.message).read_text(encoding="utf-8", errors="replace")
        checker.text("commit message", 1, message.splitlines())
        return report(checker, "commit message")
    if args.push:
        commits = pushed_commits(sys.stdin.read())
        check_commits(checker, commits)
        return report(checker, f"push of {len(commits)} commits")
    if args.history:
        has_head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q", "HEAD"],
                                  capture_output=True).returncode == 0
        commits = git("rev-list", "HEAD").decode().split() if has_head else []
        check_commits(checker, commits)
        return report(checker, f"history of {len(commits)} commits")
    if args.tree:
        for name in filter(None, git("ls-files").decode("utf-8", errors="replace").split("\n")):
            blob = git("show", f":{name}")
            checker.path(name, len(blob))
            if b"\0" not in blob and Path(name).suffix.lower() not in BLOCKED_EXT:
                checker.text(name, 1, blob.decode("utf-8", errors="replace").splitlines())
        return report(checker, "tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
