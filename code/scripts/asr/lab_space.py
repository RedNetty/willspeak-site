#!/usr/bin/env python3
"""Sort a lab computer's run output."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

APP = Path(__file__).resolve().parents[2]
GB = 1024**3
MB = 1024**2

RESULT_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".tsv", ".log", ".md"}
RESULT_REPORT_DIRS = re.compile(
    r"^(finetune-torgo|lab-\d\d-.+|lab-big-.+|sentence-only|phrasebook-matching|f\d-.+|t\d-.+|torgo-.+|prototype-.+|setup-.+"
    r"|typical-speech-check|check-.+)$")
LOG_GLOBS = ("lab-*", "big-*", "train_parakeet-*.log")
SKIP_PARTS = {"ckpt", "__pycache__", "onnx-int8"}
MAX_RESULT_FILE = 50 * MB
BUNDLE_ROOTS = ("reports", "logs", "models")

PROTECTED = re.compile(
    r"^(parakeet-ft-M02e-A3|parakeet-ft-f2-M0[124]-bottom8-5epochs|parakeet-ft-f1-loso-M0[124]-bottom8"
    r"|parakeet-ft-prototype-.+)$")
GIT_SUFFIX_NEVER = {".nemo", ".ckpt", ".pt", ".onnx", ".wav", ".flac", ".mp3", ".db", ".sqlite", ".zip", ".tgz",
                    ".tar", ".gz", ".safetensors", ".bin", ".npz"}
GIT_CODE_SUFFIX = {".py", ".sh", ".bat", ".md", ".toml", ".txt", ".html", ".js", ".css", ".yml", ".yaml", ".cfg"}
GIT_MAX_BYTES = 1 * MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(MB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


@dataclass
class Model:
    path: Path
    rel: str
    size: int
    ckpt_size: int
    final: Path | None
    final_ok: bool
    status: str | None
    scored: bool
    role: str
    note: str

    @property
    def clean_candidate(self) -> bool:
        return self.ckpt_size > 0 and self.final_ok and self.status in ("complete", None)


def nemo_intact(path: Path) -> bool:
    try:
        size = path.stat().st_size
        with tarfile.open(path, "r:") as tar:
            members = {m.name.lstrip("./"): m for m in tar.getmembers()}
        weights = members.get("model_weights.ckpt")
        return weights is not None and weights.offset_data + weights.size <= size
    except (tarfile.TarError, OSError):
        return False


def final_file(model_dir: Path) -> tuple[Path | None, bool]:
    for name in ("parakeet-ft.nemo", "adapter.pt"):
        candidate = model_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, nemo_intact(candidate) if name.endswith(".nemo") else True
    return None, False


def scored_names() -> set[str]:
    found: set[str] = set()
    base = APP / "reports" / "finetune-torgo"
    for folder, prefix in ((base, ""), (base / "lab-baselines", "lab-baselines/")):
        if folder.is_dir():
            for entry in folder.iterdir():
                if entry.is_dir() and (entry / "results.json").is_file():
                    found.add(prefix + entry.name)
    return found


def scored_folders(name: str, scored: set[str]) -> list[str]:
    own = re.compile(re.escape(name) + r"-(test|dev)(-[a-z0-9]+)?")
    return sorted(s for s in scored if s in (name, f"controls-{name}") or own.fullmatch(s))


def is_scored(name: str, scored: set[str]) -> bool:
    return bool(scored_folders(name, scored))


def scan_models() -> list[Model]:
    lab15_done = (APP / "reports" / "lab-15-int8-screen" / "screen.json").is_file()
    scored = scored_names()
    models: list[Model] = []
    for root in (APP / "models", APP / "models" / "lab-baselines"):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name == "lab-baselines":
                continue
            rel = d.relative_to(APP / "models").as_posix()
            status = None
            try:
                status = json.loads((d / "run.json").read_text()).get("status")
            except (OSError, ValueError, AttributeError):
                pass
            final, ok = final_file(d)
            ckpt = d / "ckpt"
            name = d.name.removeprefix("parakeet-ft-") if root.name == "models" else rel
            is_scored_ = is_scored(name, scored)
            if rel == "personal":
                role, note = "PROTECTED", "personal device models (scripts/asr/personal.py); private, never bundled"
            elif rel.startswith("lab-baselines/") or PROTECTED.match(d.name):
                role, note = "PROTECTED", "input to later lab stages or a deployment candidate"
            elif final is None and status is None and not d.name.startswith("parakeet-ft-"):
                role, note = "OTHER", "not a training run (app model, export or smoke test)"
            elif status not in (None, "complete") or final is None or not ok:
                role, note = "INCOMPLETE", "no intact final model; look at its log before touching it"
            elif d.name.startswith("parakeet-ft-lab-") and not lab15_done:
                role, note = "HOLD", "stage 15 (int8 screen) has not run and scans these"
            elif not is_scored_:
                role, note = "UNSCORED", "no scored decode yet"
            else:
                role, note = "FINISHED", "scored; weights only needed to re-decode or export"
            models.append(Model(d, rel, tree_size(d), tree_size(ckpt) if ckpt.is_dir() else 0, final, ok, status,
                                is_scored_, role, note))
    return models


def result_files() -> tuple[list[str], list[tuple[str, str]]]:
    keep: list[str] = []
    skipped: list[tuple[str, str]] = []

    def consider(path: Path) -> None:
        rel = path.relative_to(APP).as_posix()
        if path.suffix.lower() not in RESULT_SUFFIXES:
            return
        if path.name == "notes.md":
            return
        if SKIP_PARTS & set(path.relative_to(APP).parts):
            return
        if path.stat().st_size > MAX_RESULT_FILE:
            skipped.append((rel, f"larger than {human(MAX_RESULT_FILE)}"))
            return
        keep.append(rel)

    reports = APP / "reports"
    if reports.is_dir():
        for top in sorted(reports.iterdir()):
            if top.is_dir() and RESULT_REPORT_DIRS.match(top.name):
                for p in sorted(top.rglob("*")):
                    if p.is_file() and not p.is_symlink():
                        consider(p)
    logs = APP / "logs"
    if logs.is_dir():
        for pattern in LOG_GLOBS:
            for p in sorted(logs.glob(pattern)):
                if p.is_file():
                    consider(p)
    models = APP / "models"
    if models.is_dir():
        for p in sorted(models.rglob("run.json")):
            if p.relative_to(models).parts[0] != "personal":
                consider(p)
    return sorted(set(keep)), skipped


def safe_member(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if "\\" in name or p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] not in BUNDLE_ROOTS:
        raise ValueError(f"unsafe path in bundle: {name!r}")
    return p


def inspect_bundle(bundle: Path) -> dict:
    with tarfile.open(bundle, "r:gz") as tar:
        manifest_file = tar.extractfile("MANIFEST.json")
        if manifest_file is None:
            raise ValueError("MANIFEST.json is missing")
        manifest = json.load(manifest_file)
        expected = {item["path"]: item for item in manifest["files"]}
        seen = set()
        for member in tar:
            if member.name == "MANIFEST.json":
                continue
            safe_member(member.name)
            if not member.isreg() or member.name not in expected:
                raise ValueError(f"not in the manifest or not a plain file: {member.name}")
            data = tar.extractfile(member)
            digest = hashlib.sha256()
            for chunk in iter(lambda: data.read(MB), b""):
                digest.update(chunk)
            item = expected[member.name]
            if member.size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"damaged: {member.name}")
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f"missing from the bundle: {sorted(set(expected) - seen)[:5]}")
    return manifest


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(APP), "--no-optional-locks", "log", "-1", "--format=%h %ad",
                               "--date=format:%d %b %H:%M"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_pack(args) -> int:
    files, skipped = result_files()
    if not files:
        print("No result files found under reports/, logs/ or models/*/run.json.")
        return 1
    stamp = args.stamp or datetime.now().strftime("%m%d-%H%M")
    out = Path(args.out) if args.out else APP / "transfers" / f"lab-results-{stamp}.tgz"
    if out.exists():
        print(f"{out} already exists; pass --stamp or --out for another name.")
        return 1
    total = sum((APP / f).stat().st_size for f in files)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < total + 100 * MB:
        print(f"Only {human(free)} free where the bundle goes; it needs about {human(total)}. "
              "Run `clean` first or use --out.")
        return 1
    items = [{"path": f, "size": (APP / f).stat().st_size, "sha256": sha256_of(APP / f)} for f in files]
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": socket.gethostname(),
                "code": git_head(), "files": items}
    part = out.with_name(out.name + ".part")
    with tarfile.open(part, "w:gz") as tar:
        raw = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(raw)
        info.mtime = int(datetime.now().timestamp())
        tar.addfile(info, io.BytesIO(raw))
        for f in files:
            tar.add(APP / f, arcname=f, recursive=False)
    inspect_bundle(part)
    part.replace(out)
    digest = sha256_of(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"Bundle: {out}")
    print(f"{len(files)} files, {human(total)} before compression, {human(out.stat().st_size)} on disk; "
          "read back and verified.")
    print(f"sha256 {digest}  (also in {out.name}.sha256)")
    for rel, why in skipped:
        print(f"  skipped {rel}: {why}")
    print("\nNo private recordings: only decodes, score tables, logs and run.json files are inside.")
    print(f"Move it to the other computer (USB, scp, school drive), then there:\n"
          f"  python3 scripts/asr/lab_space.py unpack {shlex.quote(out.name)}")
    return 0


def cmd_unpack(args) -> int:
    bundle = Path(args.bundle)
    try:
        manifest = inspect_bundle(bundle)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"Bundle check failed, nothing extracted: {exc}")
        return 1
    dest = Path(args.into) if args.into else APP
    new, same, conflicts = [], 0, []
    for item in manifest["files"]:
        target = dest / safe_member(item["path"])
        if not target.exists():
            new.append(item)
        elif sha256_of(target) == item["sha256"]:
            same += 1
        else:
            conflicts.append(item["path"])
    if conflicts:
        print("These files already exist here with different contents, so nothing was extracted:")
        for c in conflicts[:20]:
            print("  ", c)
        return 1
    need = sum(i["size"] for i in new)
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest).free < need + 50 * MB:
        print(f"Not enough free space for {human(need)}.")
        return 1
    with tarfile.open(bundle, "r:gz") as tar:
        for item in new:
            target = dest / safe_member(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with tar.extractfile(item["path"]) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_of(tmp) != item["sha256"]:
                tmp.unlink()
                print(f"Damaged on extraction: {item['path']}")
                return 1
            os.replace(tmp, target)
    print(f"Bundle from {manifest.get('host')} (code {manifest.get('code')}, {manifest.get('created')}).")
    print(f"Extracted {len(new)} files into {dest}; {same} were already here and identical.")
    return 0


def git_root() -> Path:
    try:
        out = subprocess.run(["git", "-C", str(APP), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             check=True).stdout.strip()
        return Path(out)
    except (OSError, subprocess.CalledProcessError):
        return APP.parent


def git_changes(root: Path) -> list[tuple[str, str]]:
    out = subprocess.run(["git", "-C", str(root), "--no-optional-locks", "status", "--porcelain=v1", "-z", "-uall"],
                         capture_output=True, text=True, check=True).stdout
    parts = out.split("\0")
    changes, i = [], 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if code[0] in "RC":
            i += 1
        changes.append((code, path))
    return changes


def classify_git(root: Path, code: str, path: str) -> tuple[str, str]:
    p = PurePosixPath(path)
    full = root / path
    if "D" in code:
        return "REVIEW", "tracked file is missing here"
    size = full.stat().st_size if full.is_file() else 0
    in_app = p.parts[:1] == ("app",)
    rest = p.parts[1:] if in_app else p.parts
    if rest[:1] in (("data",), ("models",)) and p.name != ".gitkeep":
        return "NEVER", "private data or model weights"
    if p.suffix.lower() in GIT_SUFFIX_NEVER and rest[:3] != ("src", "willspeak", "models"):
        return "NEVER", "binary, weights or archive"
    if size > 5 * MB:
        return "NEVER", f"{human(size)}"
    if size > GIT_MAX_BYTES:
        return "REVIEW", f"{human(size)}"
    if p.name == "notes.md" or p.name == "Report Context.md":
        return "COMMIT", "experiment notes"
    if p.suffix.lower() in GIT_CODE_SUFFIX:
        return "COMMIT", "code or documentation"
    return "REVIEW", "not a type the project normally tracks"


def cmd_git(args) -> int:
    root = git_root()
    try:
        changes = git_changes(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"git status failed: {exc}")
        return 1
    return print_git(root, changes, stage=args.stage)


def print_git(root: Path, changes: list[tuple[str, str]], stage: bool = False) -> int:
    print("== Git: what to commit ==")
    if not changes:
        print("Nothing shows as changed. Results, models and logs are ignored by .gitignore, so a clean tree is\n"
              "normal after a run that only produced output. To get results off this computer through Git run\n"
              "`git-results --commit` (side branch), or `pack` for a file.")
        return 0
    groups: dict[str, list[tuple[str, str, str]]] = {"COMMIT": [], "REVIEW": [], "NEVER": []}
    for code, path in changes:
        kind, why = classify_git(root, code, path)
        groups[kind].append((code, path, why))
    for kind, title in (("COMMIT", "Safe to commit"), ("REVIEW", "Look at these first"),
                        ("NEVER", "Keep out of Git (the ignore rules have a gap if these show up)")):
        if groups[kind]:
            print(f"\n{title}:")
            for code, path, why in groups[kind]:
                print(f"  {code} {path}   ({why})")
    paths = [p for _, p, _ in groups["COMMIT"]]
    if paths:
        add = "git add -- " + " ".join(shlex.quote(p) for p in paths)
        if stage:
            subprocess.run(["git", "-C", str(root), "add", "--", *paths], check=True)
            print(f"\nStaged {len(paths)} file(s) with `git add` (nothing else was touched).")
        else:
            print(f"\nStage them with this, or rerun with --stage:\n  cd {shlex.quote(str(root))}\n  {add}")
        print("Then you commit and push, so the commit is yours:\n"
              "  git commit -m \"<what you ran and found>\"\n  git push")
    return 0


GIT_RESULT_FILE_CAP = 25 * MB
GIT_RESULT_TOTAL_CAP = 80 * MB


def git_out(root: Path, *args: str, env: dict | None = None, text: bool = True) -> str | bytes:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=text, check=True, env=env)
    return done.stdout.strip() if text else done.stdout


def repo_prefix(root: Path) -> str:
    rel = APP.resolve().relative_to(root.resolve()).as_posix()
    return "" if rel == "." else rel + "/"


def known_result_blobs(root: Path, prefix: str, newest: int = 8) -> tuple[str | None, dict[str, set[str]]]:
    refs = git_out(root, "for-each-ref", "--sort=-refname", "--sort=-committerdate", f"--count={newest}", "--format=%(refname)",
                   "refs/heads/lab-results-*", "refs/remotes/*/lab-results-*").split()
    known: dict[str, set[str]] = {}
    for ref in refs:
        raw = git_out(root, "ls-tree", "-r", "-z", ref, "--", prefix or ".", text=False).decode("utf-8", "replace")
        for entry in filter(None, raw.split("\0")):
            meta, _, path = entry.partition("\t")
            known.setdefault(path, set()).add(meta.split()[2])
    return (refs[0] if refs else None), known


def blob_ids(root: Path, paths: list[Path]) -> list[str]:
    if not paths:
        return []
    done = subprocess.run(["git", "-C", str(root), "hash-object", "--stdin-paths"], input="\n".join(map(str, paths)) + "\n",
                          capture_output=True, text=True, check=True)
    return done.stdout.split()


def split_parts(files: list[str], cap: int) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    used = 0
    for f in files:
        size = (APP / f).stat().st_size
        if parts[-1] and used + size > cap:
            parts.append([])
            used = 0
        parts[-1].append(f)
        used += size
    return parts


def cmd_git_results(args) -> int:
    root = git_root()
    prefix = repo_prefix(root)
    files, skipped = result_files()
    keep = []
    for f in files:
        if (APP / f).stat().st_size > GIT_RESULT_FILE_CAP:
            skipped.append((f, f"larger than {human(GIT_RESULT_FILE_CAP)} for Git"))
        else:
            keep.append(f)
    if not keep:
        print("No result files found under reports/, logs/ or models/*/run.json.")
        return 1
    stamp = args.stamp or datetime.now().strftime("%m%d-%H%M")
    branch = args.branch or f"lab-results-{stamp}"
    if args.commit:
        for existing in (branch, *git_out(root, "for-each-ref", "--format=%(refname:short)", f"refs/heads/{branch}-p*").split()):
            try:
                git_out(root, "rev-parse", "--verify", "-q", f"refs/heads/{existing}")
            except subprocess.CalledProcessError:
                continue
            print(f"Branch {existing} already exists; pass --stamp or --branch for another name.")
            return 1
    base_ref, known = known_result_blobs(root, prefix)
    ids = blob_ids(root, [APP / f for f in keep])
    fresh = [f for f, blob in zip(keep, ids) if blob not in known.get(prefix + f, ())]
    if base_ref and not fresh:
        print(f"{len(keep)} result files, none new since {base_ref}; no branch built.")
        return 0
    parts = split_parts(fresh, GIT_RESULT_TOTAL_CAP)
    sizes = [sum((APP / f).stat().st_size for f in part) for part in parts]
    names = [branch] if len(parts) == 1 else [f"{branch}-p{i}of{len(parts)}" for i in range(1, len(parts) + 1)]
    since = f"since {base_ref}" if base_ref else "(no earlier results branch)"
    print(f"{len(keep)} result files, {len(fresh)} new {since}: {human(sum(sizes))} in {len(parts)} branch(es)")
    for name, part, size in zip(names, parts, sizes):
        print(f"  {name}: {len(part)} files, {human(size)}")
    for rel, why in skipped:
        print(f"  skipped {rel}: {why}")
    if not args.commit:
        print("\nDry run. Add --commit to build the branch. Nothing was changed.")
        return 0
    old = [f for f in keep if f not in set(fresh)]
    built: list[tuple[str, str]] = []
    upto: list[str] = []
    for name, part in zip(names, parts):
        upto += part
        tmp = Path(tempfile.mkdtemp(prefix="lab-results-index-"))
        env = {**os.environ, "GIT_INDEX_FILE": str(tmp / "index")}
        try:
            git_out(root, "read-tree", "HEAD", env=env)
            paths = [prefix + f for f in old + upto]
            for i in range(0, len(paths), 100):
                git_out(root, "add", "-f", "--", *paths[i:i + 100], env=env)
            tree = git_out(root, "write-tree", env=env)
            commit = git_out(root, "commit-tree", tree, "-p", "HEAD", "-m", f"Lab results {stamp}" + (f" ({name})" if len(names) > 1 else ""), env=env)
            git_out(root, "update-ref", f"refs/heads/{name}", commit)
            built.append((name, commit))
        except subprocess.CalledProcessError as exc:
            print(f"git failed: {exc.stderr.strip() if exc.stderr else exc}")
            return 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    listed = " ".join(name for name, _ in built)
    print(f"\nBuilt {listed} ({built[-1][1][:10]}) from your current commit plus the results. "
          "Your branch, index and files were not touched.\nPush "
          + ("all of them" if len(built) > 1 else "it") + ":\n"
          f"  git push -u origin {listed}\n"
          "The commit uses your own Git name and email. Then on the other computer:\n"
          f"  git fetch origin {listed}\n"
          + (f"  python3 scripts/asr/lab_space.py fetch-results origin/{built[-1][0]}\n"
             "(the last branch holds everything, because each one adds to the one before)" if len(built) > 1
             else f"  python3 scripts/asr/lab_space.py fetch-results origin/{built[0][0]}"))
    return 0


def cmd_fetch_results(args) -> int:
    root = git_root()
    prefix = repo_prefix(root)
    dest = Path(args.into) if args.into else APP
    try:
        listing = git_out(root, "ls-tree", "-r", "-z", "--name-only", args.ref, "--", prefix or ".")
    except subprocess.CalledProcessError as exc:
        print(f"Cannot read {args.ref}: {exc.stderr.strip() if exc.stderr else exc}\nDid you run `git fetch` first?")
        return 1
    wanted = []
    for name in filter(None, listing.split("\0")):
        rel = name[len(prefix):]
        parts = PurePosixPath(rel).parts
        suffix = PurePosixPath(rel).suffix.lower()
        if parts[:1] not in (("reports",), ("logs",), ("models",)) or rel.endswith("notes.md"):
            continue
        if parts[0] == "reports" and not RESULT_REPORT_DIRS.match(parts[1] if len(parts) > 2 else ""):
            continue
        if parts[0] == "models" and parts[-1] != "run.json":
            continue
        if suffix in RESULT_SUFFIXES:
            safe_member(rel)
            wanted.append((name, rel))
    if not wanted:
        print(f"No result files in {args.ref}.")
        return 1
    new, same, conflicts = [], 0, []
    for name, rel in wanted:
        blob = git_out(root, "show", f"{args.ref}:{name}", text=False)
        target = dest / rel
        if not target.exists():
            new.append((target, blob))
        elif hashlib.sha256(blob).hexdigest() == sha256_of(target):
            same += 1
        else:
            conflicts.append(rel)
    if conflicts:
        print("These files already exist here with different contents, so nothing was written:")
        for c in conflicts[:20]:
            print("  ", c)
        return 1
    for target, blob in new:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    print(f"Wrote {len(new)} files into {dest}; {same} were already here and identical. Nothing was staged.")
    return 0


def disk_line() -> str:
    u = shutil.disk_usage(APP)
    return f"{human(u.free)} free of {human(u.total)} ({100 * u.used / u.total:.0f}% used)"


def cache_report() -> None:
    home = Path.home()
    rows = [("~/.cache/huggingface", home / ".cache/huggingface", "model downloads incl. the stock .nemo"),
            ("~/.cache/torch", home / ".cache/torch", "NeMo/torch downloads"),
            ("~/.cache/willspeak-tmp", home / ".cache/willspeak-tmp", "NeMo staging; `clean` empties it when idle"),
            ("~/.cache/pip", home / ".cache/pip", "package cache; `clean --also-package-caches`"),
            ("~/.cache/uv", home / ".cache/uv", "package cache; `clean --also-package-caches`")]
    print("\n== Outside the repository (read-only here) ==")
    for label, path, note in rows:
        if path.exists():
            print(f"  {human(tree_size(path)):>9}  {label}   {note}")
    venvs = home / "venvs"
    if venvs.is_dir():
        for v in sorted(venvs.iterdir()):
            print(f"  {human(tree_size(v)):>9}  ~/venvs/{v.name}   environment; keep")


def print_models(models: list[Model]) -> None:
    print("\n== Models (models/) ==")
    if not models:
        print("  none found")
        return
    for m in models:
        extra = f", ckpt/ {human(m.ckpt_size)}" if m.ckpt_size else ""
        print(f"  {human(m.size):>9}  {m.role:<10} {m.rel}{extra}")
    free_ckpt = sum(m.ckpt_size for m in models if m.clean_candidate)
    finished = [m for m in models if m.role == "FINISHED"]
    print(f"\n  Removable now (ckpt/ of finished runs): {human(free_ckpt)}  -> `clean --apply`")
    print(f"  FINISHED models ({len(finished)}, {human(sum(m.size - m.ckpt_size for m in finished))}): copy with "
          "`archive-models --to DIR`; never deleted without a verified copy")
    for role, text in (("HOLD", "run stage 15 first: `python lab.py run cheap \"15\"`"),
                       ("INCOMPLETE", "kept; a cut-off .nemo from a full disk shows up here"),
                       ("UNSCORED", "kept; decode/score them before archiving")):
        n = [m.rel for m in models if m.role == role]
        if n:
            print(f"  {role}: {len(n)} kept - {text}")


def cmd_report(args) -> int:
    print(f"Lab output report for {APP}\nDisk where the repo lives: {disk_line()}\n")
    print("== Biggest folders ==")
    for sub in ("models", "reports", "logs", "data"):
        p = APP / sub
        if p.exists():
            print(f"  {human(tree_size(p)):>9}  {sub}/")
    models = scan_models()
    print_models(models)
    files, skipped = result_files()
    print(f"\n== Results to carry back ==\n  {len(files)} files, {human(sum((APP / f).stat().st_size for f in files))}"
          f" (Git ignores them) -> `pack`; {len(skipped)} skipped as too large")
    if not args.quick:
        cache_report()
    print()
    root = git_root()
    try:
        print_git(root, git_changes(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"== Git ==\n  git status failed: {exc}")
    print("\n== Suggested order ==\n"
          "  1. python3 scripts/asr/lab_space.py git --stage        then git commit / git push yourself\n"
          "  2. python3 scripts/asr/lab_space.py git-results --commit   then git push -u origin <that branch>\n"
          "     (or `pack` if you can move a file)\n"
          "  3. python3 scripts/asr/lab_space.py clean --apply       frees the ckpt/ folders\n"
          "  4. if still short: archive-models --to /path/on/big/disk --apply --remove-after-verify")
    return 0


def saved_result_files(root: Path, prefix: str) -> set[str]:
    refs = git_out(root, "for-each-ref", "--format=%(refname)", "refs/heads/lab-results-*").split()
    saved: set[str] = set()
    for ref in refs:
        for subtree in ("reports/", "models/"):
            listing = git_out(root, "ls-tree", "-r", "-z", "--name-only", ref, "--", prefix + subtree)
            saved.update(name[len(prefix):] for name in listing.split("\0") if name)
    return saved


def training_running(allow_parent_queue: bool = False) -> bool:
    pattern = r"(scripts/asr/(train|nemo_hyp|export|hf_hyp|evaluate)\.py|lab\.py (run|queue))"
    try:
        done = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        if done.returncode != 0:
            return False
        parent = str(os.getppid())
        for line in done.stdout.splitlines():
            pid, _, command = line.partition(" ")
            if allow_parent_queue and pid == parent and re.search(r"\blab\.py queue\b", command):
                continue
            return True
        return False
    except OSError:
        return True


def remove_saved_lab_models(apply: bool) -> int:
    root = git_root()
    try:
        saved = saved_result_files(root, repo_prefix(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  lab models kept: cannot read the lab-results branches ({exc})")
        return 0
    scored = scored_names()
    freed = 0
    for m in scan_models():
        if m.role != "FINISHED" or not m.path.name.startswith("parakeet-ft-lab-"):
            continue
        folders = scored_folders(m.path.name.removeprefix("parakeet-ft-"), scored)
        needed = [
            f"reports/finetune-torgo/{f}/{name}"
            for f in folders
            for name in ("results.json", "hypotheses.jsonl")
        ]
        missing = [n for n in needed if n not in saved]
        if missing:
            print(f"  kept       {m.rel}   (its results are not in a lab-results branch yet: {missing[0]})")
            continue
        size = m.size - m.ckpt_size
        print(f"  {human(size):>9}  {m.path.relative_to(APP)}   (decodes saved in a lab-results branch)")
        freed += size
        if apply:
            shutil.rmtree(m.path)
    return freed


def remove_saved_long_models(apply: bool) -> int:
    root = git_root()
    try:
        saved = saved_result_files(root, repo_prefix(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  long models kept: cannot read the lab-results branches ({exc})")
        return 0
    folders = {
        "adapter": "f2-long-adapter",
        "acoustic": "f2-acoustic-only",
        "replay": "f2-speaker-replay",
    }
    freed = 0
    for model_root in sorted((APP / "models").glob("parakeet-ft-long-*")):
        match = re.fullmatch(r"parakeet-ft-long-(adapter|acoustic|replay)-(.+)", model_root.name)
        if not match or not model_root.is_dir():
            continue
        study, stamp = match.groups()
        report = APP / "reports" / folders[study] / stamp
        required = {f"reports/{folders[study]}/{stamp}/{name}" for name in ("summary.json", "battery.json")}
        if not all((report / name).is_file() for name in ("summary.json", "battery.json")):
            print(f"  kept       {model_root.relative_to(APP)}   (study is not complete: summary.json and battery.json are required)")
            continue
        report_files = {
            (Path("reports") / folders[study] / stamp / path.relative_to(report)).as_posix()
            for path in report.rglob("*")
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in RESULT_SUFFIXES
            and path.name != "notes.md" and not (SKIP_PARTS & set(path.relative_to(report).parts))
        }
        model_records = {path.relative_to(APP).as_posix() for path in model_root.rglob("run.json")}
        missing = sorted((required | report_files | model_records) - saved)
        if missing:
            print(f"  kept       {model_root.relative_to(APP)}   (results not safely stored in a lab-results branch: {missing[0]})")
            continue
        size = tree_size(model_root)
        print(f"  {human(size):>9}  {model_root.relative_to(APP)}   (complete study; all reports saved in a lab-results branch)")
        freed += size
        if apply:
            shutil.rmtree(model_root)
    return freed


def cmd_clean(args) -> int:
    mode = "REMOVING" if args.apply else "DRY RUN (add --apply to remove)"
    print(f"{mode}\nDisk before: {disk_line()}\n")
    freed = 0
    for m in scan_models():
        if m.clean_candidate:
            ckpt = m.path / "ckpt"
            print(f"  {human(m.ckpt_size):>9}  {ckpt.relative_to(APP)}   (final {m.final.name} verified intact)")
            freed += m.ckpt_size
            if args.apply:
                shutil.rmtree(ckpt)
        elif m.ckpt_size:
            print(f"  kept       {m.rel}/ckpt   ({m.role}: {m.note})")
    if getattr(args, "lab_models", False):
        freed += remove_saved_lab_models(args.apply)
    if getattr(args, "long_models", False):
        if training_running(allow_parent_queue=True):
            print("  long models kept   (a training or decode process appears to be active)")
        else:
            freed += remove_saved_long_models(args.apply)
    targets = [Path.home() / ".cache/willspeak-tmp"]
    if args.also_package_caches:
        targets += [Path.home() / ".cache/pip", Path.home() / ".cache/uv"]
    busy = training_running()
    for t in targets:
        if not t.is_dir():
            continue
        if busy:
            print(f"  skipped    {t}   (a run or decode looks active, or pgrep is missing)")
            continue
        size = tree_size(t)
        print(f"  {human(size):>9}  {t}")
        freed += size
        if args.apply:
            for child in t.iterdir():
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    print(f"\n{'Freed' if args.apply else 'Would free'} about {human(freed)}.")
    if args.apply:
        print(f"Disk after: {disk_line()}")
    return 0


def copy_verified(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as s, open(tmp, "wb") as d:
        for chunk in iter(lambda: s.read(MB), b""):
            digest.update(chunk)
            d.write(chunk)
        d.flush()
        os.fsync(d.fileno())
    if sha256_of(tmp) != digest.hexdigest():
        tmp.unlink()
        raise OSError(f"copy of {src} did not verify")
    os.replace(tmp, dst)


def cmd_archive(args) -> int:
    dest = Path(args.to).expanduser()
    finished = [m for m in scan_models() if m.role == "FINISHED"]
    if not finished:
        print("No FINISHED models (complete, intact, scored, not protected, not on hold). Nothing to archive.")
        return 0
    need = sum(m.size - m.ckpt_size for m in finished)
    print(f"{'COPYING' if args.apply else 'DRY RUN (add --apply)'}: {len(finished)} models, {human(need)}, to {dest}")
    if args.apply:
        dest.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dest).free < need + GB:
            print(f"Only {human(shutil.disk_usage(dest).free)} free at {dest}.")
            return 1
    for m in finished:
        print(f"  {human(m.size - m.ckpt_size):>9}  {m.rel}")
        if not args.apply:
            continue
        try:
            for f in sorted(m.path.rglob("*")):
                if f.is_file() and "ckpt" not in f.relative_to(m.path).parts:
                    target = dest / "models" / m.rel / f.relative_to(m.path)
                    if target.exists() and sha256_of(target) == sha256_of(f):
                        continue
                    if target.exists():
                        raise OSError(f"{target} exists with different contents")
                    copy_verified(f, target)
        except OSError as exc:
            print(f"    FAILED, original kept: {exc}")
            continue
        if args.remove_after_verify:
            shutil.rmtree(m.path)
            print("    copy verified, original removed")
        else:
            print("    copy verified, original kept")
    if args.apply:
        back = shlex.quote(str(dest / "models"))
        print(f"\nTo bring one back: cp -r {back}/<name> {shlex.quote(str(APP / 'models'))}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report", help="read-only overview")
    p.add_argument("--quick", action="store_true", help="skip the sizes of caches and environments outside the repo")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("git", help="what to commit")
    p.add_argument("--stage", action="store_true", help="run git add on the safe files (never commits or pushes)")
    p.set_defaults(func=cmd_git)
    p = sub.add_parser("git-results", help="results onto a side branch, built without touching your branch")
    p.add_argument("--commit", action="store_true", help="build the branch (without it: a dry run)")
    p.add_argument("--stamp")
    p.add_argument("--branch")
    p.set_defaults(func=cmd_git_results)
    p = sub.add_parser("fetch-results", help="take results from a git-results branch (after git fetch)")
    p.add_argument("ref", help="for example origin/lab-results-1007-1630")
    p.add_argument("--into")
    p.set_defaults(func=cmd_fetch_results)
    p = sub.add_parser("pack", help="results bundle for the other computer")
    p.add_argument("--stamp")
    p.add_argument("--out")
    p.set_defaults(func=cmd_pack)
    p = sub.add_parser("unpack", help="verify and extract a results bundle")
    p.add_argument("bundle")
    p.add_argument("--into")
    p.set_defaults(func=cmd_unpack)
    p = sub.add_parser("clean", help="remove ckpt/ folders of finished runs and stale temp files")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--also-package-caches", action="store_true", help="also empty ~/.cache/pip and ~/.cache/uv")
    p.add_argument("--lab-models", action="store_true",
                   help="also remove finished lab-test models whose decodes are in a lab-results-* branch")
    p.add_argument("--long-models", action="store_true",
                   help="also remove complete long-study model folders, only when all reports are saved in a lab-results-* branch")
    p.set_defaults(func=cmd_clean)
    p = sub.add_parser("archive-models", help="copy finished models elsewhere, verified by sha256")
    p.add_argument("--to", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--remove-after-verify", action="store_true")
    p.set_defaults(func=cmd_archive)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
