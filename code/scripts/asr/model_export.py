#!/usr/bin/env python3
"""Zip finished models, put them on Google Drive (or any folder), check the copy, and only then free the disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab_space  # noqa: E402

APP = lab_space.APP
MB = 1024 ** 2
GB = 1024 ** 3
BIG_NAMES = ("parakeet-ft.nemo", "adapter.pt")
STORED = (".nemo", ".pt", ".onnx", ".ckpt", ".weights", ".npz", ".zip")


@dataclass
class Destination:
    kind: str
    remote: str
    rclone: str = "rclone"

    @classmethod
    def parse(cls, text: str, rclone: str = "rclone") -> "Destination":
        if text.startswith("rclone:"):
            return cls("rclone", text[len("rclone:"):].rstrip("/"), rclone)
        return cls("folder", str(Path(text).expanduser()), rclone)

    def _join(self, name: str) -> str:
        return f"{self.remote}/{name}" if self.kind == "rclone" else str(Path(self.remote) / name)

    def rc(self, *args: str, stdin=None, timeout: int | None = 600) -> subprocess.CompletedProcess:
        return subprocess.run([self.rclone, *args], stdin=stdin, capture_output=True, text=True, timeout=timeout, check=False)

    def ready(self) -> str | None:
        if self.kind == "rclone":
            if shutil.which(self.rclone) is None and not Path(self.rclone).is_file():
                return ("rclone is not installed. sudo apt install rclone, then rclone config (see --help for the steps). "
                        "Or use --to with a folder that Google Drive for desktop syncs.")
            remote = self.remote.split(":", 1)[0] + ":"
            try:
                done = self.rc("lsd", remote, timeout=60)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return f"rclone could not reach {remote}: {exc}"
            if done.returncode != 0:
                return f"rclone cannot list {remote} ({done.stderr.strip()[:200]}); run rclone config, or rclone config reconnect {remote}"
            return None
        path = Path(self.remote)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("x")
            probe.unlink()
        except OSError as exc:
            return f"cannot write to {path}: {exc}"
        return None

    def stat(self, name: str) -> dict | None:
        if self.kind == "folder":
            p = Path(self._join(name))
            return {"size": p.stat().st_size, "md5": None} if p.is_file() else None
        done = self.rc("lsjson", "--hash", "--no-modtime", self._join(name))
        if done.returncode != 0:
            return None
        try:
            items = json.loads(done.stdout)
        except ValueError:
            return None
        if not items:
            return None
        hashes = {k.lower(): v for k, v in (items[0].get("Hashes") or {}).items()}
        return {"size": int(items[0].get("Size", -1)), "md5": hashes.get("md5"), "sha1": hashes.get("sha-1") or hashes.get("sha1")}

    def read_text(self, name: str) -> str | None:
        if self.kind == "folder":
            p = Path(self._join(name))
            return p.read_text(encoding="utf-8") if p.is_file() else None
        done = self.rc("cat", self._join(name))
        return done.stdout if done.returncode == 0 else None

    def write_text(self, name: str, text: str) -> None:
        if self.kind == "folder":
            p = Path(self._join(name))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
            return
        proc = subprocess.run([self.rclone, "rcat", self._join(name)], input=text, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise OSError(f"rclone rcat {name}: {proc.stderr.strip()[:200]}")


class HashingWriter:
    def __init__(self, sink):
        self.sink, self.sha, self.md5, self.size = sink, hashlib.sha256(), hashlib.md5(), 0  # noqa: S324

    def write(self, data) -> int:
        self.sha.update(data)
        self.md5.update(data)
        self.size += len(data)
        self.sink.write(data)
        return len(data)

    def flush(self) -> None:
        self.sink.flush()

    def tell(self) -> int:
        return self.size


def model_files(folder: Path, include_ckpt: bool = False) -> list[Path]:
    out = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and (include_ckpt or "ckpt" not in f.relative_to(folder).parts):
            out.append(f)
    return out


def file_sha(path: Path) -> str:
    return lab_space.sha256_of(path)


def write_zip(folder: Path, files: list[Path], sink, name: str) -> HashingWriter:
    out = HashingWriter(sink)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in files:
            arc = f"{name}/{f.relative_to(folder).as_posix()}"
            method = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            info = zipfile.ZipInfo.from_file(f, arc)
            info.compress_type = method
            with open(f, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(MB), b""):
                    dst.write(chunk)
    return out


def build_manifest(folder: Path, files: list[Path], name: str, role: str) -> dict:
    return {"name": name, "role": role, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": lab_space.git_head(),
            "files": {f.relative_to(folder).as_posix(): {"size": f.stat().st_size, "sha256": file_sha(f)} for f in files}}


def verify_zip_file(path: Path, manifest: dict, name: str) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return f"damaged member {bad}"
            for rel, info in manifest["files"].items():
                member = f"{name}/{rel}"
                digest = hashlib.sha256()
                with zf.open(member) as handle:
                    for chunk in iter(lambda: handle.read(MB), b""):
                        digest.update(chunk)
                if digest.hexdigest() != info["sha256"]:
                    return f"{rel} does not match its checksum"
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def upload_model(dest: Destination, folder: Path, name: str, role: str, replace: bool = False) -> dict:
    files = model_files(folder)
    if not files:
        raise OSError(f"{folder} holds no files")
    manifest = build_manifest(folder, files, name, role)
    zip_name, man_name = f"{name}.zip", f"{name}.manifest.json"
    existing = dest.read_text(man_name)
    if existing:
        try:
            old = json.loads(existing)
        except ValueError:
            old = {}
        stat = dest.stat(zip_name)
        if old.get("files") == manifest["files"] and stat and stat["size"] == old.get("zip", {}).get("size"):
            return {"status": "already-there", "zip": old["zip"]}
        if not replace:
            raise OSError(f"{zip_name} is already stored with different contents; --replace overwrites it")
    if dest.kind == "folder":
        target = Path(dest._join(zip_name))
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        with open(part, "wb") as sink:
            out = write_zip(folder, files, sink, name)
            sink.flush()
            os.fsync(sink.fileno())
        stored = part.stat().st_size
        if stored != out.size:
            part.unlink(missing_ok=True)
            raise OSError(f"wrote {stored} bytes, meant {out.size}")
        if file_sha(part) != out.sha.hexdigest():
            part.unlink(missing_ok=True)
            raise OSError("the stored zip does not match what was sent")
        os.replace(part, target)
        problem = verify_zip_file(target, manifest, name)
        if problem:
            raise OSError(f"stored zip failed verification: {problem}")
    else:
        proc = subprocess.Popen([dest.rclone, "rcat", "--drive-chunk-size", "128M", dest._join(zip_name)],
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out = write_zip(folder, files, proc.stdin, name)
            proc.stdin.close()
        except BrokenPipeError as exc:
            proc.wait()
            raise OSError(f"rclone stopped early: {(proc.stderr.read() or b'').decode(errors='replace')[:200]}") from exc
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise OSError(f"rclone rcat failed: {err.strip()[:200]}")
        stat = dest.stat(zip_name)
        if stat is None:
            raise OSError("the zip is not listed at the destination after upload")
        if stat["size"] != out.size:
            raise OSError(f"destination holds {stat['size']} bytes, sent {out.size}")
        if stat.get("md5") and stat["md5"].lower() != out.md5.hexdigest():
            raise OSError("destination checksum (MD5) differs from what was sent")
        if not stat.get("md5"):
            manifest["note"] = "destination reports no MD5: only the size was compared; run `check --paranoid` to download and verify"
    manifest["zip"] = {"name": zip_name, "size": out.size, "sha256": out.sha.hexdigest(), "md5": out.md5.hexdigest()}
    dest.write_text(man_name, json.dumps(manifest, indent=2))
    return {"status": "uploaded", "zip": manifest["zip"]}


def candidates(include_protected: bool, only: list[str] | None) -> list[lab_space.Model]:
    wanted = {"FINISHED"} | ({"PROTECTED"} if include_protected else set())
    models = [m for m in lab_space.scan_models() if m.role in wanted and m.final is not None]
    if only:
        models = [m for m in lab_space.scan_models() if m.path.name in only or m.rel in only]
        missing = [o for o in only if o not in {m.path.name for m in models} | {m.rel for m in models}]
        if missing:
            print("not found: " + ", ".join(missing))
    return sorted(models, key=lambda m: -(m.size - m.ckpt_size))


def free_weights(folder: Path) -> int:
    freed = 0
    for name in BIG_NAMES:
        f = folder / name
        if f.is_file():
            freed += f.stat().st_size
            f.unlink()
    ckpt = folder / "ckpt"
    if ckpt.is_dir():
        freed += lab_space.tree_size(ckpt)
        shutil.rmtree(ckpt, ignore_errors=True)
    return freed


def cmd_list(args) -> int:
    rows = candidates(args.include_protected, args.only)
    total = 0
    for m in rows:
        size = m.size - m.ckpt_size
        total += size
        print(f"  {lab_space.human(size):>9}  {m.role:<9} {m.rel}")
    print(f"\n{len(rows)} models, {lab_space.human(total)} without checkpoints; disk free {lab_space.human(shutil.disk_usage(APP).free)}")
    return 0


def cmd_export(args) -> int:
    dest = Destination.parse(args.to, args.rclone)
    rows = candidates(args.include_protected, args.only)
    if not rows:
        print("Nothing to export (no FINISHED models; --include-protected adds the baselines).")
        return 0
    total = sum(m.size - m.ckpt_size for m in rows)
    mode = "EXPORTING" if args.apply else "DRY RUN (add --apply)"
    print(f"{mode}: {len(rows)} models, {lab_space.human(total)}, to {args.to}"
          + ("; weights are freed after each verified upload" if args.free_after_verify else ""))
    if not args.apply:
        for m in rows:
            print(f"  {lab_space.human(m.size - m.ckpt_size):>9}  {m.role:<9} {m.rel}")
        if args.to.startswith("rclone:") and shutil.which(args.rclone) is None:
            print("\nrclone is not installed here; see --help for the setup")
        return 0
    problem = dest.ready()
    if problem:
        print("NOT READY: " + problem)
        return 2
    failed, done, freed_total = [], 0, 0
    for m in rows:
        if args.until_free_gb and shutil.disk_usage(APP).free >= args.until_free_gb * GB:
            print(f"{args.until_free_gb} GiB free: stopping")
            break
        name = m.path.name
        try:
            result = upload_model(dest, m.path, name, m.role, args.replace)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  FAILED {m.rel}: {exc}   (local files untouched)")
            failed.append(m.rel)
            continue
        done += 1
        line = f"  {result['status']:<13} {m.rel}  ({lab_space.human(result['zip']['size'])})"
        if args.free_after_verify:
            if m.role == "PROTECTED" and not args.free_protected:
                line += "  kept (protected: later stages read it)"
            else:
                freed = free_weights(m.path)
                freed_total += freed
                line += f"  freed {lab_space.human(freed)}"
        print(line, flush=True)
    print(f"\n{done} stored, {len(failed)} failed" + (f", {lab_space.human(freed_total)} freed" if freed_total else ""))
    if failed:
        print("failed: " + ", ".join(failed))
    print(f"disk free now: {lab_space.human(shutil.disk_usage(APP).free)}")
    return 1 if failed else 0


def cmd_check(args) -> int:
    dest = Destination.parse(args.to, args.rclone)
    problem = dest.ready()
    if problem:
        print("NOT READY: " + problem)
        return 2
    names = args.names
    if not names:
        if dest.kind == "folder":
            names = sorted(p.name[: -len(".manifest.json")] for p in Path(dest.remote).glob("*.manifest.json"))
        else:
            done = dest.rc("lsf", dest.remote)
            names = sorted(l[: -len(".manifest.json")] for l in done.stdout.splitlines() if l.endswith(".manifest.json"))
    bad = 0
    for name in names:
        text = dest.read_text(f"{name}.manifest.json")
        stat = dest.stat(f"{name}.zip")
        if not text or not stat:
            print(f"  MISSING   {name}")
            bad += 1
            continue
        manifest = json.loads(text)
        ok = stat["size"] == manifest["zip"]["size"] and (not stat.get("md5") or stat["md5"] == manifest["zip"]["md5"])
        detail = ""
        if ok and args.paranoid:
            with tempfile.TemporaryDirectory(dir=APP / "models") as tmp:
                local = Path(tmp) / f"{name}.zip"
                if not fetch(dest, f"{name}.zip", local):
                    ok, detail = False, "download failed"
                else:
                    issue = verify_zip_file(local, manifest, name)
                    ok, detail = issue is None, issue or "contents verified"
        print(f"  {'ok' if ok else 'DIFFERENT':<9} {name}  {detail}")
        bad += 0 if ok else 1
    print(f"\n{len(names) - bad} of {len(names)} verified")
    return 1 if bad else 0


def fetch(dest: Destination, name: str, local: Path) -> bool:
    if dest.kind == "folder":
        shutil.copyfile(dest._join(name), local)
        return True
    done = dest.rc("copyto", dest._join(name), str(local), timeout=None)
    return done.returncode == 0 and local.is_file()


def cmd_import(args) -> int:
    dest = Destination.parse(args.source, args.rclone)
    problem = dest.ready()
    if problem:
        print("NOT READY: " + problem)
        return 2
    code = 0
    for name in args.names:
        target = APP / "models" / name
        text = dest.read_text(f"{name}.manifest.json")
        if not text:
            print(f"  {name}: no manifest at the source")
            code = 1
            continue
        manifest = json.loads(text)
        have = [rel for rel in manifest["files"] if (target / rel).exists()]
        if have and not args.force:
            print(f"  {name}: {target} already has {len(have)} of its files; --force restores over them")
            code = 1
            continue
        with tempfile.TemporaryDirectory(dir=APP / "models") as tmp:
            local = Path(tmp) / f"{name}.zip"
            if not fetch(dest, f"{name}.zip", local):
                print(f"  {name}: download failed")
                code = 1
                continue
            issue = verify_zip_file(local, manifest, name)
            if issue:
                print(f"  {name}: {issue}; nothing extracted")
                code = 1
                continue
            with zipfile.ZipFile(local) as zf:
                zf.extractall(Path(tmp) / "out")
            target.mkdir(parents=True, exist_ok=True)
            for rel in manifest["files"]:
                src = Path(tmp) / "out" / name / rel
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst) if src.stat().st_dev == dst.parent.stat().st_dev else shutil.move(str(src), str(dst))
        print(f"  {name}: restored to {target} ({len(manifest['files'])} files, checksums verified)")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_to=True):
        if needs_to:
            p.add_argument("--to", required=True, help="rclone:REMOTE:PATH or a folder")
        p.add_argument("--rclone", default="rclone", help="the rclone program (default: found on PATH)")

    p = sub.add_parser("list")
    p.add_argument("--include-protected", action="store_true")
    p.add_argument("--only", nargs="*")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("export")
    common(p)
    p.add_argument("--apply", action="store_true", help="do it (without: a dry run)")
    p.add_argument("--free-after-verify", action="store_true", help="remove each model's weights once its upload verified")
    p.add_argument("--free-protected", action="store_true", help="also free protected baselines after upload (not recommended)")
    p.add_argument("--include-protected", action="store_true", help="also upload the baselines the later stages read")
    p.add_argument("--only", nargs="*", help="model folder names, for example parakeet-ft-lab-09-b2-M01")
    p.add_argument("--until-free-gb", type=float, default=0, help="stop once this much disk is free")
    p.add_argument("--replace", action="store_true", help="overwrite a stored model whose contents differ")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("check")
    common(p)
    p.add_argument("names", nargs="*")
    p.add_argument("--paranoid", action="store_true", help="download each zip and verify every file in it")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("import")
    p.add_argument("names", nargs="+")
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--rclone", default="rclone")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_import)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
