#!/usr/bin/env python3
"""Download and unpack the TORGO corpus (public, free for academic use)."""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli


def project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total and have >= total:
        print(f"  {dest.name}: complete ({have / 1e6:.0f} MB)")
        return
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:
            have = 0
        with open(dest, "ab" if have else "wb") as f:
            done = have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if total and dest.stat().st_size != total:
        raise OSError(f"{dest.name}: got {dest.stat().st_size} bytes, expected {total}; re-run to resume")


def unpack(archive: Path, torgo_root: Path) -> list[str]:
    speakers: set[str] = set()
    with tarfile.open(archive, "r:bz2") as tf:
        for m in tf:
            parts = Path(m.name).parts
            if not parts or parts[0] in ("", ".", ".."):
                continue
            idx = next((i for i, p in enumerate(parts) if len(p) in (3, 4) and p[0] in "FM" and p[-1].isdigit()), None)
            if idx is None:
                continue
            rel = Path(*parts[idx:])
            speakers.add(parts[idx])
            target = torgo_root / rel
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                if target.exists() and target.stat().st_size == m.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
    return sorted(speakers)


def main() -> int:
    from willspeak.torgo import ARCHIVES, BASE_URL, CITATION, SEVERITY

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/public/torgo")
    ap.add_argument("--controls", action="store_true", help="also fetch the control speakers (FC, MC)")
    ap.add_argument("--only", nargs="*", choices=list(ARCHIVES), help="fetch only these archives")
    ap.add_argument("--check", action="store_true", help="report presence; no download")
    a = ap.parse_args()

    out = project_path(a.out)
    torgo = out / "TORGO"
    wanted = a.only or (["F", "M", "FC", "MC"] if a.controls else ["F", "M"])
    if a.check:
        present = sorted(p.name for p in torgo.iterdir() if p.is_dir()) if torgo.is_dir() else []
        print(f"{torgo}: {len(present)} speakers present: {' '.join(present) or '-'}")
        missing = [s for s in SEVERITY if s not in present]
        print(f"missing: {' '.join(missing) or 'none'}")
        return 0
    for key in wanted:
        url = BASE_URL + ARCHIVES[key]
        archive = out / "archives" / ARCHIVES[key]
        print(f"{key}: {url}")
        download(url, archive)
        spk = unpack(archive, torgo)
        print(f"  unpacked {archive.name}: {' '.join(spk)}")
    (out / "SOURCE.md").write_text(
        "# TORGO\n\nDownloaded from " + BASE_URL + " (University of Toronto). Free for academic, "
        "non-profit use; the condition is citation.\n\n" + CITATION + "\n\n"
        "Layout: TORGO/<SPEAKER>/Session*/{prompts,wav_headMic,wav_arrayMic}/. "
        "Manifests and exports are built by scripts/corpus/torgo_manifest.py.\n", encoding="utf-8")
    print(f"done -> {torgo}\nnext: python scripts/corpus/torgo_manifest.py --export")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "fetch_torgo"))
