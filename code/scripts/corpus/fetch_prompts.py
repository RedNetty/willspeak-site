#!/usr/bin/env python3
"""Download the CMU ARCTIC phonetic prompt list into prompts/."""
from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "prompts"

SOURCES = {
    "phonetic_arctic.txt": [
        "http://festvox.org/cmu_arctic/cmuarctic.data",
        "https://raw.githubusercontent.com/festvox/festvox/master/src/vox_files/cmuarctic.data",
    ],
}

ARCTIC = re.compile(r'\(\s*\S+\s+"(.+?)"\s*\)')


def fetch(url: str, timeout: int = 45) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print(f"  ! {url}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for fname, urls in SOURCES.items():
        print(f"{fname}:")
        text = None
        for u in urls:
            print(f"  trying {u}")
            text = fetch(u)
            if text:
                break
        if not text:
            print("  FAILED -download cmuarctic.data by hand from "
                  "http://festvox.org/cmu_arctic/ and re-run.")
            continue
        lines = ARCTIC.findall(text)
        if not lines:
            lines = [l.strip() for l in text.splitlines()
                     if l.strip() and not l.startswith("#")]
        p = OUT / fname
        p.write_text(
            "# Auto-fetched phonetic prompt set. Selected for diphone coverage.\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {len(lines)} prompts -> {p}")
        total += len(lines)

    print(f"\n{total} prompts fetched. Restart the server to load them.")
    print("Harvard/IEEE lists 1-3 already ship in prompts/harvard_seed.txt;")
    print("the full 720 are widely mirrored if you want them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
