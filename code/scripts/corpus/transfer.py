#!/usr/bin/env python3
"""Create, verify, and unpack private corpus/model bundles."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.bundles import inspect_bundle, pack_corpus, pack_model, unpack_bundle


def project_path(value: str) -> Path:
    return ROOT / value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("pack-corpus", help="accepted exported clips, manifests, sticky splits only")
    corpus.add_argument("--export", type=project_path, default=ROOT / "data/export")
    corpus.add_argument("--assignments", type=project_path, default=ROOT / "data/split_assignments.json")
    corpus.add_argument("--out", type=project_path, required=True)
    model = commands.add_parser("pack-model", help="int8 model plus matching unbiased test evaluation")
    model.add_argument("--model-dir", type=project_path, required=True)
    model.add_argument("--evaluation", type=project_path, required=True)
    model.add_argument("--out", type=project_path, required=True)
    inspect = commands.add_parser("inspect", help="verify every file without extracting")
    inspect.add_argument("bundle", type=project_path)
    unpack = commands.add_parser("unpack", help="verify and extract into a new folder")
    unpack.add_argument("bundle", type=project_path)
    unpack.add_argument("--into", type=project_path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack-corpus":
            result = pack_corpus(args.export, args.assignments, args.out)
        elif args.command == "pack-model":
            result = pack_model(args.model_dir, args.evaluation, args.out)
        elif args.command == "inspect":
            result = inspect_bundle(args.bundle)
        else:
            result = unpack_bundle(args.bundle, args.into)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Transfer failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    print(f"Verified {len(result['files'])} files. Bundles contain private data; they are not encrypted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
