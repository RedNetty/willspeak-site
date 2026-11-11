#!/usr/bin/env python3
"""Restore a willspeak backup zip into a data directory."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.backup import restore_backup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, help="willspeak-backup-*.zip")
    ap.add_argument("--into", type=Path, required=True, help="new recovery directory")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    try:
        summary = restore_backup(ROOT / a.archive, ROOT / a.into, dry_run=a.dry_run)
    except (OSError, ValueError, sqlite3.Error, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    if not a.dry_run:
        print("Restored into a new directory. Keep the original corpus until recovery has been verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
