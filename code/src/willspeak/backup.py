from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
from uuid import uuid4

from .artifacts import archive_members, new_directory


def make_backup(store, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    bdir = data_dir / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    out = bdir / f"willspeak-backup-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex}.zip"
    partial = out.with_suffix(".partial")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tmp_db = Path(tf.name)
    try:
        dst = sqlite3.connect(tmp_db)
        try:
            with store._lock:
                store.con.commit()
                store.con.backup(dst)
        finally:
            dst.close()

        with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED) as z:
            z.write(tmp_db, "corpus.db")
            for sub in ("clips", "raw"):
                d = data_dir / sub
                if d.exists():
                    for f in sorted(d.rglob("*")):
                        if f.is_file():
                            z.write(f, f"{sub}/{f.relative_to(d)}")
            for name in ("split_assignments.json", "live_settings.json"):
                settings = data_dir / name
                if settings.exists():
                    z.write(settings, name)
        partial.rename(out)
    finally:
        tmp_db.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return out


def backup_stats(data_dir: str | Path) -> dict:
    bdir = Path(data_dir) / "backups"
    zips = sorted(bdir.glob("willspeak-backup-*.zip"), key=lambda path: path.stat().st_mtime_ns) if bdir.exists() else []
    last = zips[-1] if zips else None
    return {
        "count": len(zips),
        "last": last.name if last else None,
        "last_bytes": last.stat().st_size if last else 0,
    }


def restore_backup(archive_path: Path, destination: Path, *, dry_run: bool = False) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive)
        if "corpus.db" not in members:
            raise ValueError("Backup has no corpus.db at its root")
        if any(
            name not in {"corpus.db", "split_assignments.json", "live_settings.json"}
            and not name.startswith(("clips/", "raw/"))
            for name in members
        ):
            raise ValueError(
                "Unexpected file in backup; only corpus.db, settings, split assignments, clips, and raw audio are allowed"
            )
        summary = {
            "files": len(members),
            "bytes": sum(i.file_size for i in members.values()),
            "destination": str(destination.resolve()),
            "dry_run": dry_run,
        }
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Restore requires a new directory: {destination}")
        if dry_run:
            if archive.testzip() is not None:
                raise ValueError("Backup failed ZIP integrity checks")
            return summary
        with new_directory(destination) as stage:
            for name in members:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            database = sqlite3.connect((stage / "corpus.db").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
            try:
                if database.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("Restored SQLite database failed its integrity check")
                tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"clip", "prompt", "session"}.issubset(tables):
                    raise ValueError("Archive database is not a WillSpeak corpus")
            finally:
                database.close()
        return summary
