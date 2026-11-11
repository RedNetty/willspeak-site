from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def record_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def portable_name(name: str) -> str:
    parts = PurePosixPath(name).parts
    reserved = {"con", "prn", "aux", "nul"} | {f"{p}{n}" for p in ("com", "lpt") for n in range(1, 10)}
    if not parts or name.startswith("/") or "/".join(parts) != name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    for part in parts:
        if (
            part in {".", ".."}
            or part.endswith((".", " "))
            or re.search(r'[\\:<>"|?*\x00-\x1f]', part)
            or part.split(".")[0].lower() in reserved
        ):
            raise ValueError(f"Unsafe relative path: {name!r}")
    return name


def contained_file(root: Path, name: str) -> Path:
    portable_name(name)
    root = root.resolve()
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"File escapes its root: {name}")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Symlinks are not supported: {name}")
        cursor = cursor.parent
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def archive_members(archive: zipfile.ZipFile) -> dict:
    infos = archive.infolist()
    if len(infos) > 100_000 or sum(i.file_size for i in infos) > 20 * 1024**3:
        raise ValueError("Archive exceeds the 100,000-file / 20 GiB extraction limit")
    names, folded, prefixes = {}, set(), {}
    for info in infos:
        name = portable_name(info.filename)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
            raise ValueError(f"Only regular files are allowed: {name}")
        if name.casefold() in folded:
            raise ValueError(f"Duplicate or case-colliding archive path: {name}")
        for prefix in (PurePosixPath(name), *PurePosixPath(name).parents):
            key, spelling = str(prefix).casefold(), str(prefix)
            if key in prefixes and prefixes[key] != spelling:
                raise ValueError(f"Case-colliding archive directory: {name}")
            prefixes[key] = spelling
        names[name] = info
        folded.add(name.casefold())
    for name in names:
        if any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError(f"Archive file/directory conflict: {name}")
    return names


@contextmanager
def new_directory(destination: Path):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists; choose a new name: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        stage = Path(temp)
        yield stage
        destination.mkdir()
        try:
            for item in stage.iterdir():
                os.rename(item, destination / item.name)
        except BaseException:
            shutil.rmtree(destination)
            raise


def versions() -> dict:
    result = {}
    for package in (
        "willspeak",
        "sherpa-onnx",
        "numpy",
        "soundfile",
        "cmudict",
        "rapidfuzz",
        "torch",
        "nemo_toolkit",
        "lightning",
        "onnx",
        "onnxruntime",
        "omegaconf",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def model_records(directory: Path) -> dict:
    from .asr.parakeet import model_files

    files = model_files(directory)
    if files is None:
        raise FileNotFoundError(f"Incomplete Parakeet model: {directory}")
    records = {path.name: file_record(path) for path in files.values()}
    if any(path.name in {"encoder.onnx", "decoder.onnx", "joiner.onnx"} for path in files.values()):
        for path in directory.iterdir():
            if path.suffix in {".weights", ".data"}:
                records[path.name] = file_record(path)
    if any(v["size"] == 0 for v in records.values()):
        raise ValueError("Model files must not be empty")
    return records


def validate_export(root: Path, splits: tuple[str, ...] = ("train", "dev", "test")) -> tuple[dict, dict]:
    import soundfile as sf

    from .db import is_placeholder
    from .evaluate import load_manifest
    from .normalize import normalize

    manifests, records = {}, {}
    ids, prompts, audio_hashes = {}, {}, {}
    for split in splits:
        name = f"{split}.jsonl"
        manifest_path = contained_file(root, name)
        rows = load_manifest(manifest_path)
        if not rows:
            raise ValueError(f"{name} is empty; collect/export held-out prompts before training")
        records[name] = file_record(manifest_path)
        for clip_id, row in rows.items():
            if row.get("prompt_set") in {"critical", "probe"} and split != "train":
                raise ValueError(f"Critical/probe clips belong in train, not {split}: {clip_id}")
            if clip_id in ids:
                raise ValueError(f"Clip {clip_id} appears in both {ids[clip_id]} and {split}")
            ids[clip_id] = split
            if is_placeholder(row["text"]):
                raise ValueError(f"Clip {clip_id} carries the live placeholder instead of a transcript")
            texts = [row["text"]]
            if not is_placeholder(row.get("prompt_text")):
                texts.append(row.get("prompt_text", row["text"]))
            for text in texts:
                key = normalize(text, us_spelling=False)
                if not key:
                    raise ValueError(f"Empty normalized transcript for clip {clip_id}")
                if key in prompts and prompts[key] != split:
                    raise ValueError(f"Prompt leakage between {prompts[key]} and {split}: clip {clip_id}")
                prompts[key] = split
            audio_name = row["audio"]
            if not audio_name.startswith(f"{split}/"):
                raise ValueError(f"Clip {clip_id} audio must be under {split}/")
            audio_path = contained_file(root, audio_name)
            info = sf.info(audio_path)
            duration = float(row["duration_s"])
            if (
                not math.isfinite(duration)
                or duration <= 0
                or info.frames <= 0
                or info.samplerate != 16000
                or info.channels != 1
                or abs(info.duration - duration) > 0.1
            ):
                raise ValueError(f"Invalid duration or non-mono/16 kHz audio: {audio_name}")
            record = file_record(audio_path)
            digest = record["sha256"]
            if digest in audio_hashes and audio_hashes[digest] != split:
                raise ValueError(f"Identical audio in {audio_hashes[digest]} and {split}: {audio_name}")
            audio_hashes[digest] = split
            records[audio_name] = record
        manifests[split] = rows
    return manifests, records
