from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .artifacts import (
    archive_members,
    contained_file,
    file_record,
    model_records,
    new_directory,
    portable_name,
    record_id,
    timestamp,
    validate_export,
)
from .export import split_key

FORMAT = "willspeak-bundle"
MODEL_NAMES = {f"{part}.int8.onnx" for part in ("encoder", "decoder", "joiner")} | {"tokens.txt"}
REPORT_NAMES = {"provenance.json", "results.json", "difficulty.json", "hypotheses.jsonl", "report.html"}


def _pack(files: dict[str, Path], destination: Path, kind: str, details: dict, expected: dict) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Bundle already exists: {destination}")
    records = {portable_name(name): file_record(path) for name, path in sorted(files.items())}
    if records != expected:
        raise ValueError("Source files changed during validation; retry with a fixed snapshot")
    manifest = {
        "format": FORMAT,
        "version": 1,
        "kind": kind,
        "created_at": timestamp(),
        "files": records,
        "details": details,
    }
    manifest["bundle_id"] = record_id({"kind": kind, "files": records, "details": details})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=destination.parent) as temp:
        staged = Path(temp) / "bundle.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("bundle.json", json.dumps(manifest, indent=2) + "\n")
            for name, path in files.items():
                archive.write(path, name)
        inspect_bundle(staged)
        with destination.open("xb") as output:
            try:
                with staged.open("rb") as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
    return manifest


def pack_corpus(export: Path, assignments: Path, destination: Path) -> dict:
    rows, records = validate_export(export)
    assignment_record = file_record(assignments)
    split_map = json.loads(assignments.read_text(encoding="utf-8"))
    if not isinstance(split_map, dict) or any(v not in {"train", "dev", "test"} for v in split_map.values()):
        raise ValueError("Invalid split_assignments.json")
    for split, manifest in rows.items():
        for row in manifest.values():
            if row.get("prompt_set") in {"critical", "probe"}:
                continue
            key = split_key(row.get("prompt_text", row["text"]), row["text"])
            if split_map.get(key) != split:
                raise ValueError(f"Sticky assignment does not match {split}: clip {row['id']}")
    files = {f"export/{name}": export / name for name in records}
    files["split_assignments.json"] = assignments
    details = {
        "split_policy": "prompt-disjoint",
        "counts": {k: len(v) for k, v in rows.items()},
        "dataset_id": record_id(records),
    }
    expected = {f"export/{name}": record for name, record in records.items()}
    expected["split_assignments.json"] = assignment_record
    return _pack(files, destination, "corpus", details, expected)


def _release_records(model: Path, evaluation: Path) -> tuple[dict, dict, dict]:
    records = model_records(model)
    if set(records) != MODEL_NAMES:
        raise ValueError("Deployable bundles require the self-contained int8 ONNX trio and tokens.txt")
    provenance_path = contained_file(evaluation, "provenance.json")
    report_records = {"provenance.json": file_record(provenance_path)}
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("format") != "willspeak-evaluation-v1"
        or provenance.get("recognition") != "parakeet"
        or provenance.get("split") != "test"
        or provenance.get("model_files") != records
        or not provenance.get("inputs")
    ):
        raise ValueError(
            "Evaluation must decode test.jsonl with these exact model files; imported hypotheses do not qualify"
        )
    for name in REPORT_NAMES - {"provenance.json"}:
        report_records[name] = file_record(contained_file(evaluation, name))
        if report_records[name] != provenance.get("outputs", {}).get(name):
            raise ValueError(f"Evaluation output changed or was not recorded: {name}")
    return records, provenance, report_records


def pack_model(model: Path, evaluation: Path, destination: Path) -> dict:
    records, provenance, report_records = _release_records(model, evaluation)
    files = {f"model/{name}": model / name for name in records}
    files.update({f"evaluation/{name}": evaluation / name for name in REPORT_NAMES})
    expected = {f"model/{name}": record for name, record in records.items()}
    expected.update({f"evaluation/{name}": record for name, record in report_records.items()})
    for name in ("LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt", "README.md", "export.json", "training.json"):
        if (model / name).is_file():
            files[f"model/{name}"] = model / name
            expected[f"model/{name}"] = file_record(model / name)
    details = {
        "model_id": record_id(records),
        "evaluation_id": record_id(provenance),
        "release_gate": "evaluated, not automatically approved for client use",
    }
    return _pack(files, destination, "model", details, expected)


def _members(archive: zipfile.ZipFile) -> dict:
    names = archive_members(archive)
    if "bundle.json" not in names or names["bundle.json"].file_size > 4 * 1024**2:
        raise ValueError("Missing or oversized bundle.json")
    return names


def _verify(archive: zipfile.ZipFile, stage: Path | None = None) -> dict:
    members = _members(archive)
    manifest = json.loads(archive.read("bundle.json"))
    if not isinstance(manifest, dict):
        raise TypeError("Bundle manifest must be an object")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("version") != 1
        or manifest.get("kind") not in {"corpus", "model"}
    ):
        raise ValueError("Unsupported bundle format, version, or kind")
    records = manifest["files"]
    if not isinstance(records, dict) or not isinstance(manifest.get("details"), dict):
        raise TypeError("Invalid bundle inventory or metadata")
    if set(members) != set(records) | {"bundle.json"}:
        raise ValueError("Archive contents do not match the bundle inventory")
    expected_id = record_id({"kind": manifest["kind"], "files": records, "details": manifest["details"]})
    if manifest.get("bundle_id") != expected_id:
        raise ValueError("Bundle manifest checksum does not match")
    for name, record in records.items():
        if members[name].file_size != record["size"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        output = None
        try:
            if stage is not None:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open("xb")
            with archive.open(name) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    if output:
                        output.write(block)
        finally:
            if output:
                output.close()
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    if stage is not None:
        (stage / "bundle.json").write_bytes(archive.read("bundle.json"))
    return manifest


def inspect_bundle(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return _verify(archive)


def unpack_bundle(path: Path, destination: Path) -> dict:
    with zipfile.ZipFile(path) as archive, new_directory(destination) as stage:
        manifest = _verify(archive, stage)
        if manifest["kind"] == "corpus":
            _, records = validate_export(stage / "export")
            if record_id(records) != manifest["details"]["dataset_id"]:
                raise ValueError("Dataset identity mismatch")
        else:
            records, provenance, _ = _release_records(stage / "model", stage / "evaluation")
            if record_id(records) != manifest["details"]["model_id"]:
                raise ValueError("Model identity mismatch")
            if record_id(provenance) != manifest["details"]["evaluation_id"]:
                raise ValueError("Evaluation identity mismatch")
    return manifest
