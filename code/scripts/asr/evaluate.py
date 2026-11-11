#!/usr/bin/env python3
"""Evaluate Parakeet or an existing hypotheses file on an exported split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _cli
from willspeak import report as report_builder
from willspeak.artifacts import (
    contained_file,
    file_record,
    model_records,
    new_directory,
    record_id,
    timestamp,
    versions,
    write_json,
)
from willspeak.asr.parakeet import ParakeetASR, default_model_dir
from willspeak.evaluate import (
    evaluate,
    load_manifest,
    pairs_from,
    phoneme_difficulty,
    transcribe_parakeet,
)

DECODE_FORMATS = ("willspeak-nemo-hypotheses-v1", "willspeak-hf-hypotheses-v1")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def machine_info() -> dict:
    import platform
    import socket

    info = {"hostname": socket.gethostname(), "platform": platform.platform(),
            "machine": platform.machine(), "python": platform.python_version(), "cpu": platform.processor() or ""}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware", "model\t")):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return info


def load_hypotheses(path: Path) -> dict[str, str]:
    hypotheses: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in hypotheses or not isinstance(row.get("hyp"), str):
                raise ValueError(f"Duplicate id or missing/non-text hypothesis: {key}")
            hypotheses[key] = row["hyp"]
    return hypotheses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/export/test.jsonl")
    parser.add_argument("--audio-root", default="data/export")
    parser.add_argument("--model-dir", help="stock model by default; accepts a fine-tuned ONNX export")
    parser.add_argument("--hyp", help="JSONL rows shaped as {id, hyp}; skips recognition")
    parser.add_argument("--group-by", default="prompt_set",
                        help="manifest field to stratify by (prompt_set, condition, speaker, severity, kind, session)")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N manifest rows (plumbing check)")
    parser.add_argument("--out", default="reports/parakeet")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    try:
        return run_evaluation(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


def run_evaluation(args) -> int:
    manifest_path = project_path(args.manifest)
    output_dir = project_path(args.out)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new report directory: {output_dir}")
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}\nExport the corpus before evaluating.", file=sys.stderr)
        return 1
    if manifest_path.name == "train.jsonl":
        print("Warning: this is the training split. Use test.jsonl for an unbiased result.")

    manifest = load_manifest(manifest_path)
    if not manifest:
        print(f"Manifest is empty: {manifest_path}", file=sys.stderr)
        return 1
    if getattr(args, "limit", 0):
        manifest = dict(list(manifest.items())[: args.limit])
        print(f"--limit {args.limit}: scoring {len(manifest)} rows only; this is a plumbing check, not a result")
    if not any(getattr(args, "group_by", "prompt_set") in row for row in manifest.values()):
        print(f"Warning: no manifest row has a '{args.group_by}' field; the report will have one group")

    input_paths = {"manifest": manifest_path}
    input_paths.update(
        {
            f"audio/{row['audio']}": contained_file(project_path(args.audio_root), row["audio"])
            for row in manifest.values()
        }
    )
    provenance = {
        "format": "willspeak-evaluation-v1",
        "created_at": timestamp(),
        "split": manifest_path.stem,
        "utterances": len(manifest),
        "packages": versions(),
        "group_by": args.group_by,
        "limit": getattr(args, "limit", 0) or None,
        "machine": machine_info(),
    }
    provenance["code"] = {
        name: file_record(ROOT / "src/willspeak" / name)
        for name in ("evaluate.py", "normalize.py", "us_spelling.json", "g2p.py", "audio.py", "asr/parakeet.py")
    }
    provenance["code"]["scripts/asr/evaluate.py"] = file_record(Path(__file__))
    model_dir = None
    if args.hyp:
        hypothesis_path = project_path(args.hyp)
        if not hypothesis_path.is_file():
            print(f"Hypotheses file not found: {hypothesis_path}", file=sys.stderr)
            return 1
        input_paths["imported_hypotheses"] = hypothesis_path
        decode_provenance = hypothesis_path.with_suffix(".provenance.json")
        if decode_provenance.exists():
            decoded = json.loads(decode_provenance.read_text(encoding="utf-8"))
            if (decoded.get("format") not in DECODE_FORMATS
                    or decoded.get("hypotheses") != file_record(hypothesis_path)
                    or decoded.get("inputs") != {name: file_record(path) for name, path in input_paths.items()
                                                  if name != "imported_hypotheses"}):
                raise ValueError("Imported hypotheses do not match their recorded manifest/audio or decode provenance")
            input_paths["decode_provenance"] = decode_provenance
            provenance["decode"] = decoded
        else:
            print("Warning: imported hypotheses have no decode provenance; their original audio/model cannot be verified.")
        provenance["recognition"] = "imported"
        model_name = f"imported:{hypothesis_path.name}"
    else:
        model_dir = project_path(args.model_dir) if args.model_dir else default_model_dir()
        audio_root = project_path(args.audio_root)
        provenance["recognition"] = "parakeet"
        provenance["model_files"] = model_records(model_dir)
        settings = ParakeetASR(model_dir)
        provenance["decoder"] = {
            "method": "greedy_search",
            "threads": settings.threads,
            "provider": "cpu",
            "feature_dim": 128,
        }
        model_name = f"parakeet:{model_dir.name}"
    provenance["inputs"] = {name: file_record(path) for name, path in input_paths.items()}
    provenance["input_id"] = record_id(provenance["inputs"])
    started = time.perf_counter()
    if args.hyp:
        hypotheses = load_hypotheses(hypothesis_path)
    else:
        print(f"Transcribing {len(manifest)} clips with Parakeet…")
        hypotheses = transcribe_parakeet(manifest, audio_root, model_dir=model_dir)
    provenance["recognition_seconds_including_load"] = time.perf_counter() - started
    if set(hypotheses) != set(manifest):
        raise ValueError("Hypothesis IDs must match the manifest exactly (use an explicit empty hyp for a deletion)")
    if provenance["inputs"] != {name: file_record(path) for name, path in input_paths.items()}:
        raise ValueError("Evaluation inputs changed during recognition; retry on a fixed snapshot")
    if model_dir and provenance["model_files"] != model_records(model_dir):
        raise ValueError("Model files changed during recognition")

    result = evaluate(pairs_from(manifest, hypotheses, group_key=args.group_by))
    difficulty = phoneme_difficulty(result)
    with new_directory(output_dir) as stage:
        write_json(stage / "results.json", result.dict())
        write_json(stage / "difficulty.json", {
            "format": "willspeak-difficulty-v1", "source_split": manifest_path.stem,
            "manifest": provenance["inputs"]["manifest"], "weights": difficulty,
        })
        (stage / "hypotheses.jsonl").write_text(
            "".join(json.dumps({"id": key, "hyp": value}) + "\n" for key, value in hypotheses.items()),
            encoding="utf-8",
        )
        report_builder.build(
            result,
            {
                "label": args.label or manifest_path.stem,
                "model": model_name,
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "split": manifest_path.stem,
                "difficulty": difficulty,
            },
            stage / "report.html",
        )
        provenance["outputs"] = {path.name: file_record(path) for path in stage.iterdir()}
        write_json(stage / "provenance.json", provenance)

    print(f"WER {result.wer * 100:.2f}%  CER {result.cer * 100:.2f}%  PER {result.per * 100:.2f}%")
    print(f"Report: {output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli.run(main, "evaluate"))
