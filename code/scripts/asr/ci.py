#!/usr/bin/env python3
"""Paired bootstrap intervals for corpus WER and stock-minus-finetuned WER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.artifacts import file_record, record_id, write_json
from willspeak.evaluate import OK, align, load_manifest
from willspeak.normalize import normalize

N_RESAMPLES = 2000
SEED = 0


def load_hypotheses(report_dir: Path) -> dict[str, str]:
    out = {}

    path = report_dir / "hypotheses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No hypotheses.jsonl in {report_dir}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                key = str(r["id"])
                if key in out or not isinstance(r.get("hyp"), str):
                    raise ValueError(f"Duplicate id or missing/non-text hypothesis: {key}")
                out[key] = r["hyp"]
    return out



def utterance_counts(manifest: dict[str, dict], hyps: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    lengths = []

    for clip_id, row in manifest.items():
        reference_words = normalize(row["text"]).split()
        if not reference_words:
            raise ValueError(f"Clip {clip_id} has no reference words; score silence separately from WER intervals")
        hypotheses_words = normalize(hyps[clip_id]).split()
        ops = align(reference_words, hypotheses_words)

        count = 0
        for op in ops:
            if op[0] != OK:
                count += 1
        errors.append(count)
        lengths.append(len(reference_words))
    return np.array(errors), np.array(lengths)




def corpus_wer(errors: np.ndarray, lengths: np.ndarray) -> float:
    total_errors = np.sum(errors)
    total_length = np.sum(lengths)
    if total_length <= 0:
        raise ValueError("WER needs at least one reference word")
    return total_errors / total_length


def bootstrap(stock_errors: np.ndarray, lengths: np.ndarray,
              finetuned_errors: np.ndarray | None = None,
              n_resamples: int = N_RESAMPLES, seed: int = SEED,
              clusters: list[tuple] | None = None) -> dict[str, np.ndarray]:
    arrays = [stock_errors, lengths] + ([] if finetuned_errors is None else [finetuned_errors])
    if (n_resamples < 1 or any(a.ndim != 1 or a.shape != lengths.shape for a in arrays)
            or lengths.size == 0 or np.any(lengths <= 0)
            or any(not np.all(np.isfinite(a)) or np.any(a < 0) for a in arrays)):
        raise ValueError("Bootstrap requires matching finite counts, positive reference lengths, and resamples")
    if clusters is not None:
        if len(clusters) != len(lengths):
            raise ValueError("Each clip needs a cluster")
        units = list(dict.fromkeys(clusters))
        if len(units) < 2:
            raise ValueError("Cluster intervals need at least two distinct clusters; one session cannot estimate session spread")
        membership = {unit: i for i, unit in enumerate(units)}
        indices = [membership[unit] for unit in clusters]

        def totals(values):
            return np.bincount(indices, weights=values, minlength=len(units))

        stock_errors, lengths = totals(stock_errors), totals(lengths)
        if finetuned_errors is not None:
            finetuned_errors = totals(finetuned_errors)
    rng = np.random.default_rng(seed)
    n = len(stock_errors)


    stock_wers = []
    finetuned_wers = []
    deltas = []

    for _ in range(n_resamples):
        index = rng.integers(0, n, n)

        s = corpus_wer(stock_errors[index], lengths[index])
        stock_wers.append(s)

        if finetuned_errors is not None:
            a = corpus_wer(finetuned_errors[index], lengths[index])
            finetuned_wers.append(a)
            deltas.append(s - a)

    result = {"stock": np.array(stock_wers)}
    if finetuned_errors is not None:
        result["finetuned"] = np.array(finetuned_wers)
        result["delta"] = np.array(deltas)
    return result



def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return low, high


def cluster_keys(manifest: dict[str, dict], fields: list[str]) -> list[tuple]:
    keys = []
    for clip_id, row in manifest.items():
        values = []
        for field in fields:
            value = row.get(field)
            if not isinstance(value, (str, int)) or isinstance(value, bool) or str(value).strip() == "":
                raise ValueError(f"Clip {clip_id} needs a nonempty {field!r} field for clustering")
            values.append(normalize(str(value)) if field in {"text", "prompt_text"} else str(value))
        keys.append(tuple(values))
    return keys


def check_report(folder: Path, measured: float, manifest_record: dict, rows: dict) -> dict:
    paths = {name: folder / name for name in ("hypotheses.jsonl", "results.json")}
    records = {name: file_record(path) for name, path in paths.items()}
    saved = json.loads(paths["results.json"].read_text(encoding="utf-8"))["wer"]
    if not np.isfinite(saved) or abs(measured - saved) > 0.000051:
        raise ValueError(f"WER disagrees with {paths['results.json']}: {measured:.6f} vs {saved}")
    provenance_path = folder / "provenance.json"
    if not provenance_path.is_file():
        return {"path": str(folder.resolve()), "files": records, "provenance": "unavailable"}
    records["provenance.json"] = file_record(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    inputs = provenance.get("inputs", {})
    if inputs.get("manifest") != manifest_record:
        raise ValueError(f"Manifest bytes differ from the report in {folder}")
    if provenance.get("limit"):
        raise ValueError(f"A --limit plumbing check is not a full test report: {folder}")
    for name in ("hypotheses.jsonl", "results.json"):
        if records[name] != provenance.get("outputs", {}).get(name):
            raise ValueError(f"Report output changed: {folder / name}")
    audio = {clip_id: inputs.get(f"audio/{row.get('audio')}") for clip_id, row in rows.items()}
    if any(value is None for value in audio.values()):
        raise ValueError(f"Report is missing audio provenance: {folder}")
    return {"path": str(folder.resolve()), "files": records, "audio": audio}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--finetuned")
    parser.add_argument("--cluster-by", nargs="+", default=[], metavar="FIELD",
                        help="resample whole groups, e.g. text, speaker, or speaker session")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest_record = file_record(manifest_path)
    manifest = load_manifest(manifest_path)

    stock_hyps = load_hypotheses(Path(args.stock))
    if set(stock_hyps) != set(manifest):
        print("The stock report was made on a different manifest.")
        return 1

    finetuned_hyps = None
    if args.finetuned:
        finetuned_hyps = load_hypotheses(Path(args.finetuned))
        if set(finetuned_hyps) != set(manifest):
            print("The fine-tuned report was made on a different manifest.")
            return 1

    stock_errors, lengths = utterance_counts(manifest, stock_hyps)
    finetuned_errors = None
    if finetuned_hyps is not None:
        finetuned_errors, _ = utterance_counts(manifest, finetuned_hyps)

    stock_wer = corpus_wer(stock_errors, lengths)
    comparisons = [(Path(args.stock), stock_wer)]
    if finetuned_errors is not None:
        comparisons.append((Path(args.finetuned), corpus_wer(finetuned_errors, lengths)))
    sources = [check_report(folder, measured, manifest_record, manifest) for folder, measured in comparisons]
    if len(sources) == 2 and all("audio" in source for source in sources) and sources[0]["audio"] != sources[1]["audio"]:
        raise ValueError("The reports decoded different audio bytes; a paired comparison needs the same clips")
    if any(source.get("provenance") == "unavailable" for source in sources):
        print("Warning: legacy report has no provenance; original manifest/audio correspondence cannot be verified.")

    clusters = cluster_keys(manifest, args.cluster_by) if args.cluster_by else None
    draws = bootstrap(stock_errors, lengths, finetuned_errors, clusters=clusters)
    stock_low, stock_high = interval(draws["stock"])
    n = len(lengths)
    print(f"n = {n}")
    print(f"Baseline: {args.stock}")
    if args.finetuned:
        print(f"Candidate: {args.finetuned}")
    units = len(set(clusters)) if clusters is not None else n
    print(f"Paired percentile bootstrap: {units} {'/'.join(args.cluster_by) or 'clip'} units, {N_RESAMPLES} resamples, seed {SEED}")
    if units < 20:
        print("Warning: few resampling units; interval endpoints may be unstable.")
    if not args.cluster_by and len({normalize(row['text']) for row in manifest.values()}) < n:
        print("Warning: repeated transcripts are resampled separately; --cluster-by text keeps their takes together.")
    print(f"Baseline WER {stock_wer * 100:.2f}%  (95% CI {stock_low * 100:.2f}% to {stock_high * 100:.2f}%)")

    out = {"n": n, "n_resamples": N_RESAMPLES, "seed": SEED,
           "method": "paired-percentile-bootstrap", "cluster_by": args.cluster_by, "n_units": units,
           "manifest": {"path": str(manifest_path.resolve()), **manifest_record},
           "stock_report": sources[0],
           "scoring": {name: file_record(ROOT / "src/willspeak" / name)
                       for name in ("evaluate.py", "normalize.py", "us_spelling.json")},
           "stock_wer": stock_wer, "stock_ci": [stock_low, stock_high]}
    out_dir = Path(args.stock)

    if finetuned_errors is not None:
        finetuned_wer = corpus_wer(finetuned_errors, lengths)
        delta = stock_wer - finetuned_wer
        delta_low, delta_high = interval(draws["delta"])
        print(f"Candidate WER {finetuned_wer * 100:.2f}%")
        print(f"Gain (baseline - candidate) {delta * 100:.2f} points  (95% CI {delta_low * 100:.2f} to {delta_high * 100:.2f})")
        out.update({"finetuned_wer": finetuned_wer, "delta": delta,
                    "delta_ci": [delta_low, delta_high], "finetuned_report": sources[1]})
        out_dir = Path(args.finetuned)

    if file_record(manifest_path) != manifest_record:
        raise ValueError("Manifest changed while computing the interval")
    for source in sources:
        if source["files"] != {name: file_record(Path(source["path"]) / name) for name in source["files"]}:
            raise ValueError("Report changed while computing the interval")
    comparison_id = record_id(out)
    out["comparison_id"] = comparison_id
    history = out_dir / "ci" / f"{comparison_id}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    write_json(history, out)
    write_json(out_dir / "ci.json", out)
    print(f"Saved comparison: {history}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Confidence interval failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
