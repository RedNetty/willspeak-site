#!/usr/bin/env python3
"""Which encoder layer, of which model, matches his words best across days? Lab test 5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import _cli
import ci
import labkit as lab
from willspeak.matcher import training as S


def load_archive(path: str) -> dict:
    z = np.load(lab.here(path), allow_pickle=True)
    raw = z["raw"].astype(np.float32)
    layered = raw.ndim == 3
    if not layered:
        raw = raw[:, None, :]
    return {"raw": raw, "layers": [int(x) for x in z["layers"]] if layered and "layers" in z.files else list(range(raw.shape[1])),
            "layered": layered, "labels": z["labels"].astype(str), "sessions": z["sessions"].astype(str),
            "speakers": z["speakers"].astype(str), "paths": np.array([str(lab.here(x)) for x in z["paths"].astype(str)])}


def align(archives: dict[str, dict]) -> dict[str, dict]:
    names = list(archives)
    common = set(archives[names[0]]["paths"])
    for name in names[1:]:
        common &= set(archives[name]["paths"])
    if not common:
        raise ValueError("The archives share no clips; embed the same manifest with every model")
    order = [p for p in archives[names[0]]["paths"] if p in common]
    out = {}
    for name, d in archives.items():
        position = {p: i for i, p in enumerate(d["paths"])}
        index = np.array([position[p] for p in order])
        out[name] = {**d, "raw": d["raw"][index], "labels": d["labels"][index], "sessions": d["sessions"][index],
                     "speakers": d["speakers"][index], "paths": d["paths"][index]}
    return out


def layer_spec(spec: str) -> set[int] | None:
    if spec == "all":
        return None
    chosen: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        chosen.update(range(int(low), int(high or low) + 1))
    return chosen


def plan_rows(archives: dict[str, dict], wanted: set[int] | None) -> list[tuple[str, str, int]]:
    rows = []
    for label, d in archives.items():
        for column, layer in enumerate(d["layers"]):
            if d["layered"] and wanted is not None and layer not in wanted:
                continue
            rows.append((f"{label}/L{layer}" if d["layered"] else label, label, column))
    return rows


def cross_session_hits(V: np.ndarray, labels: np.ndarray, sessions: np.ndarray, only_test: str | None = None
                       ) -> tuple[np.ndarray, list[str]]:
    V = S.l2n(np.asarray(V, np.float64))
    hits: list[int] = []
    words: list[str] = []
    for s in np.unique(sessions):
        enrol, seen = [], set()
        for i in np.where(sessions == s)[0]:
            if labels[i] not in seen:
                seen.add(labels[i])
                enrol.append(i)
        enrol = np.array(enrol)
        test = [i for i in np.where(sessions != s)[0] if labels[i] in seen and (only_test is None or sessions[i] == only_test)]
        if not test or len(enrol) < 2:
            continue
        predicted = labels[enrol][(V[test] @ V[enrol].T).argmax(1)]
        hits += (predicted == labels[test]).astype(int).tolist()
        words += labels[test].tolist()
    return np.array(hits, dtype=int), words


def last_session(sessions: np.ndarray) -> str:
    return max((str(s) for s in np.unique(sessions)), key=lambda s: (len(s), s))


def score_layer(X: np.ndarray, d: dict, held: list[str], dim: int, alpha: float, causal: bool = False,
                last_only: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for speaker in held:
        test = d["speakers"] == speaker
        if len(np.unique(d["sessions"][test])) < 2:
            continue
        train = ~test
        W, mu = S.lda_fit(S.session_center(X[train], d["sessions"][train]), d["labels"][train], dim=dim, alpha=alpha)
        V = (S.session_center(X[test], d["sessions"][test]) - mu) @ W
        only = last_session(d["sessions"][test]) if last_only else None
        hits, words = cross_session_hits(V, d["labels"][test], d["sessions"][test], only)
        if not len(hits):
            continue
        entry = {"hits": hits, "words": words}
        if causal:
            entry["causal"] = S.causal_cross_session_top1(X[test], d["labels"][test], d["sessions"][test],
                                                          project=lambda v, mu=mu, W=W: (v - mu) @ W)
        out[speaker] = entry
    return out


def interval_of_accuracy(hits: np.ndarray, words: list[str]) -> list[float]:
    errors = 1 - hits
    draws = ci.bootstrap(errors, np.ones(len(errors)), clusters=[(w,) for w in words])
    low, high = ci.interval(draws["stock"])
    return [float(1 - high), float(1 - low)]


def gain_interval(hits: np.ndarray, base: np.ndarray, words: list[str]) -> dict:
    draws = ci.bootstrap(1 - base, np.ones(len(base)), 1 - hits, clusters=[(w,) for w in words])
    low, high = ci.interval(draws["delta"])
    return {"gain": float(hits.mean() - base.mean()), "ci": [float(low), float(high)]}


def pooled(per_speaker: dict[str, dict], speakers: set[str]) -> tuple[np.ndarray, list[str]]:
    chosen = [s for s in sorted(per_speaker) if s in speakers]
    if not chosen:
        return np.array([], dtype=int), []
    return (np.concatenate([per_speaker[s]["hits"] for s in chosen]), [w for s in chosen for w in per_speaker[s]["words"]])


def summarise(systems: dict[str, dict[str, dict]], severe: set[str], baseline: str | None) -> list[dict]:
    rows = []
    everyone = {s for per in systems.values() for s in per}
    base_hits = pooled(systems[baseline], severe)[0] if baseline else None
    for name, per in systems.items():
        row: dict = {"system": name, "speakers": {s: float(per[s]["hits"].mean()) for s in sorted(per)},
                     "items": {s: int(len(per[s]["hits"])) for s in sorted(per)}}
        for group, members in (("severe", severe), ("all", everyone)):
            hits, words = pooled(per, members)
            if len(hits):
                row[group] = {"accuracy": float(hits.mean()), "ci": interval_of_accuracy(hits, words), "items": int(len(hits))}
        if baseline and name != baseline and "severe" in row and base_hits is not None:
            hits, words = pooled(per, severe)
            if len(hits) == len(base_hits):
                row["severe"]["against_baseline"] = {"baseline": baseline, **gain_interval(hits, base_hits, words)}
        causal = [per[s]["causal"] for s in per if "causal" in per[s] and s in severe and per[s]["causal"][1] > 0]
        if causal:
            row["severe_causal"] = sum(a * n for a, n in causal) / sum(n for _, n in causal)
            row["severe_causal_items"] = sum(n for _, n in causal)
        rows.append(row)
    return rows


def render(rows: list[dict], baseline: str | None) -> str:
    width = max(len(r["system"]) for r in rows)
    out = [f"{'system':<{width}}  severe top-1 [95% CI]      all top-1 [95% CI]       gain over {baseline or 'baseline'} (severe)"]
    for r in rows:
        def cell(group):
            g = r.get(group)
            return f"{100 * g['accuracy']:5.1f} [{100 * g['ci'][0]:4.1f}, {100 * g['ci'][1]:4.1f}]" if g else "        n/a        "
        gain = r.get("severe", {}).get("against_baseline")
        gain_text = f"{100 * gain['gain']:+5.1f} [{100 * gain['ci'][0]:+5.1f}, {100 * gain['ci'][1]:+5.1f}]" if gain else ""
        extra = f"   causal {100 * r['severe_causal']:.1f}" if "severe_causal" in r else ""
        out.append(f"{r['system']:<{width}}  {cell('severe')}   {cell('all')}   {gain_text}{extra}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", action="append", required=True, metavar="LABEL=FILE.npz", help="an archive; repeat for each model")
    ap.add_argument("--baseline", help="a row to compare with, such as wavlm (flat archive) or v2/L10")
    ap.add_argument("--layers", default="all", help="layers of the layered archives to score: 'all' or e.g. 0,4,8-12")
    ap.add_argument("--severe", default="", help="speakers counted as severe (default: TORGO's four)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--last-session", action="store_true",
                    help="test each speaker on his last session only, enrolling on the earlier ones (the F-2 split): "
                         "the fair setting for fine-tuned models, which trained on the earlier sessions' audio but never the last")
    ap.add_argument("--causal", action="store_true", help="also report accuracy with a session mean that uses only earlier clips")
    ap.add_argument("--out", help="also write the table as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.out:
        lab.refuse_existing(a.out)
    archives: dict[str, dict] = {}
    for text in a.emb:
        label, _, path = text.partition("=")
        if not path or label in archives:
            ap.error(f"--emb wants a new LABEL=FILE.npz, got {text!r}")
        archives[label] = load_archive(path)
    archives = align(archives)
    from willspeak.torgo import SEVERE

    severe = {s.strip() for s in a.severe.split(",") if s.strip()} or set(SEVERE)
    first = next(iter(archives.values()))
    held = sorted({s for s in first["speakers"] if len(np.unique(first["sessions"][first["speakers"] == s])) >= 2})
    print(f"{len(first['paths'])} shared clips · {len(held)} speakers with two or more sessions: {' '.join(held)}")
    plan = plan_rows(archives, layer_spec(a.layers))
    if not plan:
        ap.error(f"--layers {a.layers!r} selects no layer")
    if a.baseline and a.baseline not in [name for name, _, _ in plan]:
        ap.error(f"--baseline {a.baseline!r} is not one of the rows to score: {[n for n, _, _ in plan][:6]} ...")
    systems: dict[str, dict[str, dict]] = {}
    non_finite: dict[str, int] = {}
    for name, label, column in plan:
        d = archives[label]
        X = d["raw"][:, column, :]
        bad = int((~np.isfinite(X)).sum())
        if bad:
            non_finite[name] = bad
            print(f"  SKIPPED {name}: {bad} of {X.size} values are NaN or infinite", flush=True)
            continue
        systems[name] = score_layer(X, d, held, a.dim, a.alpha, a.causal, a.last_session)
        print(f"  scored {name}", flush=True)
    if non_finite:
        print(f"{len(non_finite)} layer rows skipped for non-finite values: {', '.join(non_finite)}")
    if not systems:
        raise ValueError("No finite layer rows remain; regenerate the embedding archives")
    if a.baseline and a.baseline not in systems:
        raise ValueError(f"Baseline {a.baseline!r} has non-finite values; regenerate its embedding archive")
    rows = summarise(systems, severe, a.baseline)
    print(render(rows, a.baseline))
    if a.out:
        sources = {label: path.partition("=")[2] for label, path in zip(archives, a.emb)}
        lab.save_json(a.out, {"rows": rows, "baseline": a.baseline, "severe": sorted(severe), "dim": a.dim, "alpha": a.alpha,
                              "session_mean": "whole held-out session (includes test clips)", "last_session_only": a.last_session, "non_finite_skipped": non_finite, "provenance": lab.provenance(__file__, sources)})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "layer_screen"))
