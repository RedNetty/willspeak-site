#!/usr/bin/env python3
"""Score the device the way it will be used."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cli
import labkit as lab
import layer_screen
from willspeak.matcher import training as S
from willspeak.matcher.scoring import label_prototypes
from willspeak.normalize import normalize


def path_id(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 4 or not parts[-2].startswith("wav_"):
        raise ValueError(f"Not a TORGO audio path (.../<SPK>/<Session>/wav_<mic>/<NNNN>.wav): {path}")
    return f"{parts[-4]}_{parts[-3]}_{Path(path).stem}"


def session_key(name: str) -> tuple[int, str]:
    return (len(name), name)


def matcher_answers(d: dict, column: int, speaker: str, test_session: str, dim: int = 128, alpha: float = 0.1,
                    warmup: int = 20) -> dict[str, tuple[str, float]]:
    X = d["raw"][:, column, :]
    mine = d["speakers"] == speaker
    sessions = sorted({str(s) for s in d["sessions"][mine]}, key=session_key)
    if test_session not in sessions:
        raise ValueError(f"{speaker} has no word takes in {test_session}; sessions with words: {sessions}")
    earlier = [s for s in sessions if session_key(s) < session_key(test_session)]
    if not earlier:
        raise ValueError(f"{speaker} has no word takes before {test_session} to enrol from")
    others = ~mine
    if len(np.unique(d["labels"][others])) < 2:
        raise ValueError("The projection needs other speakers' words; the archive has none")
    W, mu = S.lda_fit(S.session_center(X[others], d["sessions"][others]), d["labels"][others], dim=dim, alpha=alpha)
    centres = {s: X[mine & (d["sessions"] == s)].mean(0) for s in earlier}
    bank_rows = np.where(mine & np.isin(d["sessions"], earlier))[0]
    bank = np.stack([X[i] - centres[str(d["sessions"][i])] for i in bank_rows])
    prior = np.mean([centres[s] for s in earlier], axis=0)
    bank_labels, prototypes = label_prototypes(S.l2n((bank - mu) @ W), d["labels"][bank_rows])
    queries = sorted(np.where(mine & (d["sessions"] == test_session))[0], key=lambda i: path_id(d["paths"][i]))
    centred = S.causal_session_center(X[queries], prior, warmup)
    similarity = S.l2n((centred - mu) @ W) @ prototypes.T
    best = similarity.argmax(1)
    return {path_id(d["paths"][i]): (bank_labels[b], float(similarity[k, b])) for k, (i, b) in enumerate(zip(queries, best))}


def route(rows: list[dict], parakeet: dict[str, str], answers: dict[str, tuple[str, float]], min_score: float | None) -> dict[str, str]:
    out = {}
    for row in rows:
        clip = str(row["id"])
        hyp = parakeet[clip]
        if row.get("kind") == "word" and clip in answers and (min_score is None or answers[clip][1] >= min_score):
            hyp = answers[clip][0]
        out[clip] = hyp
    return out


def word_clip_stats(rows: list[dict], parakeet: dict[str, str], answers: dict[str, tuple[str, float]]) -> dict:
    words = [r for r in rows if r.get("kind") == "word"]

    def right(said: str, row: dict) -> bool:
        return normalize(said) == normalize(row["text"])

    answered = [r for r in words if str(r["id"]) in answers]
    enrolled = {normalize(label) for label, _ in answers.values()}
    in_vocabulary = [r for r in answered if normalize(r["text"]) in enrolled]
    return {
        "test_clips": len(rows), "word_clips": len(words), "sentence_clips": len(rows) - len(words),
        "word_clips_without_embedding": len(words) - len(answered),
        "word_clips_whose_word_was_answerable": len(in_vocabulary),
        "matcher_right": sum(1 for r in answered if right(answers[str(r["id"])][0], r)),
        "parakeet_right_on_word_clips": sum(1 for r in words if right(parakeet[str(r["id"])], r)),
        "matcher_right_and_parakeet_wrong": sum(1 for r in answered if right(answers[str(r["id"])][0], r) and not right(parakeet[str(r["id"])], r)),
        "parakeet_right_and_matcher_wrong": sum(1 for r in answered if not right(answers[str(r["id"])][0], r) and right(parakeet[str(r["id"])], r)),
    }


def write_variant(folder: Path, hyps: dict[str, str]) -> None:
    folder.mkdir(parents=True)
    (folder / "hypotheses.jsonl").write_text("".join(json.dumps({"id": c, "hyp": h}) + "\n" for c, h in hyps.items()), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True, metavar="FILE.npz", help="embeddings of every speaker's single words (embed_manifest.py or embed_layers.py)")
    ap.add_argument("--layer", type=int, help="which layer, when the archive has several (embed_layers.py)")
    ap.add_argument("--speaker", action="append", required=True, metavar="SPK=HYPOTHESES", help="a speaker and Parakeet's answers on his test clips")
    ap.add_argument("--min-scores", type=float, nargs="*", default=[], help="also write the confidence-fallback variants at these cosine scores")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--out", required=True, help="new folder (refuses to overwrite)")
    a = ap.parse_args()
    lab.refuse_existing(a.out)
    d = layer_screen.load_archive(a.emb)
    if d["layered"] and a.layer is None:
        ap.error(f"{a.emb} has layers {d['layers'][0]} to {d['layers'][-1]}; choose one with --layer")
    if d["layered"] and a.layer not in d["layers"]:
        ap.error(f"--layer {a.layer} is not in the archive ({d['layers'][0]} to {d['layers'][-1]})")
    column = d["layers"].index(a.layer) if d["layered"] else 0
    variants = ["parakeet", "routed", *[f"routed-min{t:g}" for t in a.min_scores]]
    per_variant: dict[str, dict[str, dict[str, str]]] = {v: {} for v in variants}
    report: dict[str, dict] = {}
    sources: dict[str, str] = {"embeddings": a.emb}
    out = lab.here(a.out)
    for text in a.speaker:
        speaker, _, folder = text.partition("=")
        if not folder:
            ap.error(f"--speaker wants SPK=HYPOTHESES, got {text!r}")
        manifest = lab.read_manifest(lab.test_manifest(speaker, a.export))
        rows = list(manifest.values())
        sessions = {r.get("session") for r in rows}
        if len(sessions) != 1 or None in sessions:
            raise ValueError(f"{speaker}: the test clips must all come from one named session, found {sessions}")
        test_session = f"{speaker}_{sessions.pop()}"
        parakeet = lab.read_hyps(folder)
        missing = [str(r["id"]) for r in rows if str(r["id"]) not in parakeet]
        if missing:
            raise ValueError(f"{speaker}: no Parakeet answer for {len(missing)} test clips, first {missing[:3]}")
        answers = matcher_answers(d, column, speaker, test_session, a.dim, a.alpha, a.warmup)
        per_variant["parakeet"][speaker] = {str(r["id"]): parakeet[str(r["id"])] for r in rows}
        per_variant["routed"][speaker] = route(rows, parakeet, answers, None)
        for threshold in a.min_scores:
            per_variant[f"routed-min{threshold:g}"][speaker] = route(rows, parakeet, answers, threshold)
        stats = word_clip_stats(rows, parakeet, answers)
        stats["test_session"] = test_session
        stats["enrolled_from"] = sorted({str(s) for s in d["sessions"][d["speakers"] == speaker]
                                         if session_key(str(s)) < session_key(test_session)}, key=session_key)
        counts = {v: lab.totals(lab.clip_counts(manifest, per_variant[v][speaker])) for v in variants}
        stats["wrong_words"] = {v: counts[v][0] for v in variants}
        stats["reference_words"] = counts["parakeet"][1]
        report[speaker] = stats
        sources[f"hypotheses/{speaker}"] = folder
        print(f"{speaker}: {stats['word_clips']} word clips of {stats['test_clips']}; matcher right on {stats['matcher_right']}, "
              f"Parakeet on {stats['parakeet_right_on_word_clips']}; wrong words " +
              ", ".join(f"{v} {counts[v][0]}" for v in variants) + f" of {counts['parakeet'][1]}")
    for variant in variants:
        for speaker, hyps in per_variant[variant].items():
            write_variant(out / variant / speaker, hyps)
    lab.save_json(out / "route.json", {"format": "willspeak-route-v1", "variants": variants, "min_scores": a.min_scores,
                                       "layer": a.layer, "speakers": report, "dim": a.dim, "alpha": a.alpha, "warmup": a.warmup,
                                       "provenance": lab.provenance(__file__, sources)})
    systems = " ".join(f'--system "{v}=' + ",".join(f"{s}:{(out / v / s)}" for s in report) + '"' for v in variants)
    print(f"wrote {out}\nscore with: python scripts/asr/pool_table.py {systems} --baseline parakeet")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "route_score"))
