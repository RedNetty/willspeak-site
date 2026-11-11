#!/usr/bin/env python3
"""Choose among the systems' answers for each clip."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import hyp_bounds  # noqa: E402
import labkit as lab  # noqa: E402
from willspeak.evaluate import DEL, INS, OK, align  # noqa: E402
from willspeak.normalize import normalize  # noqa: E402

EXPORT = "data/public/torgo/export"
EXTRA = ("vote", "wvote")


def words(text: str) -> list[str]:
    return normalize(text).split()


def distance(a: list[str], b: list[str]) -> int:
    return sum(op != OK for op, _, _ in align(a, b))


def similarity(a: list[str], b: list[str]) -> float:
    n = max(len(a), len(b))
    return 1.0 if n == 0 else 1.0 - distance(a, b) / n


def weighted_vote(hyps: list[str], weights: list[float]) -> str:
    if len(hyps) != len(weights) or not hyps:
        raise ValueError("one weight per hypothesis, at least one hypothesis")
    order = sorted(range(len(hyps)), key=lambda i: (-weights[i], i))
    ws = [words(hyps[i]) for i in order]
    wt = [weights[i] for i in order]
    pivot = ws[0]
    slots: list[list[tuple[str | None, float]]] = [[(w, wt[0])] for w in pivot]
    gaps: list[list[tuple[tuple[str, ...], float]]] = [[((), wt[0])] for _ in range(len(pivot) + 1)]
    for other, weight in zip(ws[1:], wt[1:]):
        position, inserted = 0, []
        gap_for = {g: () for g in range(len(pivot) + 1)}
        for op, _, hyp_word in align(pivot, other):
            if op == INS:
                inserted.append(hyp_word)
                continue
            if inserted:
                gap_for[position] = tuple(inserted)
                inserted = []
            slots[position].append((None if op == DEL else hyp_word, weight))
            position += 1
        if inserted:
            gap_for[position] = tuple(inserted)
        for g in range(len(pivot) + 1):
            gaps[g].append((gap_for[g], weight))

    def winner(votes):
        tally: dict = {}
        for choice, weight in votes:
            tally[choice] = tally.get(choice, 0.0) + weight
        return max(tally, key=lambda c: tally[c])

    out: list[str] = []
    for g in range(len(pivot) + 1):
        out.extend(winner(gaps[g]))
        if g < len(pivot):
            chosen = winner(slots[g])
            if chosen is not None:
                out.append(chosen)
    return " ".join(out)


@dataclass
class Clip:
    speaker: str
    id: str
    duration: float
    kind: str
    systems: dict[str, str]
    ref: str | None = None
    prompt: str = ""
    cands: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.speaker, self.id)


def add_candidates(clip: Clip, order: list[str], weights: dict[str, float]) -> None:
    answers = [clip.systems[n] for n in order]
    clip.cands = dict(clip.systems)
    clip.cands["vote"] = hyp_bounds.vote(answers)
    clip.cands["wvote"] = weighted_vote(answers, [weights[n] for n in order])


def system_weights(labelled: list[Clip], names: list[str]) -> tuple[list[str], dict[str, float]]:
    wrong = {n: 0 for n in names}
    total = 0
    for c in labelled:
        ref = words(c.ref or "")
        total += len(ref)
        for n in names:
            wrong[n] += distance(ref, words(c.systems[n]))
    order = sorted(names, key=lambda n: (wrong[n], n))
    return order, {n: max(0.05, 1.0 - wrong[n] / max(total, 1)) for n in names}


class PhraseList:
    def __init__(self, phrases: list[str]):
        import phrasebook

        self.pb = phrasebook
        self.items = [phrasebook.Item(p) for p in dict.fromkeys(normalize(p) for p in phrases) if p]
        self.cache: dict[str, float] = {}

    def nearest(self, text: str) -> float:
        key = normalize(text)
        if not key or not self.items:
            return 1.0
        if key not in self.cache:
            item = self.pb.Item(key)
            self.cache[key] = min(self.pb.distance(item, p) for p in self.items)
        return self.cache[key]


def feature_names(cand_names: list[str], phrase: bool) -> list[str]:
    return [f"is:{n}" for n in cand_names] + ["agree_mean", "same_count", "n_words", "empty", "rate_dev", "short_for_clip"] + (["phrase_dist"] if phrase else [])


def features(clip: Clip, name: str, cand_names: list[str], typical_rate: float, phrases: PhraseList | None) -> list[float]:
    answer = words(clip.cands[name])
    others = [words(h) for n, h in clip.systems.items() if n != name]
    agree = float(np.mean([similarity(answer, o) for o in others])) if others else 1.0
    same = float(sum(1 for o in others if o == answer))
    n = len(answer)
    rate = n / max(clip.duration, 0.2)
    rate_dev = abs(math.log((rate + 0.05) / (typical_rate + 0.05)))
    expected = typical_rate * clip.duration
    short = max(0.0, expected - n) / max(expected, 1.0)
    row = [1.0 if name == c else 0.0 for c in cand_names] + [agree, same, float(n), 1.0 if n == 0 else 0.0, rate_dev, short]
    if phrases is not None:
        row.append(phrases.nearest(clip.cands[name]))
    return row


def typical_rate_of(labelled: list[Clip]) -> float:
    rates = [len(words(c.ref or "")) / max(c.duration, 0.2) for c in labelled if words(c.ref or "")]
    return float(np.median(rates)) if rates else 1.0


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    z = (x - mu) / sd
    ym = float(y.mean())
    a = z.T @ z + lam * np.eye(z.shape[1])
    w = np.linalg.solve(a, z.T @ (y - ym))
    return w, mu, np.concatenate([sd, [ym]])


def ridge_predict(model, x: np.ndarray) -> np.ndarray:
    w, mu, sd_ym = model
    sd, ym = sd_ym[:-1], sd_ym[-1]
    return ((x - mu) / sd) @ w + ym


def table(clips: list[Clip], cand_names: list[str], typical_rate: float, phrases: PhraseList | None, with_target: bool):
    xs, ys, owner = [], [], []
    for i, c in enumerate(clips):
        ref = words(c.ref or "") if with_target else None
        for n in cand_names:
            xs.append(features(c, n, cand_names, typical_rate, phrases))
            ys.append(distance(ref, words(c.cands[n])) / max(len(ref), 1) if ref is not None else 0.0)
            owner.append((i, n))
    return np.array(xs, dtype=float), np.array(ys, dtype=float), owner


def pick_from(prediction: np.ndarray, owner: list[tuple[int, str]], priority: list[str]) -> dict[int, str]:
    best: dict[int, tuple[tuple[float, int], str]] = {}
    for value, (i, n) in zip(prediction, owner):
        key = (round(float(value), 9), priority.index(n) if n in priority else len(priority))
        if i not in best or key < best[i][0]:
            best[i] = (key, n)
    return {i: v[1] for i, v in best.items()}


def wrong_of(clips: list[Clip], chosen: dict[int, str] | None = None, name: str | None = None) -> tuple[int, int]:
    wrong = total = 0
    for i, c in enumerate(clips):
        ref = words(c.ref or "")
        total += len(ref)
        wrong += distance(ref, words(c.cands[chosen[i] if chosen is not None else name]))
    return wrong, total


def medoid_pick(clips: list[Clip], priority: list[str]) -> dict[int, str]:
    out = {}
    for i, c in enumerate(clips):
        scores = {n: np.mean([similarity(words(c.systems[n]), words(c.systems[m])) for m in c.systems if m != n]) for n in c.systems}
        out[i] = max(scores, key=lambda n: (round(float(scores[n]), 9), -priority.index(n)))
    return out


def grouped_folds(clips: list[Clip], k: int, seed: int) -> list[list[int]]:
    from cv_folds import assign_folds

    rows = [{"id": f"{c.speaker}|{c.id}", "text": f"{c.speaker} {c.prompt}"} for c in clips]
    fold_of = assign_folds(rows, k, seed)
    out: list[list[int]] = [[] for _ in range(k)]
    for i, r in enumerate(rows):
        out[fold_of[r["id"]]].append(i)
    return out


def cross_validated(labelled: list[Clip], cand_names: list[str], priority: list[str], ridge: float, folds: int, seed: int,
                    phrases: PhraseList | None) -> dict[int, str]:
    chosen: dict[int, str] = {}
    for held in grouped_folds(labelled, folds, seed):
        rest = [i for i in range(len(labelled)) if i not in set(held)]
        train, test = [labelled[i] for i in rest], [labelled[i] for i in held]
        rate = typical_rate_of(train)
        x, y, _ = table(train, cand_names, rate, phrases, True)
        model = ridge_fit(x, y, ridge)
        tx, _, owner = table(test, cand_names, rate, phrases, False)
        for local, name in pick_from(ridge_predict(model, tx), owner, priority).items():
            chosen[held[local]] = name
    return chosen


def load_clips(root: Path, speaker: str, manifest: Path, hyps: dict[str, dict[str, str]], with_text: bool) -> list[Clip]:
    rows = lab.read_manifest(manifest)
    clips = []
    for cid, row in rows.items():
        systems = {}
        for name, per in hyps.items():
            if cid not in per:
                raise ValueError(f"system {name!r} has no answer for {speaker} clip {cid} ({manifest})")
            systems[name] = per[cid]
        text = str(row.get("text", ""))
        clips.append(Clip(speaker, str(cid), float(row.get("duration_s") or 0.0), str(row.get("kind", "unknown")), systems,
                          ref=text if with_text else None, prompt=normalize(str(row.get("prompt_text") or text), us_spelling=False)))
    return clips


def parse_system(text: str) -> tuple[str, str, str]:
    name, sep, rest = text.partition("=")
    labelled, bar, test = rest.partition("|")
    if not sep or not bar or not name or not labelled or not test:
        raise argparse.ArgumentTypeError(f"--system wants NAME=LABELLED|TEST, got {text!r}")
    return name, labelled, test


def read_systems_file(path: str | Path) -> list[tuple[str, dict, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for name, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("labelled"), dict) or not isinstance(entry.get("test"), dict):
            raise ValueError(f"{path}: system {name!r} needs a 'labelled' and a 'test' object keyed by speaker")
        out.append((name, entry["labelled"], entry["test"]))
    return out


def run(root: Path, speakers: list[str], systems: list[tuple[str, str, str]], out: Path, ridge: float = 10.0, folds: int = 5, seed: int = 17,
        phrase_manifests: dict[str, list[str]] | None = None, labelled_name: str = "cv-{speaker}-all", test_ref: bool = True) -> dict:
    if len(systems) < 2:
        raise ValueError("a selector needs at least two systems")
    names = [n for n, _, _ in systems]
    if len(set(names)) != len(names) or set(names) & set(EXTRA):
        raise ValueError(f"system names must be unique and not {EXTRA}: {names}")
    out = Path(out)
    if (out / "results.json").exists():
        raise FileExistsError(f"{out / 'results.json'} exists")
    labelled_by: dict[str, list[Clip]] = {}
    test_by: dict[str, list[Clip]] = {}
    for s in speakers:
        def get(pattern) -> dict[str, str]:
            path = Path(pattern[s] if isinstance(pattern, dict) else pattern.format(speaker=s))
            return lab.read_hyps(path if path.is_absolute() else root / path)
        lab_hyps = {n: get(l) for n, l, _ in systems}
        test_hyps = {n: get(t) for n, _, t in systems}
        labelled_by[s] = load_clips(root, s, root / EXPORT / labelled_name.format(speaker=s) / "test.jsonl", lab_hyps, True)
        test_by[s] = load_clips(root, s, root / EXPORT / s / "test.jsonl", test_hyps, test_ref)
    labelled = [c for s in speakers for c in labelled_by[s]]
    test = [c for s in speakers for c in test_by[s]]
    order, weights = system_weights(labelled, names)
    for c in labelled + test:
        add_candidates(c, order, weights)
    cand_names = names + list(EXTRA)
    phrases = None
    if phrase_manifests:
        pool: list[str] = []
        for s in speakers:
            for m in phrase_manifests.get(s, []):
                pool += [str(r.get("text", "")) for r in lab.read_manifest(root / m).values() if "(live)" not in str(r.get("text", ""))]
        shared = {normalize(p, us_spelling=False) for p in pool} & {c.prompt for c in test}
        if shared:
            raise ValueError(f"the phrase list contains {len(shared)} test prompts, e.g. {sorted(shared)[:2]}; use other speakers' prompts")
        phrases = PhraseList(pool)
    priority = order + list(EXTRA)
    methods_lab: dict[str, dict[int, str]] = {}
    methods_lab["medoid"] = medoid_pick(labelled, priority)
    methods_lab["learned"] = cross_validated(labelled, cand_names, priority, ridge, folds, seed, phrases)
    sensitivity = {str(lam): wrong_of(labelled, cross_validated(labelled, cand_names, priority, lam, folds, seed, phrases))[0] for lam in (1.0, 10.0, 100.0)}
    rate = typical_rate_of(labelled)
    x, y, _ = table(labelled, cand_names, rate, phrases, True)
    model = ridge_fit(x, y, ridge)
    tx, _, owner = table(test, cand_names, rate, phrases, False)
    methods_test: dict[str, dict[int, str]] = {"medoid": medoid_pick(test, priority), "learned": pick_from(ridge_predict(model, tx), owner, priority)}
    result: dict = {"format": "willspeak-hypothesis-selector-v1", "systems": names, "order_best_first": order, "weights": weights, "ridge": ridge,
                    "folds": folds, "speakers": speakers, "ridge_sensitivity_labelled_wrong": sensitivity,
                    "phrase_list": bool(phrases), "features": feature_names(cand_names, phrases is not None),
                    "coefficients": {n: float(v) for n, v in zip(feature_names(cand_names, phrases is not None), model[0])},
                    "labelled": {}, "test": {}}
    for label, clips, picks in (("labelled", labelled, methods_lab), ("test", test, methods_test)):
        if label == "test" and not test_ref:
            continue
        rows = {}
        for n in cand_names:
            w, t = wrong_of(clips, name=n)
            rows[n] = {"wrong": w, "words": t}
        for m, chosen in picks.items():
            w, t = wrong_of(clips, chosen)
            rows[m] = {"wrong": w, "words": t, "picked": {n: sum(1 for v in chosen.values() if v == n) for n in cand_names}}
        best_each = 0
        for c in clips:
            ref = words(c.ref or "")
            best_each += min(distance(ref, words(h)) for h in c.cands.values())
        rows["oracle"] = {"wrong": best_each, "words": rows[names[0]]["words"]}
        result[label] = rows
    hyp_dir = out / "hyps"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    for m, chosen in methods_test.items():
        for s in speakers:
            lines = []
            for i, c in enumerate(test):
                if c.speaker == s:
                    lines.append(json.dumps({"id": c.id, "hyp": c.cands[chosen[i]]}))
            (hyp_dir / f"{m}-{s}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for n in EXTRA:
        for s in speakers:
            (hyp_dir / f"{n}-{s}.jsonl").write_text("".join(json.dumps({"id": c.id, "hyp": c.cands[n]}) + "\n" for c in test if c.speaker == s), encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def show(result: dict) -> None:
    for label in ("labelled", "test"):
        if label not in result or not result[label]:
            continue
        rows = result[label]
        print(f"\n{'labelled clips, leave-prompt-out (never the test clips)' if label == 'labelled' else 'test clips'}")
        print(f"  {'system or method':<22}{'wrong':>7}{'of':>6}{'WER':>8}")
        for n, r in rows.items():
            print(f"  {n:<22}{r['wrong']:>7}{r['words']:>6}{100 * r['wrong'] / max(r['words'], 1):>7.1f}%")
    print("\n  'oracle' always takes the best candidate and cannot be run on a device.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", nargs="+", required=True)
    ap.add_argument("--system", type=parse_system, action="append", default=[], metavar="NAME=LABELLED|TEST")
    ap.add_argument("--systems-file", help="json with per-speaker file names, for systems whose files are not named alike")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--phrases-from", nargs="*", default=[], metavar="MANIFEST", help="manifests (with {speaker}) whose text is the typed phrase list")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    phrase_manifests = {s: [m.format(speaker=s) for m in a.phrases_from] for s in a.speaker} if a.phrases_from else None
    systems = list(a.system) + (read_systems_file(a.systems_file) if a.systems_file else [])
    if not systems:
        ap.error("give --system NAME=LABELLED|TEST (more than once) or --systems-file")
    result = run(Path(a.root), a.speaker, systems, Path(a.out) if Path(a.out).is_absolute() else Path(a.root) / a.out, a.ridge, a.folds, a.seed, phrase_manifests)
    show(result)
    print(f"\nwrote {a.out}/results.json and {a.out}/hyps/")
    return 0


if __name__ == "__main__":
    try:
        import _cli
    except ImportError:
        _cli = None
    sys.exit(_cli.run(main, "hyp_selector") if _cli else main())
