#!/usr/bin/env python3
"""Does a model's stock word error rate predict its rate after fine-tuning? Lab test 7."""
from __future__ import annotations

import argparse
import math
import sys

import labkit as lab


def concordance(stock: list[float], tuned: list[float]) -> dict[str, int]:
    if len(stock) != len(tuned):
        raise ValueError("stock and tuned need one value per candidate")
    out = {"concordant": 0, "discordant": 0, "ties": 0}
    for i in range(len(stock)):
        for j in range(i + 1, len(stock)):
            a, b = stock[i] - stock[j], tuned[i] - tuned[j]
            if a == 0 or b == 0:
                out["ties"] += 1
            elif (a > 0) == (b > 0):
                out["concordant"] += 1
            else:
                out["discordant"] += 1
    return out


def scope_report(stock: list[float], tuned: list[float]) -> dict:
    n = len(stock)
    rho = lab.spearman(stock, tuned) if n >= 2 else float("nan")
    return {"stock_wer": stock, "tuned_wer": tuned, "spearman": None if rho != rho else rho,
            "pairs": concordance(stock, tuned), "chance_of_this_exact_order": 1 / math.factorial(n)}


def build(stock: dict[str, lab.Pooled], tuned: dict[str, lab.Pooled]) -> dict:
    if list(stock) != list(tuned):
        raise ValueError(f"--stock and --tuned must list the same candidates in the same order: {list(stock)} vs {list(tuned)}")
    labels = list(stock)
    if len(labels) < 3:
        raise ValueError("A rank check needs at least 3 candidates")
    speakers = stock[labels[0]].speakers
    for label in labels:
        for system in (stock[label], tuned[label]):
            if system.speakers != speakers:
                raise ValueError(f"{label}: every system must cover the same speakers {speakers}")
    scopes = {"pooled": scope_report([stock[c].pooled_wrong() / stock[c].pooled_words() for c in labels],
                                     [tuned[c].pooled_wrong() / tuned[c].pooled_words() for c in labels])}
    for s in speakers:
        scopes[s] = scope_report([stock[c].wrong(s) / stock[c].words(s) for c in labels],
                                 [tuned[c].wrong(s) / tuned[c].words(s) for c in labels])
    gains = {c: 100.0 * (stock[c].pooled_wrong() - tuned[c].pooled_wrong()) / stock[c].pooled_words() for c in labels}
    return {"candidates": labels, "speakers": speakers, "scopes": scopes, "gain_points_pooled": gains,
            "tuned_pooled_wrong": {c: tuned[c].pooled_wrong() for c in labels},
            "tuned_pooled_words": tuned[labels[0]].pooled_words()}


def render(result: dict) -> str:
    labels = result["candidates"]
    pooled = result["scopes"]["pooled"]
    width = max(len(c) for c in labels)
    out = [f"{'candidate':<{width}}  stock WER  fine-tuned WER  gain (points)  fine-tuned wrong"]
    for i, c in enumerate(labels):
        out.append(f"{c:<{width}}  {100 * pooled['stock_wer'][i]:8.1f}%  {100 * pooled['tuned_wer'][i]:12.1f}%  "
                   f"{result['gain_points_pooled'][c]:+12.1f}  {result['tuned_pooled_wrong'][c]:>3}/{result['tuned_pooled_words']}")
    out.append("")
    for name, scope in result["scopes"].items():
        p = scope["pairs"]
        rho = "undefined" if scope["spearman"] is None else f"{scope['spearman']:+.2f}"
        total = p["concordant"] + p["discordant"] + p["ties"]
        out.append(f"{name:<7} Spearman {rho}; stock order right for {p['concordant']} of {total} pairs "
                   f"({p['discordant']} reversed, {p['ties']} tied); a random order matches exactly {100 * scope['chance_of_this_exact_order']:.1f}% of the time")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", action="append", required=True, metavar="LABEL=SPK:PATH,...", help="stock decodes, one per candidate")
    ap.add_argument("--tuned", action="append", required=True, metavar="LABEL=SPK:PATH,...", help="fine-tuned decodes, same labels")
    ap.add_argument("--speakers", default="M02,M04", help="speakers pooled (default M02,M04: the ones with fine-tuned candidates)")
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--out", help="also write the result as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.out:
        lab.refuse_existing(a.out)
    speakers = [s.strip() for s in a.speakers.split(",") if s.strip()]
    manifests = {s: lab.read_manifest(lab.here(f"{a.export}/{s}/test.jsonl")) for s in speakers}
    systems: dict[str, dict[str, lab.Pooled]] = {"stock": {}, "tuned": {}}
    sources: dict[str, str] = {}
    for kind, entries in (("stock", a.stock), ("tuned", a.tuned)):
        for text in entries:
            label, folders = lab.parse_system(text)
            if label in systems[kind]:
                raise ValueError(f"--{kind} lists {label!r} twice")
            systems[kind][label] = lab.load_pooled(label, folders, manifests)
            for s, folder in folders.items():
                sources[f"{kind}/{label}/{s}"] = folder
    result = build(systems["stock"], systems["tuned"])
    print(render(result))
    if a.out:
        lab.save_json(a.out, {**result, "provenance": lab.provenance(__file__, sources)})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "rank_check"))
