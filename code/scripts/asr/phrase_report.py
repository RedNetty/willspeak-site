#!/usr/bin/env python3
"""Spoken WER from phrase_score.py tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.normalize import normalize  # noqa: E402

TEXT_T = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, None]
LAMS = [0.0, 10.0, 20.0, 40.0]
TAUS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, None]
CONDS = ("closed", "closed+", "open")


def word_errors(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def candidate_sets(row: dict, shortlist: int) -> dict[str, list[dict]]:
    pool = [c for c in row["candidates"] if c["in_pool"]]
    by_rank = sorted(pool, key=lambda c: c["rank"])
    closed = [c for c in by_rank if c["rank"] < shortlist]
    plus = closed + [c for c in by_rank if c["true"] and c["rank"] >= shortlist]
    opened = [c for c in by_rank if not c["true"]][:shortlist]
    return {"closed": closed, "closed+": plus, "open": opened}


def own_nll(row: dict):
    for c in row["candidates"]:
        if c["hyp"]:
            return c["nll"]
    return row["nll_empty"] if not row["hyp"] else None


def decide(row: dict, cands: list[dict], cfg: tuple) -> str:
    if cfg[0] == "raw" or not cands:
        return row["hyp"]
    if cfg[0] == "text":
        best = cands[0]
        return best["text"] if cfg[1] is None or best["dist"] <= cfg[1] else row["hyp"]
    _, lam, tau = cfg
    scored = [c for c in cands if c["nll"] is not None]
    if not scored:
        return row["hyp"]
    best = min(scored, key=lambda c: (c["nll"] + lam * c["dist"], c["rank"]))
    mine = own_nll(row)
    if tau is None or mine is None or best["nll"] - mine <= tau:
        return best["text"]
    return row["hyp"]


def configs() -> list[tuple]:
    return [("raw",), *[("text", t) for t in TEXT_T], *[("ac", lam, tau) for lam in LAMS for tau in TAUS]]


def name_of(cfg: tuple) -> str:
    if cfg[0] == "raw":
        return "raw"
    if cfg[0] == "text":
        return "text, no rule" if cfg[1] is None else f"text, t<={cfg[1]:.2f}"
    return f"acoustic lam={cfg[1]:g}" + (", no rule" if cfg[2] is None else f", tau<={cfg[2]:g}")


class Table:
    def __init__(self, label: str, files: dict[str, str], shortlist: int | None):
        self.label = label
        self.speakers = list(files)
        self.spk, self.kind, self.words, self.ids = [], [], [], []
        rows = []
        for s, path in files.items():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.shortlist = shortlist or data["shortlist"]
            for r in data["rows"]:
                rows.append(r)
                self.spk.append(s)
                self.kind.append(r["kind"])
                self.ids.append(f"{s}:{r['id']}")
                self.words.append(len(r["ref"].split()))
        self.spk, self.kind, self.words = np.array(self.spk), np.array(self.kind), np.array(self.words)
        self.cfgs = configs()
        self.err: dict[tuple, np.ndarray] = {}
        sets = [candidate_sets(r, self.shortlist) for r in rows]
        refs = [r["ref"].split() for r in rows]
        for cfg in self.cfgs:
            for cond in CONDS:
                vals = []
                for r, cs, ref in zip(rows, sets, refs):
                    text = decide(r, cs[cond], cfg)
                    vals.append(word_errors(ref, text.split()))
                self.err[(cfg, cond)] = np.array(vals)
        self.rows = rows

    def spoken(self, cfg: tuple, cov: float, idx: np.ndarray) -> float:
        n = self.words[idx].sum()
        return 100.0 * (cov * self.err[(cfg, "closed")][idx].sum() + (1 - cov) * self.err[(cfg, "open")][idx].sum()) / max(n, 1)

    def per_clip(self, picks: dict[str, tuple], cov: float) -> np.ndarray:
        out = np.zeros(len(self.words))
        for s, cfg in picks.items():
            m = self.spk == s
            out[m] = cov * self.err[(cfg, "closed")][m] + (1 - cov) * self.err[(cfg, "open")][m]
        return out


def select(table: Table, method: str, cov: float) -> dict[str, tuple]:
    pool = [c for c in table.cfgs if c[0] in ("raw", method)]
    picks = {}
    for s in table.speakers:
        others = np.where(table.spk != s)[0]
        picks[s] = min(pool, key=lambda c: table.spoken(c, cov, others))
    return picks


def boot(values: dict[str, np.ndarray], words: np.ndarray, n: int, seed: int = 17) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    m = len(words)
    draws = rng.integers(0, m, size=(n, m))
    out = {}
    for key, v in values.items():
        est = 100.0 * v.sum() / words.sum()
        rs = 100.0 * v[draws].sum(axis=1) / words[draws].sum(axis=1)
        out[key] = (est, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    return out


def boot_diff(a: np.ndarray, b: np.ndarray, words: np.ndarray, n: int, seed: int = 17) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = len(words)
    draws = rng.integers(0, m, size=(n, m))
    d = 100.0 * (a - b)
    est = d.sum() / words.sum()
    rs = d[draws].sum(axis=1) / words[draws].sum(axis=1)
    return est, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def coverage(export_root: str, speakers: list[str], kinds: list[str]) -> dict[str, dict]:
    root = Path(export_root)

    def prompts(path: Path) -> list[str]:
        out = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    k = r.get("kind") or ("sentence" if len(str(r.get("text", "")).split()) >= 3 else "word")
                    if k in kinds and "(live)" not in str(r.get("text", "")):
                        out.append(normalize(str(r["text"])))
        return out

    folders = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("loso-"))
    res = {}
    for s in speakers:
        test = prompts(root / s / "test.jsonl")
        own = set(prompts(root / s / "train.jsonl")) | set(prompts(root / s / "dev.jsonl"))
        other = set()
        for f in folders:
            if f != s:
                for sp in ("train", "dev", "test"):
                    other |= set(prompts(root / f / f"{sp}.jsonl"))
        n = max(len(test), 1)
        res[s] = {"test_clips": len(test), "own_training_prompts": 100.0 * sum(t in own for t in test) / n,
                  "other_speakers": 100.0 * sum(t in other for t in test) / n,
                  "either": 100.0 * sum(t in own or t in other for t in test) / n,
                  "neither": 100.0 * sum(t not in own and t not in other for t in test) / n, "other_folders": len(folders) - 1}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="label=SPK:phrase_score.json,SPK:... (repeatable)")
    ap.add_argument("--shortlist", type=int, default=0, help="phrases per clip to allow (default: what the tables were scored with)")
    ap.add_argument("--coverage", default="1.0,0.9,0.8,0.5")
    ap.add_argument("--select-coverage", type=float, default=0.8)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--export-root", default="", help="export folder, to measure how much of the test prompts the other data covers")
    ap.add_argument("--kinds", default="sentence,word")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    covs = [float(x) for x in a.coverage.split(",") if x]
    out = Path(a.out)
    if (out / "results.json").exists():
        raise FileExistsError(f"Already exists; choose a new name: {out / 'results.json'}")
    result: dict = {"select_coverage": a.select_coverage, "systems": {}}
    for spec in a.system:
        label, rest = spec.split("=", 1)
        files = dict(part.split(":", 1) for part in rest.split(","))
        t = Table(label, files, a.shortlist or None)
        allidx = np.arange(len(t.words))
        print(f"\n######## {label}: {len(t.words)} clips, {int(t.words.sum())} words, {len(t.speakers)} speakers, shortlist {t.shortlist}")
        print("\n-- no rule, pooled over speakers: wrong-word % under each list (acoustic uses lam=0; closed+ scores the true phrase even when text ranks it low)")
        print(f"{'':<26} " + " ".join(f"{c:>9}" for c in CONDS) + f" {'cov ' + format(a.select_coverage, 'g'):>9}")
        raw_pct = 100.0 * t.err[(("raw",), "closed")].sum() / t.words.sum()
        print(f"{'raw answer':<26} " + " ".join(f"{raw_pct:>8.1f}%" for _ in CONDS) + f" {raw_pct:>8.1f}%")
        for cfg in (("text", None), ("ac", 0.0, None), ("ac", 20.0, None)):
            cells = [100.0 * t.err[(cfg, c)].sum() / t.words.sum() for c in CONDS]
            mix = t.spoken(cfg, a.select_coverage, allidx)
            print(f"{name_of(cfg):<26} " + " ".join(f"{v:>8.1f}%" for v in cells) + f" {mix:>8.1f}%")
        print("\n-- by speaker, no rule, closed list / open list")
        for s in t.speakers:
            m = np.where(t.spk == s)[0]
            row = [f"{s}: raw {100.0 * t.err[(('raw',), 'closed')][m].sum() / t.words[m].sum():.1f}"]
            for cfg in (("text", None), ("ac", 0.0, None)):
                row.append(f"{name_of(cfg)} {100.0 * t.err[(cfg, 'closed')][m].sum() / t.words[m].sum():.1f}/{100.0 * t.err[(cfg, 'open')][m].sum() / t.words[m].sum():.1f}")
            print("  " + "   ".join(row))
        sys_out: dict = {"clips": len(t.words), "words": int(t.words.sum()), "selection": {}}
        for method in ("text", "ac"):
            picks = select(t, method, a.select_coverage)
            print(f"\n-- {method}: settings chosen on the other speakers at coverage {a.select_coverage:g}, applied to each speaker")
            for s, cfg in picks.items():
                m = np.where(t.spk == s)[0]
                print(f"  {s}: {name_of(cfg):<28} held-out spoken WER {t.spoken(cfg, a.select_coverage, m):.1f}% (raw {t.spoken(('raw',), a.select_coverage, m):.1f}%)")
            print(f"  {'coverage':<10} {'raw':>22} {'chosen on others (95% interval)':>36} {'minus raw (95% interval)':>28}")
            sys_out["selection"][method] = {"picks": {s: name_of(c) for s, c in picks.items()}, "by_coverage": {}}
            for cov in covs:
                chosen = t.per_clip(picks, cov)
                raw = t.per_clip({s: ("raw",) for s in t.speakers}, cov)
                est = boot({"raw": raw, "chosen": chosen}, t.words, a.boot)
                diff = boot_diff(chosen, raw, t.words, a.boot)
                print(f"  {cov:<10.0%} {est['raw'][0]:>8.1f}% [{est['raw'][1]:.1f}, {est['raw'][2]:.1f}]"
                      f" {est['chosen'][0]:>12.1f}% [{est['chosen'][1]:.1f}, {est['chosen'][2]:.1f}]{'':>6}{diff[0]:>+9.1f} [{diff[1]:+.1f}, {diff[2]:+.1f}]")
                sys_out["selection"][method]["by_coverage"][f"{cov:g}"] = {"raw": est["raw"], "chosen": est["chosen"], "difference": diff}
            ac_pick = picks
            for kind in ("sentence", "word"):
                m = np.where(t.kind == kind)[0]
                if len(m):
                    wrong = t.per_clip(ac_pick, a.select_coverage)[m].sum()
                    rawk = t.per_clip({s: ("raw",) for s in t.speakers}, a.select_coverage)[m].sum()
                    print(f"  {kind + ' clips':<14} {int(t.words[m].sum())} words: raw {100.0 * rawk / t.words[m].sum():.1f}%, chosen {100.0 * wrong / t.words[m].sum():.1f}% at coverage {a.select_coverage:g}")
        best = min(((cfg, t.spoken(cfg, a.select_coverage, allidx)) for cfg in t.cfgs), key=lambda x: x[1])
        print(f"\n-- for reference only (settings chosen on these same test clips, optimistic): best is {name_of(best[0])} at {best[1]:.1f}%")
        sys_out["best_on_test_optimistic"] = {"config": name_of(best[0]), "wer": best[1]}
        result["systems"][label] = sys_out
    if a.export_root:
        speakers = sorted({s for spec in a.system for part in spec.split("=", 1)[1].split(",") for s in [part.split(":", 1)[0]]})
        kinds = [k for k in a.kinds.split(",") if k]
        cov = coverage(a.export_root, speakers, kinds)
        print("\n######## how much of each speaker's test prompts the rest of the data covers (share of test clips whose prompt appears there)")
        print(f"{'speaker':<8} {'clips':>6} {'own train+dev':>14} {'other speakers':>15} {'either':>8} {'neither':>8}")
        for s, v in cov.items():
            print(f"{s:<8} {v['test_clips']:>6} {v['own_training_prompts']:>13.1f}% {v['other_speakers']:>14.1f}% {v['either']:>7.1f}% {v['neither']:>7.1f}%")
        print(f"(other speakers: {next(iter(cov.values()))['other_folders']} folders besides the speaker's own; 'neither' is the share a list built from this data would miss)")
        result["coverage"] = cov
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
