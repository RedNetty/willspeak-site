#!/usr/bin/env python3
"""Fit the matcher's projection."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli
from willspeak.matcher import training as S
from willspeak.matcher.engine import DIM, POOL


def load_emb(paths: list[str]) -> dict:
    return S.load_embeddings(paths)


def fit(d: dict, mask: np.ndarray, dim: int, alpha: float, kind: str, k: int):
    Xc = S.session_center(d["raw"][mask], d["sessions"][mask])
    W, mu = S.lda_fit(Xc, d["labels"][mask], dim=dim, alpha=alpha)
    P = (Xc - mu) @ W
    cls = np.array([f"{a}|{b}" for a, b in zip(d["speakers"][mask], d["labels"][mask])])
    info = {}
    if kind == "wccn":
        T = S.wccn_fit(P, cls, alpha=alpha)
    elif kind == "nap":
        T, ev = S.nap_fit(P, cls, d["sessions"][mask], k=k)
        info["nap_eigenvalues_top16"] = [round(float(x), 3) for x in ev[:16]]
    else:
        T = None
    return W, mu, T, info


def evaluate(d: dict, dim: int, alpha: float, naps: list[int]) -> None:
    speakers = np.unique(d["speakers"])
    variants = ["raw", "lda", "lda+wccn"] + [f"lda+nap{k}" for k in naps] + ["lda+sessmean(control)"]
    table = {v: [] for v in variants}
    probe = {v: [] for v in variants}
    weights = []
    for held in speakers:
        m_te = d["speakers"] == held
        if len(np.unique(d["sessions"][m_te])) < 2:
            continue
        m_tr = ~m_te
        X, lab, sess = d["raw"][m_te], d["labels"][m_te], d["sessions"][m_te]
        W, mu, _, _ = fit(d, m_tr, dim, alpha, "none", 0)
        P = (X - mu) @ W
        Pc = (S.session_center(X, sess) - mu) @ W
        spaces = {"raw": X, "lda": P, "lda+sessmean(control)": Pc}
        cls = np.array([f"{a}|{b}" for a, b in zip(d["speakers"][m_tr], d["labels"][m_tr])])
        Ptr = (S.session_center(d["raw"][m_tr], d["sessions"][m_tr]) - mu) @ W
        spaces["lda+wccn"] = P @ S.wccn_fit(Ptr, cls, alpha=alpha)
        for k in naps:
            T, _ = S.nap_fit(Ptr, cls, d["sessions"][m_tr], k=k)
            spaces[f"lda+nap{k}"] = P @ T
        n = None
        row = []
        for v in variants:
            acc, n = S.cross_session_top1(spaces[v], lab, sess)
            pa, ch = S.session_probe_accuracy(spaces[v], sess, labels=lab)
            table[v].append(acc); probe[v].append(pa); row.append(f"{acc:.3f}")
        weights.append(n)
        print(f"  {held:8s} n={n:5d}  " + "  ".join(f"{v}={r}" for v, r in zip(variants, row)) + f"  (probe chance {ch:.2f})")
    if not weights:
        print("no speaker has two sessions -- nothing to evaluate cross-session"); return
    w = np.array(weights, float)
    print("\ncross-session top-1, macro over held-out speakers (weighted by n in brackets):")
    for v in variants:
        a = np.array(table[v]); p = np.array(probe[v])
        print(f"  {v:24s} {a.mean():.3f} [{(a * w).sum() / w.sum():.3f}]   session probe {p.mean():.2f}")
    print("\nCross-session top-1 is the decision metric; the control row is per-session mean subtraction only.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", nargs="+", required=True, help="embeddings .npz files from embed_manifest.py")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1, help="shrinkage toward identity for both scatters")
    ap.add_argument("--session-transform", choices=["none", "wccn", "nap"], default="none",
                    help="session transform to fit after LDA (none ships)")
    ap.add_argument("--nap-k", type=int, default=8)
    ap.add_argument("--eval", action="store_true", help="leave-one-speaker-out comparison; writes nothing")
    ap.add_argument("--nap-ks", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="k values for --eval")
    ap.add_argument("--out", help="projection .npz to write")
    a = ap.parse_args()

    d = load_emb(a.emb)
    if d["raw"].shape[1] != DIM:
        print(f"embeddings are {d['raw'].shape[1]}-d but the matcher embeds {DIM}-d ({POOL}); "
              f"re-run scripts/matcher/embed_manifest.py"); return 1
    print(f"{len(d['raw'])} vectors · {len(np.unique(d['labels']))} labels · "
          f"{len(np.unique(d['speakers']))} speakers · {len(np.unique(d['sessions']))} sessions")
    if a.eval:
        evaluate(d, a.dim, a.alpha, a.nap_ks)
        return 0
    if not a.out:
        print("give --out (or --eval)"); return 1
    t0 = time.time()
    W, mu, T, info = fit(d, np.ones(len(d["raw"]), bool), a.dim, a.alpha, a.session_transform, a.nap_k)
    meta = {"train": "+".join(Path(p).stem for p in a.emb), "dim": a.dim, "alpha": a.alpha, "pool": POOL, "in_dim": DIM,
            "session_transform": a.session_transform, "nap_k": a.nap_k if a.session_transform == "nap" else None,
            "n": len(d["raw"]), "labels": len(np.unique(d["labels"])),
            "speakers": len(np.unique(d["speakers"])), "sessions": len(np.unique(d["sessions"])),
            "fitted": time.strftime("%Y-%m-%d"), **info}
    arrays = {"W": W, "mean": mu, "meta": json.dumps(meta)}
    if T is not None:
        arrays["T"] = T
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"wrote {out}  W {W.shape}  T {'none' if T is None else T.shape}  in {time.time() - t0:.1f}s")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "train_projection"))
