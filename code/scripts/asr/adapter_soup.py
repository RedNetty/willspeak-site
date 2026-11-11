"""Average the weights of several adapters trained on the same base (a "soup")."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import block_adapter  # noqa: E402


def average_payloads(payloads: list[dict], weights: list[float] | None = None) -> dict:
    if not payloads:
        raise ValueError("no adapters to average")
    if weights is None:
        weights = [1.0] * len(payloads)
    if len(weights) != len(payloads) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative, one per adapter, and not all zero")
    first = payloads[0]
    for i, p in enumerate(payloads[1:], start=2):
        for key in ("after_blocks", "width", "base"):
            if p[key] != first[key]:
                raise ValueError(f"adapter {i} differs from adapter 1 in {key}: {p[key]!r} against {first[key]!r}")
        if set(p["state_dict"]) != set(first["state_dict"]):
            raise ValueError(f"adapter {i} has different parameters from adapter 1")
    total = float(sum(weights))
    state = {}
    for name in first["state_dict"]:
        acc = None
        for p, w in zip(payloads, weights):
            t = p["state_dict"][name].detach().cpu().double() * (w / total)
            acc = t if acc is None else acc + t
        state[name] = acc.to(first["state_dict"][name].dtype)
    return {"format": block_adapter.FORMAT, "after_blocks": first["after_blocks"], "width": first["width"],
            "base": dict(first["base"]), "state_dict": state}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="adapter.pt files from the same recipe and base")
    ap.add_argument("--out", required=True, help="where to write the averaged adapter.pt")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="one weight per input (default: equal)")
    a = ap.parse_args(argv)
    import torch

    out = Path(a.out)
    payloads = [block_adapter.read_adapter(Path(p)) for p in a.inputs]
    soup = average_payloads(payloads, a.weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(soup, tmp)
    tmp.replace(out)
    print(f"averaged {len(payloads)} adapters (after block {soup['after_blocks']}, width {soup['width']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
