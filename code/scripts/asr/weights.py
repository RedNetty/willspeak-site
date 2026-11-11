#!/usr/bin/env python3
"""Blend and merge fine-tuned checkpoints in weight space, and measure how far each one moved."""
from __future__ import annotations

import argparse
import math
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WEIGHTS_NAME = "model_weights.ckpt"
STOCK_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def here(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_state(path: str | Path) -> dict:
    import torch

    path = here(path)
    with tarfile.open(path) as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == WEIGHTS_NAME), None)
        if member is None:
            raise ValueError(f"{path} has no {WEIGHTS_NAME}; it is not a .nemo checkpoint")
        handle = tar.extractfile(member)
        state = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(hasattr(v, "dtype") for v in state.values()):
        raise ValueError(f"{path}: {WEIGHTS_NAME} is not a plain dictionary of tensors")
    return state


def stock_nemo(name: str = STOCK_NAME) -> Path:
    import os

    cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    found = sorted(p for p in cache.glob(f"models--{name.replace('/', '--')}/snapshots/*/*.nemo") if p.exists())
    if not found:
        raise FileNotFoundError(f"No cached {name} under {cache}; run any decode of it once (nemo_hyp.py --pretrained {name}), or pass the .nemo path")
    return found[-1]


def check_same_layout(base: dict, other: dict, name: str) -> None:
    if base.keys() != other.keys():
        only_base = sorted(set(base) - set(other))[:5]
        only_other = sorted(set(other) - set(base))[:5]
        raise ValueError(f"{name} does not have the base model's tensors; only in base {only_base}, only in {name} {only_other}")
    for key, value in base.items():
        if tuple(value.shape) != tuple(other[key].shape):
            raise ValueError(f"{name}: tensor {key} has shape {tuple(other[key].shape)}, base has {tuple(value.shape)}")


def changes(base: dict, other: dict, name: str = "checkpoint") -> dict:
    import torch

    check_same_layout(base, other, name)
    out = {}
    for key, value in base.items():
        if not value.is_floating_point():
            continue
        mine = other[key]
        if not mine.is_floating_point():
            raise ValueError(f"{name}: tensor {key} is {mine.dtype}, base has {value.dtype}")
        if torch.equal(value, mine):
            continue
        delta = mine.float() - value.float()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError(f"{name}: tensor {key} has non-finite values")
        out[key] = delta
    return out


def blend(base: dict, parts: list[tuple[str | Path, float]]) -> tuple[dict, dict]:
    if not parts:
        raise ValueError("blend needs at least one checkpoint")
    for path, weight in parts:
        if not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError(f"Weight for {path} must be a finite number, got {weight!r}")
    total: dict = {}
    per_file = []
    for path, weight in parts:
        delta = changes(base, read_state(path), str(path))
        norm = math.sqrt(sum(float((d * d).sum()) for d in delta.values()))
        per_file.append({"file": str(path), "weight": weight, "tensors_changed": len(delta), "delta_l2": norm})
        for key, d in delta.items():
            total[key] = d * weight if key not in total else total[key].add_(d, alpha=weight)
        del delta
    state = dict(base)
    for key, d in total.items():
        state[key] = (base[key].float() + d).to(base[key].dtype)
    report = {"weights_sum": float(sum(w for _, w in parts)), "tensors": len(base),
              "tensors_changed": len(total), "files": per_file}
    return state, report


def block_of(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "encoder" and parts[1] == "layers" and parts[2].isdigit():
        return f"block{int(parts[2]):02d}"
    return ".".join(parts[:2]) if parts[0] == "encoder" else parts[0]


def geometry(base: dict, tuned: dict[str, dict]) -> dict:
    deltas = {name: changes(base, state, name) for name, state in tuned.items()}
    names = list(deltas)
    size = {}
    for name in names:
        per_block: dict[str, float] = {}
        for key, d in deltas[name].items():
            per_block[block_of(key)] = per_block.get(block_of(key), 0.0) + float((d * d).sum())
        size[name] = {"total_l2": math.sqrt(sum(per_block.values())),
                      "per_block_l2": {b: math.sqrt(v) for b, v in sorted(per_block.items())}}
    base_norm = {}
    for key, value in base.items():
        if value.is_floating_point():
            base_norm[block_of(key)] = base_norm.get(block_of(key), 0.0) + float((value.float() ** 2).sum())
    for name in names:
        size[name]["relative_to_weights"] = {b: v / math.sqrt(base_norm[b]) if base_norm.get(b) else None
                                            for b, v in size[name]["per_block_l2"].items()}
    cosine = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys = deltas[a].keys() & deltas[b].keys()
            dot = sum(float((deltas[a][k] * deltas[b][k]).sum()) for k in keys)
            denom = size[a]["total_l2"] * size[b]["total_l2"]
            cosine[f"{a}~{b}"] = dot / denom if denom else None
    return {"change_size": size, "cosine_between_changes": cosine}


def main() -> int:
    import labkit as lab

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the .nemo every checkpoint was trained from, or 'auto' for the cached stock model")
    ap.add_argument("--tuned", nargs="+", required=True, metavar="NAME=FILE", help="fine-tuned .nemo checkpoints, each with a short name")
    ap.add_argument("--out", required=True, help="geometry .json to write")
    a = ap.parse_args()
    lab.refuse_existing(a.out)
    named = {}
    for item in a.tuned:
        if "=" not in item:
            ap.error(f"--tuned wants NAME=FILE, got {item!r}")
        name, path = item.split("=", 1)
        if name in named:
            ap.error(f"--tuned lists {name} twice")
        named[name] = path
    base_path = stock_nemo() if a.base == "auto" else here(a.base)
    base = read_state(base_path)
    tuned = {name: read_state(path) for name, path in named.items()}
    result = geometry(base, tuned)
    result["inputs"] = {"base": str(base_path), **named}
    result["provenance"] = lab.provenance(__file__, {})
    lab.save_json(a.out, result)
    for name, info in result["change_size"].items():
        print(f"{name}: change L2 {info['total_l2']:.3f} over {len(info['per_block_l2'])} groups")
    for pair, value in result["cosine_between_changes"].items():
        print(f"cosine {pair}: {value if value is None else round(value, 4)}")
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "weights"))
