from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "willspeak-adapter-v1"


class AdapterLayer(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        return h + self.linear(h)


class BlockThenAdapter(torch.nn.Module):
    def __init__(self, block, adapter: AdapterLayer):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.adapter(out[0]), *out[1:])
        return self.adapter(out)


def add_adapter(model, after_blocks: int) -> AdapterLayer:
    layers = model.encoder.layers
    if not 1 <= after_blocks <= len(layers):
        raise ValueError(f"--adapter-after-blocks must be between 1 and {len(layers)}, not {after_blocks}")
    if any(isinstance(layer, BlockThenAdapter) for layer in layers):
        raise ValueError("this model already has an adapter")
    width = getattr(model.encoder, "d_model", None) or model.cfg.encoder.d_model
    adapter = AdapterLayer(int(width))
    index = after_blocks - 1
    layers[index] = BlockThenAdapter(layers[index], adapter)
    return adapter


def freeze_all_but_adapter(model, adapter: AdapterLayer) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = True


def save_adapter(path: Path, adapter: AdapterLayer, after_blocks: int, base: dict) -> None:
    torch.save(
        {
            "format": FORMAT,
            "after_blocks": int(after_blocks),
            "width": int(adapter.linear.in_features),
            "base": dict(base),
            "state_dict": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
        },
        path,
    )


def read_adapter(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a willspeak adapter file (expected format {FORMAT})")
    return payload


def check_base(payload: dict, base: dict) -> None:
    if payload["base"] != base:
        raise ValueError(
            "this adapter was trained on a different base model.\n"
            f"  adapter file says: {payload['base']}\n"
            f"  model loaded now:  {base}\n"
            "Decode it with the model it was trained on (--stock, or --nemo with that run's --base file)."
        )


def attach_saved_adapter(model, payload: dict) -> AdapterLayer:
    adapter = add_adapter(model, int(payload["after_blocks"]))
    adapter.load_state_dict(payload["state_dict"])
    return adapter
