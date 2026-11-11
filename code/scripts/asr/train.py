#!/usr/bin/env python3
"""Fine-tune Parakeet from train/dev manifests; never open the test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cli
import wandb_log
from willspeak.artifacts import file_record, record_id, timestamp, validate_export, versions, write_json

BASE = "nvidia/parakeet-tdt-0.6b-v2"
GOOD = {"good_day", "unknown"}

BLOCK_PARTS = (
    "norm_feed_forward1", "feed_forward1",
    "norm_self_att", "self_attn",
    "norm_conv", "conv",
    "norm_feed_forward2", "feed_forward2",
    "norm_out",
)
LOWER_BLOCKS = 8
_active_run = None
_wandb = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [train] {msg}", flush=True)


def log_section(title: str, fields: dict) -> None:
    log(f"--- {title} ---")
    width = max((len(key) for key in fields), default=0)
    for key, value in fields.items():
        log(f"  {key:<{width}} : {value}")


def elapsed_text(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def wandb_start(run: dict, a, out: Path):
    if a.wandb:
        import os
        os.environ["WILLSPEAK_WANDB"] = a.wandb
    if wandb_log.mode() == "off":
        return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=3).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    base_short = Path(a.base).stem if Path(a.base).is_file() else a.base.split("/")[-1]
    config = dict(run["configuration"])
    config.update(base_model=a.base, export_name=Path(a.export).name, git_commit=commit,
                  device=run.get("device"), cuda=run.get("cuda"), train_clips=run.get("train_clips"),
                  unique_train_minutes=run.get("unique_train_minutes"), dev_clips=run.get("dev_clips"))
    tags = ["train", base_short, *[t.strip() for t in (a.wandb_tags or "").split(",") if t.strip()]]
    return wandb_log.start(a.wandb_name or out.name, config, group=a.wandb_group, tags=tags,
                           job_type="train", run_id=None)


def wandb_close(run: dict, out: Path, status: str) -> None:
    handle = _wandb
    if handle is None or handle.closed:
        return
    curve = run.get("curve") or []
    wandb_log.summary(handle, {
        "released_dev_wer": run.get("released_dev_wer"),
        "released_dev_wer_pct": None if run.get("released_dev_wer") is None else run["released_dev_wer"] * 100,
        "best_dev_wer": run.get("best_dev_wer"),
        "base_dev_wer": run.get("stock_dev_wer"),
        "completed_epochs": len(curve),
        "optimizer_updates": curve[-1]["optimizer_steps"] if curve else 0,
        "elapsed_seconds": None if run.get("minutes") is None else run["minutes"] * 60,
        "trainable_parameters": (run.get("parameters") or {}).get("trainable"),
        "total_parameters": (run.get("parameters") or {}).get("total"),
        "selected": run.get("selected"),
        "stop_reason": run.get("stop_reason"),
        "error": run.get("error"),
    })
    for name in ("run.json", "curve.csv"):
        wandb_log.log_file(handle, out / name, kind="train-record")
    wandb_log.finish(handle, status)


def release_summary(run: dict, final: Path, seconds: float) -> None:
    def wer(value):
        return f"{value * 100:.2f}%" if value is not None else "not measured"

    log_section("COMPLETE", {
        "Released": run["selected"],
        "Training stopped": run.get("stop_reason", "not recorded"),
        "Completed epochs": len(run.get("curve", [])),
        "Optimizer updates": sum(row.get("epoch_optimizer_steps", 0) for row in run.get("curve", [])),
        "Base dev WER": wer(run.get("stock_dev_wer")),
        "Best checkpoint dev WER": wer(run.get("best_dev_wer")),
        "Released model dev WER": wer(run.get("released_dev_wer")),
        "Checkpoints averaged": len(run.get("averaged_checkpoints", [])),
        "Elapsed": elapsed_text(seconds),
        "Saved": f"{final} ({final.stat().st_size / 1e6:.1f} MB)",
        "Records": f"{final.parent / 'run.json'}; {final.parent / 'curve.csv'}",
    })
    log("Test WER is measured separately with evaluate.py on the held-out test manifest.")


def write_nemo_manifest(rows: list[dict], audio_root: Path, out: Path, normalize=None) -> int:
    n = 0
    skipped = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            text = " ".join(str(r["text"]).split())
            if normalize is not None:
                text = normalize(text)
            if not text:
                skipped += 1
                continue
            f.write(
                json.dumps(
                    {
                        "audio_filepath": str((audio_root / r["audio"]).resolve()),
                        "duration": float(r["duration_s"]),
                        "text": text,
                        "condition": r.get("condition", "unknown"),
                    }
                )
                + "\n"
            )
            n += 1
    write_nemo_manifest.skipped = skipped
    return n


write_nemo_manifest.skipped = 0


def target_checker(model):
    tok = getattr(model, "tokenizer", None)
    unk = getattr(tok, "unk_id", None)
    if tok is None or unk is None or not hasattr(tok, "text_to_ids"):
        return None

    def check(texts):
        bad = []
        for t in texts:
            try:
                ids = tok.text_to_ids(t)
            except Exception:  # noqa: BLE001
                ids = []
            if unk in ids:
                bad.append(t)
        return bad
    return check


def freeze_lower_encoder(model, n_layers: int) -> int:
    if n_layers <= 0:
        return 0
    enc = model.encoder
    layers = list(getattr(enc, "layers", []))
    n = min(n_layers, len(layers))
    for name in ("pre_encode", "pos_enc"):
        mod = getattr(enc, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
    for layer in layers[:n]:
        for p in layer.parameters():
            p.requires_grad = False
    return n


def parse_train_parts(text: str) -> tuple[str, ...]:
    if text.strip() == "all":
        return BLOCK_PARTS
    parts = tuple(p.strip() for p in text.split(",") if p.strip())
    unknown = [p for p in parts if p not in BLOCK_PARTS]
    if not parts or unknown:
        raise ValueError(f"--train-parts: unknown part(s) {unknown or [text]}; "
                         f"use 'all' or a comma list of: {', '.join(BLOCK_PARTS)}")
    return parts


EXTRA_AUG_PROB = 0.5


def extra_aug_prob(a) -> float:
    return EXTRA_AUG_PROB if a.extra_aug_prob is None else float(a.extra_aug_prob)


def build_augmentor(a) -> dict | None:
    chain: dict = {}
    if not a.no_speed_perturb:
        lo, hi = a.speed_perturb
        chain["speed"] = {"prob": 1.0, "sr": 16000, "resample_type": "kaiser_fast",
                          "min_speed_rate": float(lo), "max_speed_rate": float(hi), "num_rates": 5}
    if a.time_stretch:
        lo, hi = a.time_stretch
        chain["time_stretch"] = {"prob": extra_aug_prob(a), "min_speed_rate": float(lo),
                                 "max_speed_rate": float(hi), "num_rates": 5, "n_fft": 512}
    if a.white_noise:
        lo, hi = a.white_noise
        chain["white_noise"] = {"prob": extra_aug_prob(a), "min_level": int(lo), "max_level": int(hi)}
    return chain or None


def augmentation_text(a) -> str:
    parts = []
    if not a.no_speed_perturb:
        parts.append(f"speed {a.speed_perturb[0]:g}-{a.speed_perturb[1]:g} on every clip")
    if a.time_stretch:
        parts.append(f"tempo {a.time_stretch[0]:g}-{a.time_stretch[1]:g}, pitch kept, "
                     f"on {extra_aug_prob(a):.0%} of clips")
    if a.white_noise:
        parts.append(f"white noise {a.white_noise[0]} to {a.white_noise[1] - 1} dBFS "
                     f"on {extra_aug_prob(a):.0%} of clips")
    return "; ".join(parts) or "none"


def check_augmentor(augmentor: dict | None) -> list[str]:
    import copy
    import random
    from types import SimpleNamespace

    import numpy as np
    from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

    problems: list[str] = []
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        for name in ("time_stretch", "white_noise"):
            if not augmentor or name not in augmentor:
                continue
            try:
                cfg = copy.deepcopy({name: augmentor[name]})
                cfg[name]["prob"] = 1.0
                chain = process_augmentations(cfg)
                seen = []
                for _ in range(8):
                    base = (np.random.randn(16000) * 0.05).astype(np.float32)
                    clip = SimpleNamespace(_samples=base.copy())
                    chain.perturb(clip)
                    out = np.asarray(clip._samples)
                    if out.size == 0 or not np.isfinite(out).all():
                        raise ValueError("it returned empty or non-finite audio")
                    seen.append(out.size / 16000 if name == "time_stretch"
                                else 20 * math.log10(float(np.std(out - base)) + 1e-12))
                if name == "time_stretch":
                    log(f"augmentation check: tempo stretch turned 1.00 s of test audio into "
                        f"{min(seen):.2f}-{max(seen):.2f} s")
                else:
                    log(f"augmentation check: white noise measured {min(seen):.0f} to {max(seen):.0f} dBFS "
                        "on test audio")
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                problems.append(f"{name} failed on a test clip: {msg}")
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return problems


def train_lower_parts(model, parts: tuple[str, ...], n_blocks: int = LOWER_BLOCKS, first_block: int = 0) -> int:
    enc = model.encoder
    for p in enc.parameters():
        p.requires_grad = False
    layers = list(getattr(enc, "layers", []))
    if first_block < 0 or n_blocks < 1 or (layers and first_block + n_blocks > len(layers)):
        raise ValueError(f"Encoder blocks {first_block}-{first_block + n_blocks - 1} do not exist; "
                         f"the model has {len(layers)} blocks")
    blocks = layers[first_block:first_block + n_blocks]
    for block in blocks:
        if parts == BLOCK_PARTS:
            for p in block.parameters():
                p.requires_grad = True
            continue
        for name in parts:
            module = getattr(block, name, None)
            if module is None:
                raise ValueError(f"encoder block has no part named {name!r}")
            for p in module.parameters():
                p.requires_grad = True
    return len(blocks)


def freeze_transducer_head(model) -> int:
    modules = [getattr(model, name, None) for name in ("decoder", "joint")]
    if any(module is None for module in modules):
        raise ValueError("--freeze-transducer-head needs a transducer decoder and joint")
    parameters = {id(p): p for module in modules for p in module.parameters()}
    for p in parameters.values():
        p.requires_grad = False
    return sum(p.numel() for p in parameters.values())


def score_dev_like_evaluate(normalize) -> None:
    from nemo.collections.asr.metrics import wer as wer_module

    cls = wer_module.WER
    if getattr(cls, "_scores_like_evaluate", False):
        return
    original_update = cls.update

    def update(self, *args, **kwargs):
        decode, ids_to_str = self.decode, self.decoding.decode_ids_to_str

        def normalised_decode(*a, **k):
            hyps = decode(*a, **k)
            for h in hyps or []:
                h = h[0] if isinstance(h, list) else h
                h.text = normalize(h.text or "")
            return hyps

        self.decode = normalised_decode
        self.decoding.decode_ids_to_str = lambda ids: normalize(ids_to_str(ids) or "")
        try:
            return original_update(self, *args, **kwargs)
        finally:
            self.decode = decode
            self.decoding.decode_ids_to_str = ids_to_str

    cls.update = update
    cls._scores_like_evaluate = update


def check_dev_scorer(model) -> None:
    from nemo.collections.asr.metrics import wer as wer_module

    ours = getattr(wer_module.WER, "_scores_like_evaluate", None)
    fn = getattr(model, "wer", None)
    fn = getattr(fn, "update", None)
    for _ in range(5):
        if fn is None or fn is ours:
            break
        fn = getattr(fn, "__wrapped__", None) or getattr(fn, "__func__", None)
    if fn is not ours:
        raise RuntimeError("the model's dev scorer is not the evaluate.py-style one; "
                           "score_dev_like_evaluate() must run before the model is loaded")


def offloaded_adamw(torch):
    class OffloadedAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
            super().__init__(params, {"lr": lr, "betas": tuple(betas), "eps": eps,
                                      "weight_decay": weight_decay, "amsgrad": amsgrad})
            pin = torch.cuda.is_available()
            self._pairs = []
            self._groups = []
            ram_groups = []
            for group in self.param_groups:
                ram_params = []
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    h = torch.empty(p.shape, dtype=torch.float32, device="cpu", pin_memory=pin)
                    h.copy_(p.detach())
                    buf = torch.empty(p.shape, dtype=torch.float32, device="cpu", pin_memory=pin)
                    self._pairs.append((p, h, buf))
                    ram_params.append(h)
                if ram_params:
                    ram_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": ram_params})
                    self._groups.append(group)
            try:
                self.cpu = torch.optim.AdamW(ram_groups, fused=True)
            except (RuntimeError, ValueError):
                self.cpu = torch.optim.AdamW(ram_groups)
            self.trained_numbers = sum(p.numel() for p, _, _ in self._pairs)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for outer, inner in zip(self._groups, self.cpu.param_groups):
                for k in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
                    inner[k] = outer[k]
            moved = []
            for p, h, buf in self._pairs:
                if p.grad is None:
                    h.grad = None
                    continue
                buf.copy_(p.grad, non_blocking=True)
                h.grad = buf
                moved.append((p, h))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.cpu.step()
            for p, h in moved:
                p.copy_(h, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return loss

        def state_dict(self):
            return self.cpu.state_dict()

        def load_state_dict(self, state_dict):
            self.cpu.load_state_dict(state_dict)

    return OffloadedAdamW


def batchnorm_freezer(pl, torch):
    class FreezeBatchNorm(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, *args):
            for module in pl_module.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                    module.eval()

    return FreezeBatchNorm()


def curve_recorder(pl, run: dict, out: Path, torch, phase: str = "train_all"):
    class Curve(pl.Callback):
        def __init__(self):
            self.losses: list[float] = []
            self.started = time.monotonic()
            self.seconds = None
            self.last_progress = self.started
            self.best_wer = None
            self.epoch_active = False
            self.gpu = False
            self.validation_started = None
            self.validation_seconds = 0.0
            self.validation_complete = False
            self.steps_at_start = 0
            self.batches_seen = 0
            self.learning_rate = None
            self.pending_losses: list[float] = []
            self.logged_step = -1

        def on_train_epoch_start(self, trainer, pl_module):
            self.losses = []
            self.started = time.monotonic()
            self.seconds = None
            self.last_progress = self.started
            self.epoch_active = True
            self.validation_seconds = 0.0
            self.validation_started = None
            self.validation_complete = False
            self.steps_at_start = int(trainer.global_step)
            self.batches_seen = 0
            self.learning_rate = None
            self.planned_batches = int(trainer.num_training_batches)
            cap = getattr(trainer, "max_steps", -1)
            if cap > 0:
                self.planned_batches = min(self.planned_batches,
                                           (cap - trainer.global_step) * trainer.accumulate_grad_batches)
            self.gpu = getattr(getattr(getattr(trainer, "strategy", None), "root_device", None), "type", None) == "cuda"
            if self.gpu:
                torch.cuda.reset_peak_memory_stats()
            log(f"{phase} | epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | "
                f"{self.planned_batches} batches | starting at optimizer step {trainer.global_step}")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None:
                value = float(loss.detach().float()) * trainer.accumulate_grad_batches
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite training loss at optimizer step {trainer.global_step}")
                self.losses.append(value)
                self.pending_losses.append(value)
            now = time.monotonic()
            self.seconds = now - self.started - self.validation_seconds
            self.batches_seen = batch_idx + 1
            self.learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
            step = int(trainer.global_step)
            if step != self.logged_step and self.pending_losses:
                wandb_log.log(_wandb, {"optimizer_step": step,
                                       "train/loss": sum(self.pending_losses) / len(self.pending_losses),
                                       "train/lr": self.learning_rate}, step_metric="optimizer_step")
                self.logged_step = step
                self.pending_losses = []
            cap = getattr(trainer, "max_steps", -1)
            if cap > 0 and trainer.global_step >= cap:
                trainer.should_stop = True
            every = run["configuration"].get("log_every_seconds", 30)
            if now - self.last_progress >= every:
                self.last_progress = now
                batches = self.planned_batches
                speed = (batch_idx + 1) / max(self.seconds, 1e-6)
                eta = (batches - batch_idx - 1) / speed
                loss_text = f"{self.losses[-1]:.4f}" if self.losses else "n/a"
                log(f"  batch {batch_idx + 1}/{batches} ({100 * (batch_idx + 1) / batches:.0f}%) | "
                    f"step {trainer.global_step} | loss {loss_text} | lr {self.learning_rate:.2e} | "
                    f"{speed:.2f} batches/s | train ETA {elapsed_text(eta)}")

        def on_validation_start(self, trainer, pl_module):
            if self.epoch_active:
                self.validation_started = time.monotonic()
                self.last_progress = self.validation_started
                log(f"  Validating {phase} epoch {trainer.current_epoch + 1} on dev audio...")

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            now = time.monotonic()
            if self.epoch_active and now - self.last_progress >= run["configuration"].get("log_every_seconds", 30):
                self.last_progress = now
                batches = trainer.num_val_batches[dataloader_idx]
                log(f"  dev batch {batch_idx + 1}/{batches} | "
                    f"elapsed {elapsed_text(now - self.validation_started)}")

        def on_validation_end(self, trainer, pl_module):
            if self.epoch_active and self.validation_started is not None:
                self.validation_seconds += time.monotonic() - self.validation_started
                self.validation_started = None
                self.validation_complete = True

        def on_train_epoch_end(self, trainer, pl_module):
            wer = trainer.callback_metrics.get("val_wer")
            value = float(wer) if wer is not None else None
            if not self.validation_complete or value is None or not math.isfinite(value) or value < 0:
                raise RuntimeError("Validation produced no finite dev WER; refusing to select a checkpoint")
            previous_best = self.best_wer
            self.best_wer = min(value, self.best_wer) if self.best_wer is not None else value
            row = {
                "phase": phase,
                "epoch": int(trainer.current_epoch),
                "display_epoch": int(trainer.current_epoch) + 1,
                "optimizer_steps": int(trainer.global_step),
                "epoch_optimizer_steps": int(trainer.global_step) - self.steps_at_start,
                "batches": self.batches_seen,
                "learning_rate": self.learning_rate,
                "dev_wer": round(value, 4),
                "train_loss": round(sum(self.losses) / len(self.losses), 4) if self.losses else None,
                "train_minutes": round(self.seconds / 60, 2) if self.seconds is not None else None,
                "train_seconds": round(self.seconds, 2) if self.seconds is not None else None,
                "validation_seconds": round(self.validation_seconds, 2),
                "epoch_seconds": round(time.monotonic() - self.started, 2),
                "peak_gpu_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                          if self.gpu else None),
                "peak_gpu_gb": (round(torch.cuda.max_memory_reserved() / 1e9, 2)
                                if self.gpu else None),
            }
            run.setdefault("curve", []).append(row)
            write_json(out / "run.json", run)
            path = out / "curve.csv"
            new = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            wandb_log.log(_wandb, {
                "epoch": len(run["curve"]),
                "epoch/dev_wer": row["dev_wer"],
                "epoch/dev_wer_pct": row["dev_wer"] * 100,
                "epoch/train_loss": row["train_loss"],
                "epoch/learning_rate": row["learning_rate"],
                "epoch/optimizer_steps": row["optimizer_steps"],
                "epoch/train_seconds": row["train_seconds"],
                "epoch/validation_seconds": row["validation_seconds"],
                "epoch/epoch_seconds": row["epoch_seconds"],
                "epoch/peak_gpu_gb": row["peak_gpu_gb"],
            }, step_metric="epoch")
            config = run["configuration"]
            selection = "fixed schedule" if config.get("select") == "last" else (
                "new best" if previous_best is None or value < previous_best else "no improvement")
            stock = run.get("stock_dev_wer")
            delta = f" | base gain {(stock - value) * 100:+.2f} pp" if stock is not None else ""
            loss_text = f"{row['train_loss']:.4f}" if row["train_loss"] is not None else "n/a"
            gpu_text = f"{row['peak_gpu_gb']:.2f} GB" if row["peak_gpu_gb"] is not None else "n/a (CPU)"
            lr_text = f"{self.learning_rate:.2e}" if self.learning_rate is not None else "n/a"
            log(f"Epoch {row['epoch'] + 1}/{trainer.max_epochs} | dev WER {value * 100:.2f}%{delta} | "
                f"{selection}")
            log(f"  loss {loss_text} | lr {lr_text} | "
                f"updates {row['epoch_optimizer_steps']} this epoch / {row['optimizer_steps']} total | "
                f"train {elapsed_text(self.seconds or 0)} | dev {elapsed_text(self.validation_seconds)}")
            allocated = row["peak_gpu_allocated_gb"]
            log(f"  peak GPU allocated {allocated:.2f} GB | reserved {gpu_text}" if allocated is not None
                else "  peak GPU memory: n/a (CPU)")
            self.epoch_active = False

        def on_exception(self, trainer, pl_module, exception):
            run.update(status="interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                       finished_at=timestamp(), error=f"{type(exception).__name__}: {exception}")
            write_json(out / "run.json", run)
            wandb_close(run, out, run["status"])
            log(f"Run {run['status']} at optimizer step {trainer.global_step}: {exception}")

    return Curve()


def save_only_what_changes(model) -> None:
    keep = {name for name, p in model.named_parameters() if p.requires_grad}
    keep |= {name for name, _ in model.named_buffers()}
    original = model.on_save_checkpoint

    def pruned(checkpoint):
        original(checkpoint)
        sd = checkpoint.get("state_dict", {})
        for key in [k for k in sd if k not in keep]:
            del sd[key]

    model.on_save_checkpoint = pruned


def load_trained_weights(model, state: dict) -> None:
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    result = model.load_state_dict(state, strict=False)
    if result is None:
        return
    missing = set(result.missing_keys) - frozen
    if result.unexpected_keys or missing:
        raise RuntimeError(f"Checkpoint does not match the model: missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(result.unexpected_keys)[:5]}")


def average_checkpoints(model, ckpt, top_n: int, torch, below: float | None = None) -> list[str]:
    ranked = sorted(ckpt.best_k_models, key=lambda k: float(ckpt.best_k_models[k]))
    if below is not None:
        ranked = [p for p in ranked if float(ckpt.best_k_models[p]) < below]
    paths = ranked[: max(1, top_n)]
    if len(paths) < 2:
        return [str(p) for p in paths]
    avg = None
    keys = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
        if keys is not None and set(sd) != keys:
            raise RuntimeError("Cannot average checkpoints with different state keys")
        keys = set(sd)
        if avg is None:
            avg = {k: v.detach().clone().float() for k, v in sd.items() if v.is_floating_point()}
        else:
            for k, v in sd.items():
                if k in avg:
                    if v.shape != avg[k].shape:
                        raise RuntimeError(f"Cannot average checkpoint tensor {k!r} with different shapes")
                    avg[k] += v.float()
    for k in avg:
        avg[k] /= len(paths)
    current = model.state_dict()
    state = {k: v for k, v in current.items() if not v.is_floating_point()}
    state.update({k: v.to(current[k].dtype) for k, v in avg.items()})
    load_trained_weights(model, state)
    log(f"averaged {len(paths)} checkpoints")
    return [str(p) for p in paths]


def visible_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f if line.split()}
        return (info.get("MemTotal", 0) + info.get("SwapTotal", 0)) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return None


def disk_space_problems(out: Path, average_top: int, frozen_layers: int) -> list[str]:
    import shutil
    import tempfile

    gb = 1024 ** 3
    problems = []
    target = out
    while not target.exists():
        target = target.parent
    share = max(0.05, (24 - min(max(frozen_layers, 0), 24)) / 24)
    need_out = max(1, average_top) * 2.5 * share + 2.5
    free_out = shutil.disk_usage(target).free / gb
    if free_out < need_out:
        problems.append(f"{free_out:.1f} GB free where {out} goes; this run needs about {need_out:.0f} GB "
                        "(checkpoints + final .nemo). Use --out under models/, not /tmp")
    tmp = Path(tempfile.gettempdir())
    free_tmp = shutil.disk_usage(tmp).free / gb
    if free_tmp < 3:
        problems.append(f"only {free_tmp:.1f} GB free in the temp folder {tmp}, where the .nemo is staged; "
                        "export TMPDIR=$HOME/.cache/willspeak-tmp (./willspeak.sh train-setup adds it)")
    return problems


def gpu_environment_problems() -> list[str]:
    import os

    problems: list[str] = []
    try:
        import nvidia.cuda_nvcc as _nvcc
        nvcc_dir = Path(next(iter(_nvcc.__path__)))
        if not os.environ.get("CUDA_HOME") and (nvcc_dir / "nvvm").is_dir():
            os.environ["CUDA_HOME"] = str(nvcc_dir)
    except ImportError:
        problems.append("nvidia-cuda-nvcc-cu12 missing (libNVVM for the TDT loss): "
                        'python -m pip install -e ".[train]"')
    try:
        import resampy  # noqa: F401
    except ImportError:
        problems.append("resampy missing -speed perturbation would be skipped silently: "
                        "python -m pip install resampy")
    try:
        import numpy as np
        from numba import cuda

        @cuda.jit
        def _probe(arr):
            arr[0] = 1.0

        dev = cuda.to_device(np.zeros(1))
        _probe[1, 1](dev)
        if dev.copy_to_host()[0] != 1.0:
            problems.append("numba CUDA probe kernel ran but returned the wrong value")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        hint = "see docs/training.md Troubleshooting"
        if "row_stack" in msg:
            hint = 'python -m pip install "numpy<2.3"'
        elif "NVVM" in msg or "nvvm" in msg:
            hint = "re-activate the env after ./willspeak.sh train-setup (CUDA_HOME / LD_LIBRARY_PATH)"
        elif "compute capabilit" in msg:
            hint = 'python -m pip install "numba-cuda[cu12]"'
        problems.append(f"numba cannot compile for the GPU ({msg}): {hint}")
    return problems


def main() -> int:
    global _active_run, _wandb
    _active_run = None
    _wandb = None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/export")
    ap.add_argument("--out", default="models/parakeet-ft")
    ap.add_argument("--base", default=BASE, help="HF name or a local .nemo to start from")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="phase 1: good/typical days only (--curriculum-epochs), then phase 2: all data",
    )
    ap.add_argument("--curriculum-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-duration", type=float, default=25.0)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--wandb", choices=("auto", "online", "offline", "off"), default=None,
                    help="experiment tracking: metrics and small records only, never audio, text or weights "
                         "(default: WILLSPEAK_WANDB, else auto)")
    ap.add_argument("--wandb-group", default=None, help="group for related runs in the tracking project")
    ap.add_argument("--wandb-tags", default=None, help="comma-separated extra tags")
    ap.add_argument("--wandb-name", default=None, help="tracking run name (default: the output folder name)")
    ap.add_argument("--verbose", action="store_true", help="include NeMo setup details and decoded dev examples")
    ap.add_argument("--log-every-seconds", type=float, default=30,
                    help="interval between plain-text batch progress lines (default: 30 seconds)")
    ap.add_argument("--freeze-encoder-layers", type=int, default=0,
                    help="freeze the subsampling front end and the lowest N of the encoder's conformer layers "
                         "(0 = full fine-tune; 16 fits an 8 GB card and guards against forgetting)")
    ap.add_argument("--train-parts", default=None,
                    help="train only encoder blocks 0-7 (or the window --train-blocks and --train-first-block "
                         "choose) and freeze the rest of the encoder: 'all' for whole blocks, or a comma list of "
                         "parts such as self_attn,conv (the which-parts comparison). "
                         "Cannot be combined with --freeze-encoder-layers")
    ap.add_argument("--train-blocks", type=int, default=LOWER_BLOCKS, metavar="N",
                    help="how many encoder blocks --train-parts opens (default 8); the depth sweep varies it")
    ap.add_argument("--train-first-block", type=int, default=0, metavar="K",
                    help="the first block --train-parts opens (default 0, the bottom); 8 with --train-blocks 8 "
                         "trains blocks 8-15 and leaves the lowest blocks alone")
    ap.add_argument("--freeze-transducer-head", action="store_true",
                    help="freeze the prediction network and joint; train only the requested encoder weights. "
                         "Gradients still pass through the frozen head. Off by default; not combined with an adapter")
    ap.add_argument("--empty-margin", type=float, default=0.0, metavar="NATS",
                    help="silence margin (lab test 10): penalise every training clip with words on which saying "
                         "nothing is less than NATS nats unlikely, so the model cannot settle on an empty answer. "
                         "0 = off. TDT models only; see margin_loss.py")
    ap.add_argument("--empty-weight", type=float, default=1.0,
                    help="weight of the --empty-margin penalty in the loss (default 1)")
    ap.add_argument("--empty-clips-per-kind", type=int, default=0, metavar="K",
                    help="add K synthetic non-speech clips of each of battery.py's 7 kinds to every training epoch, "
                         "with an empty transcript, so the model learns to answer noise with nothing (lab test 10). "
                         "TDT models only; the clips are written under the run folder")
    ap.add_argument("--empty-clips-seed", type=int, default=1000,
                    help="seed of those clips; the test battery uses 2000, so keep this below it")
    ap.add_argument("--adapter-after-blocks", type=int, default=None, metavar="N",
                    help="put a small trainable adapter after the first N encoder blocks and train only it: it passes on "
                         "h + W h + b (1,049,600 numbers) and starts at zero, so the model is unchanged until training "
                         "moves it. N=8 sits after blocks 0-7. Everything else stays frozen, so the run saves adapter.pt "
                         "(decode it with nemo_hyp.py --adapter), not a .nemo. The adapter needs a larger --lr than "
                         "block training (3e-4 in the research). Cannot be combined with --train-parts or "
                         "--freeze-encoder-layers")
    ap.add_argument("--freeze-batchnorm", action=argparse.BooleanOptionalAction, default=True,
                    help="keep every batch-norm layer's stored statistics at the base model's values (default). "
                         "--no-freeze-batchnorm lets batches of two clips rewrite them in every layer, which cost "
                         "19.6 WER points on TORGO F01 in a six-step run (24 Sep 2026)")
    ap.add_argument("--normalize-targets", action=argparse.BooleanOptionalAction, default=True,
                    help="train on transcripts in the scoring form (lowercase, no punctuation) so dev WER, "
                         "which picks the checkpoints, counts words and not formatting (default). "
                         "--no-normalize-targets reproduces the first F-2 run of 24 Sep 2026")
    ap.add_argument("--target-form", choices=("scoring", "stock", "raw"), default=None,
                    help="the form of the training targets. 'scoring' is lowercase without punctuation (what "
                         "--normalize-targets gives, and what every recorded run used); 'stock' is the model's own "
                         "output form, a capital first letter and a terminal mark, so 'tomato' is trained as "
                         "'Tomato.' on the token path the stock model already uses; 'raw' is the manifest text as "
                         "written. Dev WER normalises both sides whatever the form. Overrides --normalize-targets")
    ap.add_argument("--weight-decay", type=float, default=1e-3,
                    help="AdamW weight decay on every trainable parameter (default 1e-3, as in every recorded run); "
                         "on an adapter that starts at zero it pulls the layer back toward identity each step")
    ap.add_argument("--dev-scoring", choices=("evaluate", "nemo"), default="evaluate",
                    help="how dev WER picks checkpoints: 'evaluate' normalises output and transcript the way "
                         "evaluate.py does (default from 27 Sep 2026); 'nemo' is NeMo's raw comparison, which "
                         "counts capitals and punctuation, as used by every run before the which-parts rerun")
    ap.add_argument("--keep-stock", action=argparse.BooleanOptionalAction, default=True,
                    help="score the untrained model on dev first and release it unchanged if no checkpoint "
                         "beats it (--no-keep-stock reproduces runs before 27 Sep 2026)")
    ap.add_argument("--select", choices=("dev", "last"), default="dev",
                    help="which weights are released: 'dev' averages the best --average-top checkpoints by dev "
                         "WER and stops early after --patience epochs without improvement (default); 'last' "
                         "trains exactly --epochs epochs and averages the last --average-top of them, so the "
                         "dev set decides nothing (F-2 speaker runs: M01's dev set is too small to choose on). 'last' needs "
                         "--no-keep-stock, because keeping stock is itself a choice made on dev")
    ap.add_argument("--kinds", default=None,
                    help="train and pick checkpoints on only these recording kinds from the manifest's 'kind' "
                         "field: 'sentence', 'word', or both comma-separated (default: every recording)")
    ap.add_argument("--dev-speakers", default=None,
                    help="pick checkpoints on only these speakers' dev recordings (comma list, e.g. "
                         "F01,M01,M02,M04), so selection follows the speakers that matter most")
    ap.add_argument("--repeat-speakers", default=None,
                    help="comma list of speakers whose training recordings are used more than once per epoch "
                         "(see --repeat)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="how many times each --repeat-speakers training recording appears per epoch (default 2)")
    ap.add_argument("--patience", type=int, default=3, help="stop when dev WER has not improved for this many epochs")
    ap.add_argument("--speed-perturb", type=float, nargs=2, default=(0.85, 1.15), metavar=("MIN", "MAX"),
                    help="speed-perturbation range applied to every training clip (the strongest augmentation "
                         "for low-intelligibility speech in the literature); --no-speed-perturb disables it")
    ap.add_argument("--no-speed-perturb", action="store_true")
    ap.add_argument("--time-stretch", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="also change the tempo of training clips by a factor drawn from MIN-MAX without moving "
                         "the pitch (1.25 is 25 percent faster, 0.75 is 25 percent slower); --speed-perturb "
                         "changes tempo and pitch together. Off by default; runs after speed perturbation on "
                         "--extra-aug-prob of clips")
    ap.add_argument("--white-noise", type=int, nargs=2, default=None, metavar=("MIN_DB", "MAX_DB"),
                    help="also add white noise to training clips at a whole-number level in dB below full scale "
                         "drawn from MIN_DB up to but not including MAX_DB (for example -70 -40). Off by "
                         "default; runs on --extra-aug-prob of clips")
    ap.add_argument("--extra-aug-prob", type=float, default=None, metavar="P",
                    help=f"chance that each of --time-stretch and --white-noise runs on a clip (default "
                         f"{EXTRA_AUG_PROB}); only valid with one of them")
    ap.add_argument("--spec-augment", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the base model's SpecAugment, which blanks random stretches of time and frequency "
                         "in every training clip (default, as in every run so far); --no-spec-augment trains "
                         "on the clips with speed perturbation only")
    ap.add_argument("--average-top", type=int, default=4,
                    help="average the weights of the best N dev-WER checkpoints before saving (1 = best only)")
    ap.add_argument("--max-steps", type=int, default=0, help="plumbing checks: stop after N optimizer steps")
    ap.add_argument("--offload-optimizer", action=argparse.BooleanOptionalAction, default=False,
                    help="keep AdamW's running averages and its update in system RAM instead of on the GPU: "
                         "frees about 1.7 GB of GPU memory with --train-parts all, at the cost of copying the "
                         "gradients and weights once per optimizer step; same AdamW")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without CUDA: reaches the first training step and stops there, because NeMo's TDT "
                         "loss is CUDA-only (verified 10 Sep 2026). Useful to prove manifests, model load and "
                         "freezing work before booking GPU time; not a way to train")
    ap.add_argument(
        "--check", action="store_true", help="validate train/dev and exit; no CUDA, downloads, or training files"
    )
    a = ap.parse_args()

    export = Path(a.export) if Path(a.export).is_absolute() else ROOT / a.export
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    base_path = Path(a.base) if Path(a.base).is_absolute() else ROOT / a.base
    if base_path.is_file() or base_path.suffix == ".nemo":
        if not base_path.is_file():
            ap.error(f"Base checkpoint not found: {base_path}")
        a.base = str(base_path.resolve())
    if (min(a.epochs, a.curriculum_epochs, a.batch, a.grad_accum, a.average_top, a.patience) < 1
            or any(not math.isfinite(x) or x <= 0 for x in (a.lr, a.max_duration, a.log_every_seconds))
            or min(a.max_steps, a.warmup_steps, a.freeze_encoder_layers) < 0
            or not math.isfinite(a.weight_decay) or a.weight_decay < 0):
        ap.error("epochs, batch, accumulation, averaging, patience, learning rate, duration and log interval "
                 "must be positive; steps, warmup, frozen layers and weight decay must be nonnegative")
    if a.target_form is None:
        a.target_form = "scoring" if a.normalize_targets else "raw"
    if not all(math.isfinite(x) and x > 0 for x in a.speed_perturb) or a.speed_perturb[0] > a.speed_perturb[1]:
        ap.error("--speed-perturb requires positive finite speeds in increasing order")
    if a.time_stretch and (not all(math.isfinite(x) and x > 0 for x in a.time_stretch)
                           or a.time_stretch[0] > a.time_stretch[1]):
        ap.error("--time-stretch requires positive finite factors in increasing order")
    if a.white_noise and not (-120 <= a.white_noise[0] < a.white_noise[1] <= 0):
        ap.error("--white-noise takes two whole-number levels in dB below full scale, MIN_DB lower than "
                 "MAX_DB, both from -120 to 0")
    if a.extra_aug_prob is not None:
        if not (a.time_stretch or a.white_noise):
            ap.error("--extra-aug-prob needs --time-stretch or --white-noise; on its own it changes nothing")
        if not 0 < a.extra_aug_prob <= 1:
            ap.error("--extra-aug-prob must be above 0 and at most 1")
    if a.select == "last":
        if a.max_steps:
            ap.error("--select last requires complete epochs; do not combine it with --max-steps")
        if a.keep_stock:
            ap.error("--select last lets nothing be chosen on dev, including stock; add --no-keep-stock")
        if a.curriculum:
            ap.error("--select last is not combined with --curriculum")
        if a.epochs < a.average_top:
            ap.error(f"--select last averages the last --average-top {a.average_top} epochs; "
                     f"--epochs {a.epochs} is fewer")
    train_parts = None
    if a.train_parts is not None:
        if a.freeze_encoder_layers:
            ap.error("--train-parts and --freeze-encoder-layers contradict each other; use one")
        try:
            train_parts = parse_train_parts(a.train_parts)
        except ValueError as e:
            ap.error(str(e))
    if a.train_parts is None and (a.train_blocks != LOWER_BLOCKS or a.train_first_block != 0):
        ap.error("--train-blocks and --train-first-block only apply together with --train-parts")
    if a.train_blocks < 1 or a.train_first_block < 0 or a.train_first_block + a.train_blocks > 24:
        ap.error("--train-blocks must be at least 1 and --train-first-block + --train-blocks at most 24 "
                 "(Parakeet has 24 encoder blocks)")
    if (not math.isfinite(a.empty_margin) or a.empty_margin < 0 or not math.isfinite(a.empty_weight) or a.empty_weight < 0
            or a.empty_clips_per_kind < 0):
        ap.error("--empty-margin, --empty-weight and --empty-clips-per-kind cannot be negative")
    if (a.empty_margin > 0 or a.empty_clips_per_kind) and a.curriculum:
        ap.error("the silence loss and empty clips are not combined with --curriculum")
    if a.adapter_after_blocks is not None:
        if a.adapter_after_blocks < 1:
            ap.error("--adapter-after-blocks must be 1 or more")
        if a.train_parts is not None or a.freeze_encoder_layers:
            ap.error("--adapter-after-blocks trains only the adapter; do not combine it with "
                     "--train-parts or --freeze-encoder-layers")
        if a.freeze_transducer_head:
            ap.error("an adapter already freezes the transducer head; omit --freeze-transducer-head")
    manifests, inputs = validate_export(export, ("train", "dev"))
    train_rows = [r for r in manifests["train"].values() if float(r["duration_s"]) <= a.max_duration]
    dev_rows = list(manifests["dev"].values())
    n_over = len(manifests["train"]) - len(train_rows)
    if a.kinds:
        kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}
        unknown = kinds - {"sentence", "word"}
        if unknown:
            ap.error(f"--kinds takes sentence and/or word, not {sorted(unknown)}")
        missing = [r["id"] for r in train_rows + dev_rows if "kind" not in r]
        if missing:
            ap.error(f"--kinds needs a 'kind' field in every manifest row; {len(missing)} rows lack it")
        before = len(train_rows), len(dev_rows)
        train_rows = [r for r in train_rows if r["kind"] in kinds]
        dev_rows = [r for r in dev_rows if r["kind"] in kinds]
        log(f"--kinds {','.join(sorted(kinds))}: train {before[0]} -> {len(train_rows)} clips, "
            f"dev {before[1]} -> {len(dev_rows)} clips (dev is filtered too, so checkpoints are picked "
            "on the same kind of recording)")
        if not dev_rows:
            raise ValueError("--kinds left no dev clips to pick checkpoints on")
    if a.dev_speakers:
        keep = {x.strip() for x in a.dev_speakers.split(",") if x.strip()}
        before = len(dev_rows)
        dev_rows = [r for r in dev_rows if r.get("speaker") in keep]
        log(f"--dev-speakers {','.join(sorted(keep))}: dev {before} -> {len(dev_rows)} clips "
            "(checkpoints and the stock comparison are scored on these speakers only)")
        if not dev_rows:
            raise ValueError("--dev-speakers left no dev clips")
    if a.repeat_speakers:
        if a.repeat < 2:
            ap.error("--repeat must be 2 or more when --repeat-speakers is given")
        rep_spk = {x.strip() for x in a.repeat_speakers.split(",") if x.strip()}
        chosen = [r for r in train_rows if r.get("speaker") in rep_spk]
        if not chosen:
            raise ValueError(f"--repeat-speakers matched no training clips: {sorted(rep_spk)}")
        before = len(train_rows)
        train_rows = train_rows + [dict(r) for r in chosen for _ in range(a.repeat - 1)]
        log(f"--repeat-speakers {','.join(sorted(rep_spk))} x{a.repeat}: {len(chosen)} clips repeated, "
            f"train {before} -> {len(train_rows)} clip uses per epoch")
    good_rows = [r for r in train_rows if r.get("condition", "unknown") in GOOD]
    if not train_rows:
        raise ValueError("No training clips remain after --max-duration filtering")
    n_all, n_dev, n_good = len(train_rows), len(dev_rows), len(good_rows)
    n_empty = 0
    if a.empty_clips_per_kind:
        import battery
        n_empty = a.empty_clips_per_kind * len(battery.KINDS)
    unique_train = {str(r["id"]): r for r in train_rows}
    unique_secs = sum(float(r["duration_s"]) for r in unique_train.values())
    batches = math.ceil((n_all + n_empty) / a.batch)
    steps = math.ceil(batches / a.grad_accum)
    log_section(f"RUN {out.name}", {
        "Base model": a.base,
        "Data": export,
        "Training audio": f"{len(unique_train)} unique clips / {unique_secs / 60:.2f} min; {n_all} clip uses / epoch"
                          + (f"; plus {n_empty} empty-transcript noise clips (seed {a.empty_clips_seed})" if n_empty else ""),
        "Development": f"{n_dev} clips / {sum(float(r['duration_s']) for r in dev_rows) / 60:.2f} min",
        "Selection": f"last {a.average_top} epochs, fixed {a.epochs}-epoch schedule" if a.select == "last"
                     else f"best {a.average_top} dev checkpoints; patience {a.patience}; base guard {a.keep_stock}",
        "Batch": f"{a.batch} clips x {a.grad_accum} accumulation = up to {a.batch * a.grad_accum} clips / update",
        "Budget": f"~{batches} batches / epoch; ~{steps} updates / epoch; cap {a.max_steps or 'none'}",
        "Learning rate": f"{a.lr:.2e}; warmup {a.warmup_steps} updates; minimum {a.lr / 20:.2e}",
        "Seed / precision": f"{a.seed} / {a.precision}",
        "Augmentation": augmentation_text(a),
        "Excluded": f"{n_over} training clips over {a.max_duration:g}s",
        "Output": out,
    })
    if a.warmup_steps >= (a.max_steps or steps * a.epochs):
        log("WARNING: the update budget ends during warmup; the requested peak learning rate will not be reached")
    if a.check:
        log(f"preflight passed; train/dev input ID {record_id(inputs)} (test split was not opened)")
        return 0
    if out.exists():
        raise FileExistsError(f"Training output already exists; choose a new run name: {out}")

    try:
        import nemo.collections.asr as nemo_asr
        import torch
        from nemo.utils import logging as nemo_logging
        from omegaconf import open_dict

        try:
            import lightning.pytorch as pl
        except ImportError:
            import pytorch_lightning as pl
    except ImportError as e:
        log(
            f"missing dependency: {e}\n  install a CUDA-enabled torch build\n"
            '  python -m pip install -e ".[train]"   (use Linux or WSL2)'
        )
        return 1
    nemo_logging.set_verbosity(nemo_logging.INFO if a.verbose else nemo_logging.WARNING)
    pl.seed_everything(a.seed)
    if a.offload_optimizer:
        from nemo.core.optim import optimizers as nemo_optimizers
        nemo_optimizers.AVAILABLE_OPTIMIZERS["adamw"] = offloaded_adamw(torch)
        log("--offload-optimizer: AdamW's running averages and its update live in system RAM; "
            "the GPU holds the model, the gradients and the batch")
    if not torch.cuda.is_available() and not a.allow_cpu:
        log("no CUDA device -a 0.6 B fine-tune on CPU is not a realistic run; stopping (--allow-cpu for a plumbing check)")
        return 1
    if torch.cuda.is_available():
        frozen_for_disk = (24 if a.adapter_after_blocks else
                           24 - a.train_blocks if train_parts else a.freeze_encoder_layers)
        problems = gpu_environment_problems() + disk_space_problems(out, a.average_top, frozen_for_disk)
        if problems:
            log("training environment is not ready:\n  - " + "\n  - ".join(problems)
                + "\n  one-command fix: ./willspeak.sh train-setup   (docs/training.md §2)")
            return 1
        log("environment check passed (numba GPU kernel, resampy, disk space)")
        ram = visible_ram_gb()
        if ram is not None and ram < 14:
            log(f"WARNING: only {ram:.1f} GB of RAM + swap visible. Saving and averaging checkpoints "
                "peaks well above the 2.4 GB model; if the run ends in 'Killed', add swap in "
                "C:\\Users\\<you>\\.wslconfig (docs/training.md §2) and run `wsl --shutdown`")
    if a.precision.startswith("bf16") and not torch.cuda.is_available():
        a.precision = "32-true"
    if a.time_stretch or a.white_noise:
        problems = check_augmentor(build_augmentor(a))
        if problems:
            log("the extra augmentation cannot run here:\n  - " + "\n  - ".join(problems))
            return 1

    out.mkdir(parents=True, exist_ok=False)
    man_all = out / "train_all.json"
    man_dev = out / "dev.json"
    man_good = out / "train_good.json"
    target_form = None
    if a.target_form == "scoring":
        from willspeak.normalize import normalize as target_form
        log("transcripts in scoring form: lowercase, no punctuation (--target-form scoring)")
    elif a.target_form == "stock":
        from willspeak.normalize import stock_form as target_form
        log("transcripts in the model's own form: capital first letter, terminal mark (--target-form stock)")
    else:
        log("transcripts as written in the manifest (--target-form raw)")
    skipped = 0
    for rows, path in ((train_rows, man_all), (dev_rows, man_dev), (good_rows, man_good)):
        write_nemo_manifest(rows, export, path, target_form)
        skipped += write_nemo_manifest.skipped
    if skipped:
        log(f"WARNING: {skipped} manifest rows left out because their target was empty in this form")
    if n_empty:
        noise = battery.build(out / "empty-clips", a.empty_clips_per_kind, a.empty_clips_seed)
        with open(man_all, "a", encoding="utf-8") as f:
            for r in noise:
                f.write(json.dumps({"audio_filepath": str((out / "empty-clips" / r["audio"]).resolve()),
                                    "duration": float(r["duration_s"]), "text": "", "condition": r["condition"]}) + "\n")
        log(f"{len(noise)} empty-transcript noise clips written to {out / 'empty-clips'} and added to every epoch")
    run = {
        "format": "willspeak-training-v1",
        "status": "running",
        "started_at": timestamp(),
        "configuration": vars(a),
        "inputs": inputs,
        "input_id": record_id(inputs),
        "packages": versions(),
        "script": file_record(Path(__file__)),
        "train_clips": n_all,
        "unique_train_clips": len(unique_train),
        "unique_train_minutes": unique_secs / 60,
        "dev_clips": n_dev,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    t0 = time.monotonic()
    if Path(a.base).is_file():
        run["base_file"] = file_record(Path(a.base))
    write_json(out / "run.json", run)
    _active_run = (run, out)
    _wandb = wandb_start(run, a, out)
    log_section("ENVIRONMENT", {
        "Device": run["device"],
        "Torch / CUDA": f"{torch.__version__} / {torch.version.cuda or 'none'}",
        "Precision": a.precision,
        "Optimizer": "AdamW in system RAM" if a.offload_optimizer else "AdamW on the training device",
        "Details": "verbose" if a.verbose else "concise; add --verbose for NeMo details and dev examples",
    })

    if a.dev_scoring == "evaluate":
        from willspeak.normalize import normalize as score_form
        score_dev_like_evaluate(score_form)
    log(f"loading {a.base}")
    if Path(a.base).is_file():
        model = nemo_asr.models.ASRModel.restore_from(a.base)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(a.base)
    if getattr(model, "wer", None) is not None:
        model.wer.log_prediction = a.verbose
    adapter = None
    base_id = None
    if a.adapter_after_blocks:
        import block_adapter
        adapter = block_adapter.add_adapter(model, a.adapter_after_blocks)
        block_adapter.freeze_all_but_adapter(model, adapter)
        frozen = len(model.encoder.layers)
        what = (f"only the adapter after encoder blocks 0-{a.adapter_after_blocks - 1} trainable; "
                "everything else frozen")
        base_id = run.get("base_file") or {"name": a.base}
        if a.lr <= 5e-5:
            log(f"WARNING: --lr {a.lr} is a block-training rate; an adapter that starts at zero learned at 3e-4 "
                "in the research, and at this rate it will barely move")
    elif train_parts:
        opened = train_lower_parts(model, train_parts, a.train_blocks, a.train_first_block)
        frozen = len(getattr(model.encoder, "layers", [])) - opened
        what = (f"only {', '.join(train_parts)} in encoder blocks {a.train_first_block}-{a.train_first_block + opened - 1} "
                "trainable; rest of encoder frozen")
    else:
        frozen = freeze_lower_encoder(model, a.freeze_encoder_layers)
        what = f"{frozen} encoder layers + front end frozen"
    if a.freeze_transducer_head:
        head_weights = freeze_transducer_head(model)
        what += "; prediction network and joint frozen"
        run["frozen_transducer_head_parameters"] = head_weights
        log(f"transducer head frozen: {head_weights:,} parameters; encoder gradients remain enabled")
    spec = getattr(model, "spec_augmentation", None)
    spec_cfg = model.cfg.get("spec_augment") or {}
    spec_settings = {k: spec_cfg[k] for k in ("freq_masks", "freq_width", "time_masks", "time_width")
                     if k in spec_cfg}
    if spec is None:
        log("SpecAugment: the base model has none")
    elif a.spec_augment:
        log(f"SpecAugment on, base model's settings {spec_settings} (training clips only; dev is never masked)")
    else:
        model.spec_augmentation = None
        log("SpecAugment off (--no-spec-augment): training clips are speed-perturbed only")
    run["spec_augment"] = spec_settings if (spec is not None and a.spec_augment) else None
    run["augmentation"] = build_augmentor(a)
    save_only_what_changes(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"parameters: {total / 1e6:.0f} M total, {trainable / 1e6:.0f} M trainable ({what})")
    log(f"parameters exact: {total} total, {trainable} trainable")
    run["parameters"] = {"total": total, "trainable": trainable, "frozen_encoder_layers": frozen,
                         "train_parts": list(train_parts) if train_parts else None,
                         "train_blocks": a.train_blocks if train_parts else None,
                         "train_first_block": a.train_first_block if train_parts else None,
                         "adapter_after_blocks": a.adapter_after_blocks}
    run["target_form"] = a.target_form
    check_targets = target_checker(model)
    if check_targets is not None:
        texts = []
        for path in (man_all, man_dev):
            with open(path, encoding="utf-8") as f:
                texts.extend(json.loads(line)["text"] for line in f)
        unspellable = check_targets(texts)
        if unspellable:
            log(f"WARNING: {len(unspellable)} of {len(texts)} targets contain a character the tokenizer has no "
                f"piece for (trained as <unk>), e.g. {unspellable[:3]!r}")
        else:
            log(f"targets check: all {len(texts)} train and dev targets are spelled by the tokenizer")
    if a.dev_scoring == "evaluate":
        check_dev_scorer(model)
        log("dev WER scored like evaluate.py: output and transcript normalised (--dev-scoring evaluate)")
    else:
        log("dev WER scored by NeMo's raw comparison, capitals and punctuation count (--dev-scoring nemo)")
    if a.freeze_batchnorm:
        log("batch-norm statistics stay at the base model's values (--freeze-batchnorm)")
    else:
        log("batch-norm statistics update from every training batch (--no-freeze-batchnorm)")
    if a.empty_margin > 0 or n_empty:
        import margin_loss

        def silence_report(stats: dict) -> None:
            log("silence loss: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()))

        margin_loss.install(model, a.empty_margin, a.empty_weight, report=silence_report)
        run["silence_loss"] = {"margin_nats": a.empty_margin, "weight": a.empty_weight, "empty_clips": n_empty,
                               "empty_clips_seed": a.empty_clips_seed, "reduction": model.loss.reduction,
                               "empty_target_length_for_reduction": 1}
        log(f"silence-aware loss in place: margin {a.empty_margin:g} nats x weight {a.empty_weight:g}, {n_empty} empty-transcript clips per epoch")
        log(f"silence loss reduction: {model.loss.reduction}; empty targets count as one sequence in the reduction")
    write_json(out / "run.json", run)

    def configure(manifest: Path, epochs: int):
        cfg = model.cfg
        with open_dict(cfg):
            cfg.train_ds.manifest_filepath = str(manifest)
            cfg.train_ds.batch_size = a.batch
            cfg.train_ds.shuffle = True
            cfg.train_ds.max_duration = a.max_duration
            cfg.train_ds.is_tarred = False
            cfg.train_ds.num_workers = 4 if torch.cuda.is_available() else 0
            cfg.train_ds.use_lhotse = False
            cfg.train_ds.augmentor = build_augmentor(a)
            cfg.validation_ds.manifest_filepath = str(man_dev)
            cfg.validation_ds.batch_size = a.batch
            cfg.validation_ds.shuffle = False
            cfg.validation_ds.num_workers = 2 if torch.cuda.is_available() else 0
            cfg.validation_ds.use_lhotse = False
            cfg.validation_ds.max_duration = None
            cfg.optim = {
                "name": "adamw",
                "lr": a.lr,
                "betas": [0.9, 0.98],
                "weight_decay": a.weight_decay,
                "sched": {"name": "CosineAnnealing", "warmup_steps": a.warmup_steps, "min_lr": a.lr / 20},
            }
        model.setup_training_data(cfg.train_ds)
        model.setup_validation_data(cfg.validation_ds)
        loader = model.train_dataloader()
        actual_batches = len(loader)
        if actual_batches < 1:
            raise ValueError(f"NeMo loaded no training batches from {manifest}")
        actual_steps = math.ceil(actual_batches / a.grad_accum)
        budget = {
            "clips": len(loader.dataset), "batches_per_epoch": actual_batches,
            "updates_per_epoch": actual_steps,
            "maximum_updates": min(actual_steps * epochs, a.max_steps) if a.max_steps else actual_steps * epochs,
            "drop_last": loader.drop_last,
        }
        run.setdefault("phase_budgets", {})[manifest.stem] = budget
        write_json(out / "run.json", run)
        log_section(f"TRAINING PLAN: {manifest.stem}", {
            "Loaded": f"{budget['clips']} clip uses; {actual_batches} batches / epoch; drop_last={loader.drop_last}",
            "Updates": f"{actual_steps} / full epoch; up to {budget['maximum_updates']} in this phase",
            "Warmup": f"{a.warmup_steps} updates",
        })
        if a.warmup_steps >= budget["maximum_updates"]:
            log("WARNING: the loaded update budget ends during warmup")
        if a.select == "last":
            ckpt = pl.callbacks.ModelCheckpoint(
                dirpath=str(out / "ckpt" / manifest.stem),
                monitor="epoch",
                mode="max",
                save_top_k=max(1, a.average_top),
                save_weights_only=True,
                filename="{epoch}",
            )
            callbacks = [ckpt]
        else:
            ckpt = pl.callbacks.ModelCheckpoint(
                dirpath=str(out / "ckpt" / manifest.stem),
                monitor="val_wer",
                mode="min",
                save_top_k=max(1, a.average_top),
                save_weights_only=True,
                filename="best-{epoch}-{val_wer:.3f}",
            )
            callbacks = [ckpt, pl.callbacks.EarlyStopping(monitor="val_wer", mode="min", patience=a.patience)]
        callbacks.append(curve_recorder(pl, run, out, torch, phase=manifest.stem))
        if a.freeze_batchnorm:
            callbacks.append(batchnorm_freezer(pl, torch))
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=epochs,
            max_steps=a.max_steps or -1,
            precision=a.precision,
            accumulate_grad_batches=a.grad_accum,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            val_check_interval=1.0,
            log_every_n_steps=10,
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
        )
        model.set_trainer(trainer)
        model.setup_optimization(cfg.optim)
        return trainer, ckpt

    stock_dev = None

    def score_stock(trainer) -> None:
        nonlocal stock_dev
        if not a.keep_stock or stock_dev is not None:
            return
        trainer.validate(model, verbose=False)
        value = trainer.callback_metrics.get("val_wer")
        if value is None:
            raise RuntimeError("validation logged no val_wer; cannot score the stock model")
        stock_dev = float(value)
        if not math.isfinite(stock_dev):
            raise RuntimeError("Base validation produced non-finite dev WER")
        run["stock_dev_wer"] = stock_dev
        write_json(out / "run.json", run)
        log(f"stock dev WER {stock_dev:.4f} ({stock_dev * 100:.2f}%; base model; a checkpoint must beat this to be released)")

    if a.curriculum and n_good and n_good < n_all:
        log(f"phase 1/2: {n_good} good/typical-day clips for {a.curriculum_epochs} epochs")
        trainer, _ = configure(man_good, a.curriculum_epochs)
        score_stock(trainer)
        trainer.fit(model)
        log(f"phase 2/2: all {n_all} clips (bad days included) for {a.epochs} epochs")
    elif a.curriculum:
        log("curriculum requested but every session is 'good/typical' or none is -single phase")
    trainer, ckpt = configure(man_all, a.epochs)
    score_stock(trainer)
    trainer.fit(model)
    run["stop_reason"] = (f"step limit ({a.max_steps})" if a.max_steps and trainer.global_step >= a.max_steps
                          else f"early stopping (patience {a.patience})" if trainer.should_stop
                          else f"epoch limit ({a.epochs})")
    write_json(out / "run.json", run)
    if {name: file_record(export / name) for name in inputs} != inputs:
        raise ValueError("Training inputs changed during the run; no deployable checkpoint will be saved")
    if "base_file" in run and file_record(Path(a.base)) != run["base_file"]:
        raise ValueError("Base model file changed during the run; no deployable checkpoint will be saved")
    best = ckpt.best_model_path
    if a.select == "last":
        kept = sorted(ckpt.best_k_models, key=lambda k: float(ckpt.best_k_models[k]))
        log(f"fixed schedule (--select last): kept epochs "
            f"{', '.join(str(1 + int(float(ckpt.best_k_models[k]))) for k in kept)}; dev chose nothing")
    else:
        log(f"best dev WER {ckpt.best_model_score} at {best}")
    if not best or ckpt.best_model_score is None:
        raise RuntimeError("No checkpoint was saved; refusing to release the final epoch")
    final = out / ("adapter.pt" if adapter is not None else "parakeet-ft.nemo")
    if stock_dev is not None and float(ckpt.best_model_score) >= stock_dev:
        log(f"no checkpoint beat stock on dev ({float(ckpt.best_model_score):.4f} against {stock_dev:.4f}); "
            "releasing the stock model unchanged")
        if adapter is not None:
            block_adapter.save_adapter(final, block_adapter.AdapterLayer(adapter.linear.in_features),
                                       a.adapter_after_blocks, base_id)
            run.update(status="complete", finished_at=timestamp(), selected="stock",
                       best_dev_wer=float(ckpt.best_model_score), averaged_checkpoints=[],
                       released_dev_wer=stock_dev,
                       minutes=round((time.monotonic() - t0) / 60, 1), checkpoint=file_record(final))
            write_json(out / "run.json", run)
            release_summary(run, final, time.monotonic() - t0)
            wandb_close(run, out, "complete")
            return 0
        import gc

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        if Path(a.base).is_file():
            stock_model = nemo_asr.models.ASRModel.restore_from(a.base)
        else:
            stock_model = nemo_asr.models.ASRModel.from_pretrained(a.base)
        stock_model.save_to(str(final))
        run.update(status="complete", finished_at=timestamp(), selected="stock",
                   best_dev_wer=float(ckpt.best_model_score), averaged_checkpoints=[],
                   released_dev_wer=stock_dev,
                   minutes=round((time.monotonic() - t0) / 60, 1), checkpoint=file_record(final))
        write_json(out / "run.json", run)
        release_summary(run, final, time.monotonic() - t0)
        wandb_close(run, out, "complete")
        return 0
    import gc

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    state = torch.load(best, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    load_trained_weights(model, state)
    del state
    gc.collect()
    averaged = average_checkpoints(model, ckpt, a.average_top, torch, below=stock_dev)
    released_dev = float(ckpt.best_model_score) if a.select == "dev" and len(averaged) <= 1 else None
    if len(averaged) > 1:
        log("Checking the averaged model on dev before release...")
        checker = pl.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             precision=a.precision, logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False)
        model.set_trainer(checker)
        checker.validate(model, verbose=False)
        value = checker.callback_metrics.get("val_wer")
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError("Averaged model validation produced no finite dev WER")
        released_dev = float(value)
        if a.select == "dev" and stock_dev is not None and released_dev >= stock_dev:
            log(f"Averaged dev WER {released_dev * 100:.2f}% did not beat base {stock_dev * 100:.2f}%; "
                "releasing the best individual checkpoint instead")
            state = torch.load(best, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
            load_trained_weights(model, state)
            del state
            averaged = [best]
            released_dev = float(ckpt.best_model_score)
        del checker
    if adapter is not None:
        block_adapter.save_adapter(final, adapter, a.adapter_after_blocks, base_id)
    else:
        model.save_to(str(final))
    run.update(
        status="complete",
        selected="fine-tuned" if a.select == "dev" else "fine-tuned, fixed schedule",
        finished_at=timestamp(),
        best_dev_wer=float(ckpt.best_model_score) if a.select == "dev" else None,
        averaged_checkpoints=averaged,
        released_dev_wer=released_dev,
        minutes=round((time.monotonic() - t0) / 60, 1),
        checkpoint=file_record(final),
    )
    write_json(out / "run.json", run)
    release_summary(run, final, time.monotonic() - t0)
    wandb_close(run, out, "complete")
    if adapter is not None:
        source = ["--nemo", a.base] if Path(a.base).is_file() else (
            ["--stock"] if a.base == BASE else ["--pretrained", a.base])
        command = shlex.join(["python", "scripts/asr/nemo_hyp.py", *source, "--adapter", str(final),
                              "--manifest", str(export / "test.jsonl"), "--audio-root", str(export),
                              "--out", f"reports/{out.name}-test-nemo.jsonl"])
        log(f"saved {final}\n\nnext (from app/):\n  {command}")
        return 0
    commands = [
        ["python", "scripts/asr/export.py", "--nemo", str(final)],
        ["python", "scripts/asr/evaluate.py", "--manifest", str(export / "test.jsonl"),
         "--audio-root", str(export), "--model-dir", str(out / "onnx-int8"), "--out", f"reports/{out.name}"],
    ]
    log(f"saved {final}\n\nnext (from app/):\n  " + "\n  ".join(shlex.join(cmd) for cmd in commands))
    return 0


def run_training() -> int:
    try:
        return main()
    except (Exception, KeyboardInterrupt) as exc:
        if _active_run is not None:
            run, out = _active_run
            run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       finished_at=timestamp(), error=f"{type(exc).__name__}: {exc}")
            write_json(out / "run.json", run)
            wandb_close(run, out, run["status"])
        raise


if __name__ == "__main__":
    sys.exit(_cli.run(run_training, "train_parakeet",
                      console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
