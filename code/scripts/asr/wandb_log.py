from __future__ import annotations

import json
import math
import netrc
import os
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = "willspeak-torgo"
MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".json", ".csv", ".txt"}
INIT_TIMEOUT_S = 60

_warned: set[str] = set()


def _warn(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"[wandb] {message}", flush=True)


def _import():
    try:
        import wandb
        return wandb
    except Exception:  # noqa: BLE001
        return None


def available() -> bool:
    return _import() is not None


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return "api.wandb.ai" in netrc.netrc().hosts
    except Exception:  # noqa: BLE001
        return False


def mode() -> str:
    choice = os.environ.get("WILLSPEAK_WANDB", "auto").strip().lower() or "auto"
    if choice == "off" or choice not in {"auto", "online", "offline"}:
        if choice not in {"auto", "online", "offline", "off"}:
            _warn(f"unknown WILLSPEAK_WANDB={choice!r}; tracking is off")
        return "off"
    if not available():
        return "off"
    if choice == "offline":
        return "offline"
    if choice == "online":
        return "online" if _has_credentials() else "offline"
    if _has_credentials():
        return "online"
    return "offline" if os.environ.get("WANDB_MODE", "").lower() == "offline" else "off"


class Handle:
    def __init__(self, run, name: str, run_mode: str):
        self.run, self.name, self.mode = run, name, run_mode
        self.defined: set[str] = set()
        self.closed = False


def _clean(value, depth: int = 0):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth < 4 and isinstance(value, dict):
        return {str(k): _clean(v, depth + 1) for k, v in value.items()}
    if depth < 4 and isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in value]
    return str(value)


def _finite(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            out[key] = value
    return out


def _tags(extra) -> list[str]:
    tags = [t.strip() for t in os.environ.get("WILLSPEAK_WANDB_TAGS", "").split(",") if t.strip()]
    tags += [str(t) for t in (extra or [])]
    return list(dict.fromkeys(tags))[:20]


def start(name, config=None, *, group=None, tags=None, job_type="train", notes=None,
          run_id=None, project=None, dir=None):
    run_mode = mode()
    if run_mode == "off":
        return None
    try:
        wandb = _import()
        directory = Path(dir or os.environ.get("WANDB_DIR") or APP / "logs")
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(directory))
        for key, value in (("WANDB_SILENT", "true"), ("WANDB_CONSOLE", "off"),
                           ("WANDB_DISABLE_GIT", "true"), ("WANDB_DISABLE_CODE", "true")):
            os.environ.setdefault(key, value)
        kwargs = dict(
            project=project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT,
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=str(name), config=_clean(config or {}), group=group or os.environ.get("WILLSPEAK_WANDB_GROUP") or None,
            tags=_tags(tags), job_type=job_type, notes=notes, dir=str(directory),
            save_code=False, reinit=True,
        )
        if run_id:
            kwargs.update(id=str(run_id), resume="allow")
        try:
            settings = wandb.Settings(init_timeout=INIT_TIMEOUT_S, console="off", silent=True)
        except Exception:  # noqa: BLE001
            settings = None
        run = None
        if run_mode == "online":
            try:
                run = wandb.init(mode="online", settings=settings, **kwargs)
            except Exception as exc:  # noqa: BLE001
                _warn(f"online start failed ({type(exc).__name__}); using offline mode")
                run_mode = "offline"
        if run is None:
            run = wandb.init(mode="offline", settings=settings, **kwargs)
        handle = Handle(run, str(name), run_mode)
        run.define_metric("optimizer_step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("epoch/*", step_metric="epoch")
        handle.defined.update({"optimizer_step", "epoch"})
        return handle
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not start a run ({type(exc).__name__}: {exc}); tracking is off")
        return None


def log(handle, metrics: dict, step_metric: str | None = None) -> None:
    if handle is None or handle.closed:
        return
    try:
        values = _finite(metrics)
        if step_metric:
            for key in values:
                if key != step_metric and key not in handle.defined:
                    handle.run.define_metric(key, step_metric=step_metric)
                    handle.defined.add(key)
        handle.run.log(values)
    except Exception as exc:  # noqa: BLE001
        _warn(f"log failed ({type(exc).__name__}: {exc})")


def summary(handle, values: dict) -> None:
    if handle is None or handle.closed:
        return
    try:
        for key, value in values.items():
            if value is None:
                continue
            handle.run.summary[str(key)] = _clean(value)
    except Exception as exc:  # noqa: BLE001
        _warn(f"summary failed ({type(exc).__name__}: {exc})")


def log_table(handle, name, columns, rows) -> None:
    if handle is None or handle.closed:
        return
    try:
        wandb = _import()
        table = wandb.Table(columns=[str(c) for c in columns],
                            data=[[_clean(v) for v in row] for row in rows])
        handle.run.log({str(name): table})
    except Exception as exc:  # noqa: BLE001
        _warn(f"table failed ({type(exc).__name__}: {exc})")


def log_file(handle, path, *, kind="record") -> None:
    if handle is None or handle.closed:
        return
    try:
        path = Path(path)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            _warn(f"refused to upload {path.name}: only .json, .csv and .txt records are allowed")
            return
        if not path.is_file() or path.is_symlink():
            return
        if path.stat().st_size > MAX_FILE_BYTES:
            _warn(f"refused to upload {path.name}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
            return
        wandb = _import()
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{handle.name}-{path.stem}")[:100]
        artifact = wandb.Artifact(label, type=re.sub(r"[^A-Za-z0-9_.-]+", "-", str(kind)) or "record")
        artifact.add_file(str(path), name=path.name)
        handle.run.log_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        _warn(f"file upload failed ({type(exc).__name__}: {exc})")


def finish(handle, status="complete") -> None:
    if handle is None or handle.closed:
        return
    try:
        handle.closed = True
        handle.run.summary["status"] = str(status)
        handle.run.finish(exit_code=0 if status == "complete" else 1)
    except Exception as exc:  # noqa: BLE001
        _warn(f"finish failed ({type(exc).__name__}: {exc})")


def _counts(prefix: str, block) -> dict:
    out = {}
    if isinstance(block, dict):
        for key in ("errors", "n_ref", "sub", "dele", "ins", "rate"):
            if isinstance(block.get(key), (int, float)):
                out[f"{prefix}/{key}"] = block[key]
    return out


def eval_metrics(results: dict) -> dict:
    out = {}
    for key in ("wer", "cer", "per", "n_utts"):
        if isinstance(results.get(key), (int, float)):
            out[f"eval/{key}"] = results[key]
    for unit in ("word", "char", "phone"):
        out.update(_counts(f"eval/{unit}", results.get(unit)))
    for kind, block in (results.get("by_group") or {}).items():
        out.update(_counts(f"eval/kind/{kind}", block))
    return out


def log_eval(name, results_json_path, *, group=None, tags=None, config_extra=None, run_id=None):
    handle = start(name, config_extra, group=group, tags=["eval", *(tags or [])],
                   job_type="eval", run_id=run_id)
    if handle is None:
        return None
    try:
        results = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
        metrics = eval_metrics(results)
        summary(handle, metrics)
        log(handle, metrics)
        finish(handle, "complete")
    except Exception as exc:  # noqa: BLE001
        _warn(f"eval record not logged ({type(exc).__name__}: {exc})")
        finish(handle, "failed")
    return handle
