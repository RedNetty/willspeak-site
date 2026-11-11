#!/usr/bin/env python3
"""Upload runs that already exist on disk to Weights & Biases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402

KINDS = ("runs", "logs", "evals", "tables")
TABLE_NAME = re.compile(r"^(table.*|score|screen|bounds.*)\.json$")
SPEAKER = re.compile(r"(?<![A-Za-z0-9])(FC?0\d|MC?0\d)(?![0-9])")
TEXT_KEYS = {"system", "name", "label", "speaker", "arm", "model", "baseline", "split", "unit"}
SKIP_KEYS = {"provenance", "confusions", "worst", "hyp", "ref", "text", "transcript", "hypotheses"}


ID_VERSION = "v2"


def run_id_for(kind: str, relpath: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return hashlib.sha1(f"{ID_VERSION}\0{kind}\0{relpath}\0{digest}".encode()).hexdigest()[:32]


def flat_numbers(value, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    out = {} if out is None else out
    if depth > 6 or len(out) > 400:
        return out
    if isinstance(value, bool):
        out[prefix] = int(value)
    elif isinstance(value, (int, float)):
        if math.isfinite(value):
            out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SKIP_KEYS:
                continue
            flat_numbers(item, f"{prefix}.{key}" if prefix else str(key), out, depth + 1)
    elif isinstance(value, list) and len(value) <= 64:
        for i, item in enumerate(value):
            flat_numbers(item, f"{prefix}.{i}" if prefix else str(i), out, depth + 1)
    return out


def flat_row(row: dict) -> dict:
    out = flat_numbers(row)
    for key, value in row.items():
        if key in TEXT_KEYS and isinstance(value, str) and len(value) <= 80:
            out[key] = value
    return out


def seconds_from_text(text: str) -> float | None:
    match = re.fullmatch(r"\s*(?:(\d+)h\s*)?(\d+)m\s*(\d+)s\s*", text or "")
    if not match:
        return None
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def percent_from_text(text: str) -> float | None:
    match = re.search(r"([\d.]+)%", text or "")
    return float(match.group(1)) / 100 if match else None


def iso_to_epoch(text: str | None) -> float | None:
    try:
        return datetime.fromisoformat(text).timestamp() if text else None
    except ValueError:
        return None


def stamp_from_name(name: str) -> float | None:
    match = re.search(r"(\d{8})-(\d{6})", name)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()


FIELD = re.compile(r"^\[[\d:]+\] \[train\]   (.+?)\s+: (.*)$")
SECTION = re.compile(r"^\[[\d:]+\] \[train\] --- (.+?) ---$")
EPOCH = re.compile(r"^\[[\d:]+\] \[train\] Epoch (\d+)/(\d+) \| dev WER ([\d.]+)%")
EPOCH_DETAIL = re.compile(r"loss (\S+) \| lr (\S+) \| updates (\d+) this epoch / (\d+) total \| "
                          r"train ([^|]+?) \| dev (.+?)\s*$")
GPU = re.compile(r"peak GPU allocated ([\d.]+) GB \| reserved ([\d.]+) GB")
STOCK = re.compile(r"stock dev WER ([\d.]+) ")


def _number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(text: str) -> dict:
    lines = text.splitlines()
    result = {"argv": [], "out": None, "name": None, "sections": {}, "epochs": [], "complete": {},
              "stock_dev_wer": None, "status": "failed", "error": None}
    if lines and lines[0].startswith("# train_parakeet argv="):
        try:
            result["argv"] = ast.literal_eval(lines[0].split("argv=", 1)[1].rsplit(" cwd=", 1)[0])
        except (ValueError, SyntaxError):
            pass
    argv = result["argv"]
    if "--out" in argv and argv.index("--out") + 1 < len(argv):
        result["out"] = argv[argv.index("--out") + 1].rstrip("/\\")
    section = None
    traceback_at = None
    for index, line in enumerate(lines):
        match = SECTION.match(line)
        if match:
            section = match.group(1)
            result["sections"].setdefault(section, {})
            if section.startswith("RUN "):
                result["name"] = section[4:]
            continue
        match = FIELD.match(line)
        if match and section:
            result["sections"][section][match.group(1).strip()] = match.group(2).strip()
            continue
        match = EPOCH.match(line)
        if match:
            epoch = {"epoch": int(match.group(1)), "of": int(match.group(2)),
                     "dev_wer": float(match.group(3)) / 100}
            for follow in lines[index + 1:index + 4]:
                detail = EPOCH_DETAIL.search(follow)
                if detail:
                    epoch.update(train_loss=_number(detail.group(1)), learning_rate=_number(detail.group(2)),
                                 epoch_optimizer_steps=int(detail.group(3)), optimizer_steps=int(detail.group(4)),
                                 train_seconds=seconds_from_text(detail.group(5)),
                                 validation_seconds=seconds_from_text(detail.group(6)))
                gpu = GPU.search(follow)
                if gpu:
                    epoch.update(peak_gpu_allocated_gb=float(gpu.group(1)), peak_gpu_gb=float(gpu.group(2)))
            result["epochs"].append(epoch)
            continue
        match = STOCK.search(line)
        if match and "stock dev WER" in line:
            result["stock_dev_wer"] = float(match.group(1))
        if line.startswith("Traceback") and traceback_at is None:
            traceback_at = index
    if "COMPLETE" in result["sections"]:
        result["complete"] = result["sections"]["COMPLETE"]
        result["status"] = "complete"
    elif any("KeyboardInterrupt" in line for line in lines[-40:]):
        result["status"] = "interrupted"
    if traceback_at is not None and result["status"] != "complete":
        tail = [ln.strip() for ln in lines[traceback_at:] if ln.strip()]
        result["error"] = tail[-1][:300] if tail else "Traceback"
        if "KeyboardInterrupt" in (result["error"] or ""):
            result["status"] = "interrupted"
    return result


def log_summary(parsed: dict) -> dict:
    complete = parsed["complete"]
    elapsed = seconds_from_text(complete.get("Elapsed", ""))
    epochs = parsed["epochs"]
    updates = complete.get("Optimizer updates")
    return {
        "released_dev_wer": percent_from_text(complete.get("Released model dev WER", "")),
        "best_dev_wer": percent_from_text(complete.get("Best checkpoint dev WER", "")),
        "base_dev_wer": percent_from_text(complete.get("Base dev WER", "")) or parsed["stock_dev_wer"],
        "completed_epochs": int(complete["Completed epochs"]) if complete.get("Completed epochs", "").isdigit()
        else len(epochs),
        "optimizer_updates": int(updates) if (updates or "").isdigit()
        else (epochs[-1].get("optimizer_steps") if epochs else 0),
        "elapsed_seconds": elapsed,
        "selected": complete.get("Released"),
        "stop_reason": complete.get("Training stopped"),
        "error": parsed["error"],
    }


class Item:
    def __init__(self, kind, name, source, steps, run_id, upload, when=None):
        self.kind, self.name, self.source, self.steps = kind, name, source, steps
        self.run_id, self.upload, self.when = run_id, upload, when


def rel(path: Path) -> str:
    return path.resolve().relative_to(APP).as_posix()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def replay_epochs(handle, epochs: list[dict]) -> None:
    for number, row in enumerate(epochs, start=1):
        step = row.get("optimizer_steps")
        if step is not None and row.get("train_loss") is not None:
            wandb_log.log(handle, {"optimizer_step": step, "train/loss": row["train_loss"],
                                   "train/lr": row.get("learning_rate")}, step_metric="optimizer_step")
        wandb_log.log(handle, {
            "epoch": number,
            "epoch/dev_wer": row.get("dev_wer"),
            "epoch/dev_wer_pct": None if row.get("dev_wer") is None else row["dev_wer"] * 100,
            "epoch/train_loss": row.get("train_loss"),
            "epoch/learning_rate": row.get("learning_rate"),
            "epoch/optimizer_steps": step,
            "epoch/train_seconds": row.get("train_seconds"),
            "epoch/validation_seconds": row.get("validation_seconds"),
            "epoch/peak_gpu_gb": row.get("peak_gpu_gb"),
        }, step_metric="epoch")


def _without_none(values: dict) -> dict:
    return {k: v for k, v in values.items() if v is not None}


def collect_runs(args) -> tuple[list[Item], list[dict]]:
    items, covered = [], []
    for path in sorted((APP / "models").glob("**/run.json")):
        try:
            content = read_bytes(path)
            record = json.loads(content)
        except (OSError, ValueError):
            print(f"skipped unreadable {rel(path)}")
            continue
        if not isinstance(record, dict) or record.get("format") != "willspeak-training-v1":
            continue
        config = record.get("configuration") or {}
        out = str(config.get("out", path.parent.name)).rstrip("/\\")
        covered.append({"out": out, "started": iso_to_epoch(record.get("started_at"))})
        name = path.parent.name
        curve = record.get("curve") or []

        def upload(item, record=record, config=config, name=name, curve=curve, path=path):
            settings = dict(config)
            settings.update(
                original_started_at=record.get("started_at"), original_finished_at=record.get("finished_at"),
                device=record.get("device"), cuda=record.get("cuda"), train_clips=record.get("train_clips"),
                dev_clips=record.get("dev_clips"), unique_train_minutes=record.get("unique_train_minutes"),
                parameters=record.get("parameters"), source=rel(path))
            base = str(config.get("base", "")).split("/")[-1] or "unknown-base"
            handle = wandb_log.start(name, settings, group=args.group, tags=["backfill", "train", base],
                                     job_type="train", run_id=item.run_id, project=args.project)
            if handle is None:
                return False
            replay_epochs(handle, curve)
            released = record.get("released_dev_wer")
            wandb_log.summary(handle, _without_none({
                "released_dev_wer": released,
                "released_dev_wer_pct": None if released is None else released * 100,
                "best_dev_wer": record.get("best_dev_wer"), "base_dev_wer": record.get("stock_dev_wer"),
                "completed_epochs": len(curve),
                "optimizer_updates": curve[-1].get("optimizer_steps") if curve else 0,
                "elapsed_seconds": None if record.get("minutes") is None else record["minutes"] * 60,
                "trainable_parameters": (record.get("parameters") or {}).get("trainable"),
                "total_parameters": (record.get("parameters") or {}).get("total"),
                "selected": record.get("selected"), "stop_reason": record.get("stop_reason"),
                "error": record.get("error")}))
            wandb_log.log_file(handle, path, kind="train-record")
            wandb_log.log_file(handle, path.parent / "curve.csv", kind="train-record")
            status = record.get("status", "complete")
            wandb_log.finish(handle, status if status in {"complete", "failed", "interrupted"} else "failed")
            return True

        items.append(Item("runs", name, rel(path), len(curve), run_id_for("runs", rel(path), content),
                          upload, iso_to_epoch(record.get("started_at"))))
    return items, covered


def collect_logs(args, covered: list[dict]) -> list[Item]:
    items = []
    for path in sorted((APP / "logs").glob("train_parakeet-*.log")):
        try:
            content = read_bytes(path)
            parsed = parse_log(content.decode("utf-8", errors="replace"))
        except OSError:
            print(f"skipped unreadable {rel(path)}")
            continue
        if not parsed["name"] and not parsed["argv"]:
            continue
        when = stamp_from_name(path.name)
        out = parsed["out"] or parsed["name"]
        if any(c["out"] == out and c["started"] and when and abs(c["started"] - when) < 180 for c in covered):
            continue
        name = parsed["name"] or Path(out).name

        def upload(item, parsed=parsed, name=name, path=path, when=when):
            header = {k: v for sec, fields in parsed["sections"].items() if sec.startswith("RUN ")
                      for k, v in fields.items()}
            settings = {"argv": parsed["argv"], "log_header": header, "source": rel(path),
                        "log_started_at": None if when is None else
                        datetime.fromtimestamp(when, timezone.utc).isoformat()}
            base = header.get("Base model", "unknown-base").split("/")[-1]
            handle = wandb_log.start(name, settings, group=args.group,
                                     tags=["backfill", "from-log", "train", base],
                                     job_type="train", run_id=item.run_id, project=args.project)
            if handle is None:
                return False
            replay_epochs(handle, parsed["epochs"])
            wandb_log.summary(handle, _without_none(log_summary(parsed)))
            wandb_log.finish(handle, parsed["status"])
            return True

        items.append(Item("logs", name, rel(path), len(parsed["epochs"]), run_id_for("logs", rel(path), content),
                          upload, when))
    return items


def group_for(path: Path) -> str:
    match = SPEAKER.search(path.parent.name)
    return match.group(1) if match else path.parent.parent.name


def collect_evals(args) -> list[Item]:
    items = []
    for path in sorted((APP / "reports").glob("**/results.json")):
        try:
            content = read_bytes(path)
            results = json.loads(content)
        except (OSError, ValueError):
            print(f"skipped unreadable {rel(path)}")
            continue
        if not isinstance(results, dict) or "wer" not in results:
            continue
        name = path.parent.name

        def upload(item, results=results, name=name, path=path):
            extra = {"source": rel(path), "test_name": name}
            handle = wandb_log.start(name, extra, group=args.group or group_for(path),
                                     tags=["backfill", "eval", path.parent.parent.name], job_type="eval",
                                     run_id=item.run_id, project=args.project)
            if handle is None:
                return False
            metrics = wandb_log.eval_metrics(results)
            ci = path.parent / "ci.json"
            if ci.is_file():
                try:
                    metrics.update({f"ci/{k}": v for k, v in flat_numbers(json.loads(ci.read_text())).items()})
                except (OSError, ValueError):
                    pass
            wandb_log.summary(handle, metrics)
            wandb_log.log(handle, metrics)
            wandb_log.finish(handle)
            return True

        items.append(Item("evals", name, rel(path), 1, run_id_for("evals", rel(path), content), upload,
                          path.stat().st_mtime))
    return items


def collect_tables(args) -> list[Item]:
    items = []
    for path in sorted((APP / "reports").glob("lab-*/**/*.json")):
        if not TABLE_NAME.match(path.name):
            continue
        try:
            content = read_bytes(path)
            data = json.loads(content)
        except (OSError, ValueError):
            print(f"skipped unreadable {rel(path)}")
            continue
        rows = data.get("rows") if isinstance(data, dict) else None
        rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        lab = path.relative_to(APP / "reports").parts[0]
        name = f"{lab}/{path.stem}" if path.parent.name == lab else f"{lab}/{path.parent.name}/{path.stem}"

        def upload(item, data=data, rows=rows, lab=lab, name=name, path=path):
            top = {k: v for k, v in data.items() if k not in {"rows", "provenance"}} if isinstance(data, dict) else {}
            settings = {"source": rel(path), "lab_test": lab, **flat_row(top)}
            handle = wandb_log.start(name, settings, group=args.group or lab, tags=["backfill", "lab-table", lab],
                                     job_type="lab-table", run_id=item.run_id, project=args.project)
            if handle is None:
                return False
            numbers = {}
            if rows:
                flat = [flat_row(r) for r in rows]
                columns = list(dict.fromkeys(k for r in flat for k in r))
                wandb_log.log_table(handle, "systems", columns, [[r.get(c) for c in columns] for r in flat])
                for i, r in enumerate(flat):
                    label = r.get("system") or r.get("name") or r.get("label") or str(i)
                    for key in ("pooled_wer", "pooled_wrong", "pooled_words"):
                        if key in r:
                            numbers[f"{label}/{key}"] = r[key]
                wers = [r["pooled_wer"] for r in flat if isinstance(r.get("pooled_wer"), (int, float))]
                if wers:
                    numbers["best_pooled_wer"] = min(wers)
                numbers["systems"] = len(rows)
            else:
                numbers = flat_numbers(top if top else data)
            wandb_log.summary(handle, numbers)
            wandb_log.log_file(handle, path, kind="lab-table")
            wandb_log.finish(handle)
            return True

        items.append(Item("tables", name, rel(path), max(len(rows), 1), run_id_for("tables", rel(path), content),
                          upload, path.stat().st_mtime))
    return items


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all", help="comma list of runs,logs,evals,tables, or all")
    ap.add_argument("--mode", choices=("online", "offline"), default="online")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded; upload nothing")
    ap.add_argument("--project", default=None, help="wandb project (default: WANDB_PROJECT or willspeak-torgo)")
    ap.add_argument("--group", default=None, help="override the group of every run")
    ap.add_argument("--limit", type=int, default=None, help="at most N runs of each kind")
    ap.add_argument("--replace", action="store_true", help="delete a run that is already on wandb and send it again")
    ap.add_argument("--since", default=None, metavar="YYYYMMDD", help="skip sources older than this date")
    return ap.parse_args(argv)


def build_items(args) -> list[Item]:
    wanted = set(KINDS) if args.what == "all" else {w.strip() for w in args.what.split(",") if w.strip()}
    unknown = wanted - set(KINDS)
    if unknown:
        raise SystemExit(f"unknown --what value(s): {', '.join(sorted(unknown))}")
    run_items, covered = collect_runs(args)
    items = []
    if "runs" in wanted:
        items += run_items
    if "logs" in wanted:
        items += collect_logs(args, covered)
    if "evals" in wanted:
        items += collect_evals(args)
    if "tables" in wanted:
        items += collect_tables(args)
    if args.since:
        cutoff = datetime.strptime(args.since, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
        items = [i for i in items if i.when is None or i.when >= cutoff]
    if args.limit is not None:
        kept, seen = [], {}
        for item in items:
            seen[item.kind] = seen.get(item.kind, 0) + 1
            if seen[item.kind] <= args.limit:
                kept.append(item)
        items = kept
    return items


def existing_runs(args) -> dict:
    try:
        import wandb
        api = wandb.Api(timeout=60)
        entity = os.environ.get("WANDB_ENTITY") or api.default_entity
        project = args.project or os.environ.get("WANDB_PROJECT") or wandb_log.DEFAULT_PROJECT
        return {r.id: r for r in api.runs(f"{entity}/{project}")}
    except Exception as exc:  # noqa: BLE001
        print(f"could not list existing runs ({type(exc).__name__}); sending everything")
        return {}


def main(argv=None) -> int:
    args = parse_args(argv)
    items = build_items(args)
    if args.dry_run:
        print(f"{'kind':<7} {'name':<48} {'steps':>5}  source")
        for item in items:
            print(f"{item.kind:<7} {item.name[:48]:<48} {item.steps:>5}  {item.source}")
        counts = {k: sum(1 for i in items if i.kind == k) for k in KINDS}
        print("dry run, nothing uploaded: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
        return 0
    if not wandb_log.available():
        print("wandb is not installed here: python -m pip install wandb")
        return 2
    if args.mode == "online" and not wandb_log._has_credentials():
        print("No wandb login found. Run `wandb login` once (or set WANDB_API_KEY), "
              "or use --mode offline and `wandb sync` the runs later.")
        return 2
    os.environ["WILLSPEAK_WANDB"] = args.mode
    existing = existing_runs(args) if args.mode == "online" else {}
    done = failed = skipped = 0
    for item in items:
        have = existing.get(item.run_id)
        if have is not None and not args.replace:
            skipped += 1
            print(f"skip {item.kind:<7} {item.name}   (already on wandb; --replace to send again)", flush=True)
            continue
        if have is not None:
            try:
                have.delete(delete_artifacts=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {item.kind:<7} {item.name}   (could not remove the old copy: {type(exc).__name__})", flush=True)
                failed += 1
                continue
        ok = item.upload(item)
        done, failed = done + bool(ok), failed + (not ok)
        print(f"{'ok  ' if ok else 'FAIL'} {item.kind:<7} {item.name}", flush=True)
    print(f"{args.mode}: {done} runs sent, {skipped} already there, {failed} failed of {len(items)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
