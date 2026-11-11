#!/usr/bin/env python3
"""Train and test the recording-quality gate used by the Record page.

From app/, in the training environment (needs torch):

  python scripts/quality/gate.py fetch       download the noise, room and speech corpora (about 4 GB)
  python scripts/quality/gate.py prepare     file lists with train / val / test splits
  python scripts/quality/gate.py train       train the network on mixtures made on the fly
  python scripts/quality/gate.py evaluate    tests A-C on held-out data, thresholds from val
  python scripts/quality/gate.py export      write src/willspeak/models/quality_gate.npz, check NumPy = PyTorch

TORGO comes from `./willspeak.sh torgo` (data/public/torgo). Everything else goes
to data/public/quality. Splits never share speakers, noise recordings or rooms,
and the test rooms are real recordings while training uses simulated rooms.
See docs/design.md, **Recording quality gate**.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(APP / "src"))
sys.path.insert(0, str(APP / "scripts" / "corpus"))

DATA = APP / "data" / "public" / "quality"
TORGO = APP / "data" / "public" / "torgo" / "TORGO"
MAN = DATA / "manifests"
RUNS = DATA / "runs"
NPZ = APP / "src" / "willspeak" / "models" / "quality_gate.npz"

# What counts as a truly fine recording; fixed before training.
TRUTH = {"snr_min_db": 15.0, "c50_min_db": 8.0, "spk2_max": 0.5}

TORGO_SEVERITY = {"F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
                  "M05": "moderate", "F03": "mild", "F04": "mild", "M03": "mild"}
TORGO_SPLIT = {"F01": "train", "F03": "train", "M02": "train", "M03": "train", "M05": "train",
               "M01": "val", "M04": "test", "F04": "test"}
DEMAND = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
          "PCAFETER", "PRESTO", "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"]
DEMAND_SPLIT = {"DKITCHEN": "val", "PSTATION": "val",
                "DLIVING": "test", "PCAFETER": "test", "TCAR": "test", "OOFFICE": "test"}
# Sounds the speaker makes are not background noise.
ESC_EXCLUDE = {"breathing", "coughing", "sneezing", "snoring", "laughing"}

DOWNLOADS = [
    ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "dev-clean.tar.gz"),
    ("https://www.openslr.org/resources/12/test-clean.tar.gz", "test-clean.tar.gz"),
    ("https://openslr.elda.org/resources/28/rirs_noises.zip", "rirs_noises.zip"),
] + [(f"https://zenodo.org/records/1227121/files/{e}_16k.zip", f"demand/{e}_16k.zip") for e in DEMAND]


# ---------------------------------------------------------------- fetch

def fetch(_args) -> None:
    from fetch_torgo import download

    raw = DATA / "downloads"
    for url, name in DOWNLOADS:
        download(url, raw / name)
    libri = DATA / "libri"
    for name in ("dev-clean.tar.gz", "test-clean.tar.gz"):
        if not (libri / "LibriSpeech" / name.split(".")[0]).exists():
            with tarfile.open(raw / name) as tf:
                tf.extractall(libri, filter="data")
    if not (DATA / "RIRS_NOISES").exists():
        with zipfile.ZipFile(raw / "rirs_noises.zip") as z:
            z.extractall(DATA)
    for e in DEMAND:
        if not (DATA / "demand" / e).exists():
            with zipfile.ZipFile(raw / "demand" / f"{e}_16k.zip") as z:
                z.extractall(DATA / "demand")
    from huggingface_hub import snapshot_download
    snapshot_download("ashraq/esc50", repo_type="dataset", local_dir=str(DATA / "esc50"))
    print("done; next: python scripts/quality/gate.py prepare")


# ---------------------------------------------------------------- prepare

def _hsplit(key: str, val: float = 0.15, test: float = 0.15) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def _base_snr(path: str) -> tuple[float | None, float]:
    """Speech level minus the quietest-10% frame level: how clean the recording already is."""
    import soundfile as sf

    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError):  # a few TORGO files are unreadable
        return None, 0.0
    x = x[:, 0]
    if sr != 16000 or len(x) < 320:
        return None, len(x) / max(sr, 1)
    n = 1 + (len(x) - 320) // 160
    p = np.mean(x[np.arange(320)[None, :] + 160 * np.arange(n)[:, None]] ** 2, axis=1) + 1e-12
    pdb = 10 * np.log10(p)
    active = pdb >= max(pdb.max() - 35, -70)
    return float(10 * np.log10(p[active].mean()) - np.percentile(pdb, 10)), len(x) / sr


def prepare(_args) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    speech = []
    for wav in sorted(glob.glob(str(TORGO / "*" / "Session*" / "wav_headMic" / "*.wav"))):
        spk = Path(wav).parts[-4]
        if spk not in TORGO_SPLIT:
            continue
        snr, dur = _base_snr(wav)
        if snr is not None and dur >= 0.5:
            speech.append({"path": wav, "speaker": spk, "group": TORGO_SEVERITY[spk], "corpus": "torgo",
                           "split": TORGO_SPLIT[spk], "base_snr": round(snr, 1)})
    for sub in ("dev-clean", "test-clean"):
        for flac in sorted(glob.glob(str(DATA / "libri" / "LibriSpeech" / sub / "*" / "*" / "*.flac"))):
            spk = Path(flac).parts[-3]
            split = "test" if sub == "test-clean" else ("val" if _hsplit("libri" + spk, 0.2, 0.0) == "val" else "train")
            speech.append({"path": flac, "speaker": "L" + spk, "group": "typical", "corpus": "libri", "split": split})

    noise = []
    for env in DEMAND:
        for ch in sorted(glob.glob(str(DATA / "demand" / env / "ch*.wav"))):
            noise.append({"path": ch, "source": "demand", "kind": env, "split": DEMAND_SPLIT.get(env, "train")})
    esc_out = DATA / "esc50_16k"
    esc_out.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(str(DATA / "esc50" / "data" / "*.parquet"))):
        for r in pq.read_table(f, columns=["filename", "fold", "category", "audio"]).to_pylist():
            if r["category"] in ESC_EXCLUDE:
                continue
            dst = esc_out / r["filename"]
            if not dst.exists():
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                x = resample_poly(x[:, 0], 160, sr // 100).astype(np.float32) if sr != 16000 else x[:, 0]
                nz = np.nonzero(np.abs(x) > 1e-4)[0]
                if len(nz) < 1600:
                    continue
                sf.write(dst, x[nz[0]:nz[-1] + 1], 16000)
            split = "test" if r["fold"] == 5 else "val" if r["fold"] == 4 else "train"
            noise.append({"path": str(dst), "source": "esc50", "kind": r["category"], "split": split})

    rr = DATA / "RIRS_NOISES"
    for w in sorted(glob.glob(str(rr / "pointsource_noises" / "*.wav"))):
        noise.append({"path": w, "source": "pointsource", "kind": "pointsource", "split": _hsplit(Path(w).name)})
    rirs = []
    for w in sorted(glob.glob(str(rr / "real_rirs_isotropic_noises" / "*.wav"))):
        if "noise" in Path(w).name:
            noise.append({"path": w, "source": "isotropic", "kind": "isotropic", "split": "test"})
        else:
            rirs.append({"path": w, "source": "real", "room": Path(w).name, "split": "test"})
    for w in sorted(glob.glob(str(rr / "simulated_rirs" / "*" / "Room*" / "*.wav"))):
        room = "/".join(Path(w).parts[-3:-1])
        rirs.append({"path": w, "source": "simulated", "room": room,
                     "split": "val" if _hsplit(room, 0.1, 0.0) == "val" else "train"})

    MAN.mkdir(parents=True, exist_ok=True)
    for name, items in (("speech", speech), ("noise", noise), ("rirs", rirs)):
        (MAN / f"{name}.json").write_text(json.dumps(items))
        counts: dict = {}
        for it in items:
            k = f'{it["split"]} {it.get("group") or it.get("source")}'
            counts[k] = counts.get(k, 0) + 1
        print(name, len(items), dict(sorted(counts.items())))


# ---------------------------------------------------------------- train

def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from mix import Mixer, collate
    from net import QualityNet, count_params
    from torch.utils.data import DataLoader, IterableDataset

    class Stream(IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng(args.seed + 1000 * (info.id if info else 0))
            mixer = Mixer(MAN, "train")
            while True:
                yield mixer(rng)

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model = QualityNet()
    print("trainable parameters:", count_params(model), flush=True)
    mixer = Mixer(MAN, "val")
    rng = np.random.default_rng(12345)
    val = [mixer(rng) for _ in range(1200)]

    dl = DataLoader(Stream(), batch_size=args.bs, collate_fn=collate, num_workers=args.workers,
                    persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    best, log, run, t0 = float("inf"), [], np.zeros(3), time.time()
    for step, (wav, lens, snr, c50, spk2, _) in enumerate(dl, start=1):
        p_snr, p_c50, p_spk = model(wav, lens)
        parts = (F.smooth_l1_loss(p_snr / 10, snr / 10), F.smooth_l1_loss(p_c50 / 10, c50 / 10),
                 F.binary_cross_entropy_with_logits(p_spk, spk2))
        opt.zero_grad()
        sum(parts).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * np.array([p.item() for p in parts])
        if step % 100 == 0:
            print(f"step {step} loss snr {run[0]:.3f} c50 {run[1]:.3f} spk2 {run[2]:.3f} "
                  f"({(time.time() - t0) / step:.2f} s/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            pred = _run(model, [w for w, _ in val])
            model.train()
            lab = [lab for _, lab in val]
            snr_mae = float(np.mean(np.abs(pred[:, 0] - [lab_["snr"] for lab_ in lab])))
            c50_mae = float(np.mean(np.abs(pred[:, 1] - [lab_["c50"] for lab_ in lab])))
            spk_acc = float(np.mean((pred[:, 2] > 0.5) == (np.array([lab_["spk2"] for lab_ in lab]) > 0.5)))
            score = snr_mae / 10 + c50_mae / 10 + (1 - spk_acc) * 3
            log.append({"step": step, "snr_mae": snr_mae, "c50_mae": c50_mae, "spk2_acc": spk_acc, "score": score})
            print("VAL", json.dumps(log[-1]), flush=True)
            if score < best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")
            (out / "val_log.json").write_text(json.dumps(log, indent=1))
        if step >= args.steps:
            break


# ---------------------------------------------------------------- evaluate

def _run(model, wavs, bs: int = 16) -> np.ndarray:
    import torch
    from mix import collate

    rows = []
    with torch.no_grad():
        for i in range(0, len(wavs), bs):
            wav, lens, *_ = collate([(w, {"snr": 0, "c50": 0, "spk2": 0}) for w in wavs[i:i + bs]])
            s, c, k = model(wav, lens)
            rows += list(zip(s.tolist(), c.tolist(), torch.sigmoid(k).tolist()))
    return np.array(rows)


def _decide(pred, th) -> np.ndarray:
    return (pred[:, 0] >= th["snr_min_db"]) & (pred[:, 1] >= th["c50_min_db"]) & (pred[:, 2] <= th["spk2_max"])


def _auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = (y == 1).sum(), (y != 1).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_ci(v, n: int = 2000) -> list[float]:
    v = np.asarray(v, float)
    rng = np.random.default_rng(0)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _load(name: str):
    import torch
    from net import QualityNet

    model = QualityNet()
    model.load_state_dict(torch.load(RUNS / name / "best.pt"))
    return model.eval()


def evaluate(args) -> None:
    from mix import Mixer, read

    from willspeak import audio

    model = _load(args.name)

    # Thresholds from the validation split: 97.5% of truly fine val clips pass each check.
    mixer, rng = Mixer(MAN, "val"), np.random.default_rng(4242)
    items = [mixer(rng) for _ in range(4000)]
    pred = _run(model, [w for w, _ in items])
    snr = np.array([lab["snr"] for _, lab in items])
    c50 = np.array([lab["c50"] for _, lab in items])
    spk = np.array([lab["spk2"] for _, lab in items])
    th = {"snr_min_db": float(min(TRUTH["snr_min_db"], np.percentile(pred[snr >= TRUTH["snr_min_db"], 0], 2.5))),
          "c50_min_db": float(min(TRUTH["c50_min_db"], np.percentile(pred[c50 >= TRUTH["c50_min_db"], 1], 2.5))),
          "spk2_max": float(max(TRUTH["spk2_max"], np.percentile(pred[spk == 0, 2], 97.5)))}
    res: dict = {"thresholds": th}
    print("thresholds", th, flush=True)

    # A. Held-out mixtures in real recorded rooms.
    mixer, rng = Mixer(MAN, "test", rir_source="real"), np.random.default_rng(777)
    items = [mixer(rng) for _ in range(args.n)]
    pred = _run(model, [w for w, _ in items])
    lab = [lab_ for _, lab_ in items]
    snr = np.array([x["snr"] for x in lab])
    c50 = np.array([x["c50"] for x in lab])
    spk = np.array([x["spk2"] for x in lab])
    grp = np.array([x["group"] for x in lab])
    good = (snr >= TRUTH["snr_min_db"]) & (c50 >= TRUTH["c50_min_db"]) & (spk == 0)
    keep = _decide(pred, th)
    res["A_mixtures"] = {}
    for g in ("all", "severe", "mild", "typical"):
        sel = np.ones(len(lab), bool) if g == "all" else grp == g
        noisy, rev = sel & (snr < 40), sel & (c50 < 35)
        res["A_mixtures"][g] = {
            "n": int(sel.sum()),
            "snr_mae_noisy": float(np.mean(np.abs(pred[noisy, 0] - snr[noisy]))),
            "c50_mae_echoed": float(np.mean(np.abs(pred[rev, 1] - c50[rev]))),
            "second_voice_auroc": _auroc(spk[sel], pred[sel, 2]),
            "kept_when_fine": float(keep[sel & good].mean()),
            "rejected_when_bad": float((~keep[sel & ~good]).mean()),
        }
    print(json.dumps(res["A_mixtures"], indent=1), flush=True)

    # B. Untouched held-out recordings: the gate should keep severe speech as often as typical speech.
    speech = [s for s in json.loads((MAN / "speech.json").read_text()) if s["split"] == "test"]
    res["B_untouched"] = {}
    for g in ("severe", "mild", "typical"):
        sel = [s for s in speech if s["group"] == g]
        wavs = [read(s["path"]) for s in sel]
        k = _decide(_run(model, wavs), th)
        levels_ok = np.array([audio.analyze(w, 16000, 16000, 1).ok for w in wavs])
        res["B_untouched"][g] = {"n": len(sel), "gate_keeps": float(k.mean()), "gate_keeps_ci95": _boot_ci(k),
                                 "level_checks_pass": float(levels_ok.mean())}
    print(json.dumps(res["B_untouched"], indent=1), flush=True)

    # C. TORGO's far array microphone against its head microphone, same utterances.
    res["C_far_mic"] = {}
    for spk_dir in sorted(TORGO.glob("[FM]0[0-9]")):
        heads, arrays = [], []
        for h in sorted(spk_dir.glob("Session*/wav_headMic/*.wav")):
            a = Path(str(h).replace("wav_headMic", "wav_arrayMic"))
            if not a.exists():
                continue
            try:
                xh, xa = read(str(h)), read(str(a))
            except (OSError, RuntimeError):  # unreadable TORGO file
                continue
            if min(len(xh), len(xa)) >= 8000:
                heads.append(xh)
                arrays.append(xa)
        if not heads:
            continue
        ph, pa = _run(model, heads), _run(model, arrays)
        res["C_far_mic"][spk_dir.name] = {
            "pairs": len(heads), "head_snr_median": float(np.median(ph[:, 0])),
            "array_snr_median": float(np.median(pa[:, 0])),
            "array_noisier": float(np.mean(pa[:, 0] < ph[:, 0])), "array_more_echo": float(np.mean(pa[:, 1] < ph[:, 1])),
            "head_kept": float(_decide(ph, th).mean()), "array_kept": float(_decide(pa, th).mean())}
    print(json.dumps(res["C_far_mic"], indent=1), flush=True)
    (RUNS / args.name / "test_results.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- export

def export(args) -> None:
    import torch
    from net import mel_filterbank

    from willspeak.quality import QualityGate

    model = _load(args.name)
    th = json.loads((RUNS / args.name / "test_results.json").read_text())["thresholds"]
    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    out = {"mel_fb": mel_filterbank()}
    for i, conv in enumerate(model.convs):
        out[f"conv{i}.weight"], out[f"conv{i}.bias"] = sd[f"convs.{i}.weight"], sd[f"convs.{i}.bias"]
        out[f"conv{i}.dilation"] = np.array(conv.dilation[0], dtype=np.float32)
        for p in ("weight", "bias", "running_mean", "running_var"):
            out[f"norm{i}.{p}"] = sd[f"norms.{i}.{p}"]
    out["head1.weight"], out["head1.bias"] = sd["head.0.weight"], sd["head.0.bias"]
    out["head2.weight"], out["head2.bias"] = sd["head.2.weight"], sd["head.2.bias"]
    for k in ("snr_min_db", "c50_min_db", "spk2_max"):
        out[k] = np.array(th[k], dtype=np.float32)
    dst = Path(args.out)
    np.savez_compressed(dst, **{k: np.asarray(v, dtype=np.float32) for k, v in out.items()})

    gate, rng, worst = QualityGate(dst), np.random.default_rng(0), 0.0
    for n in (8000, 32000, 160000):
        x = (rng.standard_normal(n) * 0.05).astype(np.float32)
        with torch.no_grad():
            ref = [t.item() for t in model(torch.from_numpy(x)[None], torch.tensor([n]))]
        got = gate.estimate(x)
        worst = max(worst, abs(got[0] - ref[0]), abs(got[1] - ref[1]), abs(got[2] - 1 / (1 + math.exp(-ref[2]))))
    print(f"wrote {dst} ({os.path.getsize(dst) // 1024} KB); largest NumPy/PyTorch difference {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("NumPy and PyTorch disagree; do not use this file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=32000)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name", default="gate")
    e = sub.add_parser("evaluate")
    e.add_argument("--name", default="gate")
    e.add_argument("--n", type=int, default=3000)
    x = sub.add_parser("export")
    x.add_argument("--name", default="gate")
    x.add_argument("--out", default=str(NPZ))
    args = ap.parse_args()
    {"fetch": fetch, "prepare": prepare, "train": train, "evaluate": evaluate, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
