#!/usr/bin/env python3
"""Embed labelled audio for projection fitting and matcher evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli


def rows_from_csv(path: Path) -> list[dict]:
    base = path.parent
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["path"])
            out.append({"path": str(p if p.is_absolute() else base / p), "label": r["label"],
                        "session": r.get("session") or "s0", "speaker": r.get("speaker") or "spk"})
    return out


def rows_from_torgo(root: Path) -> list[dict]:
    out = []
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sess_dir in sorted(spk_dir.glob("Session*")):
            prompts = {p.stem: p.read_text(encoding="utf-8", errors="replace").strip()
                       for p in (sess_dir / "prompts").glob("*.txt")} if (sess_dir / "prompts").exists() else {}
            for wav_dir in sorted(sess_dir.glob("wav_*")):
                mic = wav_dir.name.replace("wav_", "")
                for wav in sorted(wav_dir.glob("*.wav")):
                    text = prompts.get(wav.stem, "")
                    if not text or text.startswith("[") or len(text.split()) != 1:
                        continue
                    lab = re.sub(r"[^a-z']+", "", text.lower())
                    if not lab:
                        continue
                    out.append({"path": str(wav), "label": lab, "speaker": spk_dir.name,
                                "session": f"{spk_dir.name}_{sess_dir.name}_{mic}"})
    return out


def rows_from_corpus(data_dir: Path) -> list[dict]:
    from willspeak.db import Store
    from willspeak.matcher.engine import norm_label
    store = Store(data_dir)
    out = []
    for r in store.con.execute(
            """SELECT c.id, c.path, COALESCE(c.transcript, c.prompt_text) label, c.session_id, p.prompt_set
               FROM clip c LEFT JOIN prompt p ON p.id = c.prompt_id WHERE c.status = 'accepted' ORDER BY c.id"""):
        lab = norm_label(r["label"])
        if not lab or len(lab.split()) > 6 or r["prompt_set"] in ("phonetic", "probe"):
            continue
        out.append({"path": str(data_dir / str(r["path"]).replace("\\", "/")), "label": lab,
                    "session": f"s{r['session_id']}", "speaker": "own"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV: path,label,session,speaker")
    src.add_argument("--torgo", help="TORGO root directory")
    src.add_argument("--corpus", action="store_true", help="this project's data/ corpus")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--limit", type=int, default=0, help="debug: embed only the first N")
    a = ap.parse_args()

    if a.manifest:
        rows = rows_from_csv(Path(a.manifest))
    elif a.torgo:
        rows = rows_from_torgo(Path(a.torgo))
    else:
        rows = rows_from_corpus(Path(a.data))
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("nothing to embed"); return 1
    print(f"{len(rows)} clips · {len({r['label'] for r in rows})} labels · "
          f"{len({r['speaker'] for r in rows})} speakers · {len({r['session'] for r in rows})} sessions")

    from willspeak import audio as A
    from willspeak.matcher.engine import DIM, LAYER, MODEL_ID, POOL, Embedder
    if not Embedder.available():
        print("torch/transformers are not installed in THIS Python:\n  python -m pip install -r requirements.txt")
        return 1
    import soundfile as sf
    emb = Embedder()
    raw, keep = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            x, sr = sf.read(r["path"], dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {r['path']}: {e}"); continue
        try:
            raw.append(emb(A.to_mono_16k(x, sr))); keep.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {r['path']}: {e}"); continue
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  ({(i + 1) / (time.time() - t0):.1f} clips/s)")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, raw=np.stack(raw).astype(np.float32),
             labels=np.array([r["label"] for r in keep]), sessions=np.array([r["session"] for r in keep]),
             speakers=np.array([r["speaker"] for r in keep]), paths=np.array([r["path"] for r in keep]),
             meta=json.dumps({"model": MODEL_ID, "layer": LAYER, "pool": POOL, "dim": DIM,
                              "source": a.manifest or a.torgo or "corpus", "n": len(keep),
                              "created": time.strftime("%Y-%m-%d")}))
    print(f"wrote {out}  [{len(keep)} x {DIM}]")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "embed_manifest"))
