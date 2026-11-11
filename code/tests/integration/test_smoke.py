#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.helpers import fake_wav


def run_smoke() -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="willspeak-smoke-"))
    import os

    os.environ["WILLSPEAK_DATA"] = str(tmp)
    # The uploads below are synthetic tones, not speech; the recording-quality
    # gate is tested in tests/unit/test_quality_gate.py.
    os.environ.setdefault("WILLSPEAK_QUALITY_GATE", "0")
    for m in list(sys.modules):
        if m.startswith("willspeak"):
            del sys.modules[m]

    from fastapi.testclient import TestClient

    from willspeak.server import app, store

    c = TestClient(app)
    fails = []

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    print("\n[0] g2p suffix fallback (OOV possessives must keep their sibilant)")
    from willspeak.g2p import word_phones

    tb = word_phones("toothbrush's")
    check("sibilant stem -> IH Z", tb is not None and tb[-2:] == ["IH", "Z"])
    rc = word_phones("recliner's")
    check("voiced stem -> Z", rc is not None and rc[-1] == "Z")
    check("true OOV stays None", word_phones("willspeak's") is None)

    print("\n[1] prompt bank")
    c.get("/api/state").raise_for_status()
    nxt = c.get("/api/next?n=12").json()["prompts"]
    check(f"{len(nxt)} prompts queued", len(nxt) == 12)
    check("critical set is first", nxt[0]["prompt_set"] == "critical")
    check("phones resolved", len(nxt[0]["phones"]) > 1)

    print("\n[2] session + upload + QC")
    sid = c.post("/api/session/start", data={"label": "smoke", "condition": "good_day"}).json()["session_id"]
    check("session started", sid > 0)

    clip_ids = []
    for i, p in enumerate(nxt[:10]):
        r = c.post(
            "/api/clip",
            files={"audio": ("c.wav", fake_wav(), "audio/wav")},
            data={"session_id": sid, "prompt_text": p["text"], "prompt_id": p["prompt_id"]},
        ).json()
        clip_ids.append(r["clip_id"])
    check("10 clips stored", len(clip_ids) == 10)
    q = c.get("/api/clips?status=pending").json()["clips"]
    check("clips pending review", len(q) == 10)

    r = c.post(
        "/api/clip",
        files={"audio": ("c.wav", fake_wav(clip=True), "audio/wav")},
        data={"session_id": sid, "prompt_text": "clipped take"},
    ).json()
    check("clipping detected", any("Clipping" in w for w in r["qc"]["warnings"]))

    r = c.post(
        "/api/clip",
        files={"audio": ("c.wav", fake_wav(seconds=0.1), "audio/wav")},
        data={"session_id": sid, "prompt_text": "too short"},
    ).json()
    check("too-short rejected", not r["qc"]["ok"])

    r = c.post(
        "/api/clip",
        files={"audio": ("c.wav", fake_wav(level=0.0004), "audio/wav")},
        data={"session_id": sid, "prompt_text": "too quiet"},
    ).json()
    check("too-quiet rejected", not r["qc"]["ok"])

    phon = c.get("/api/next?n=8&prompt_set=phonetic").json()["prompts"]
    phon_ids = []
    for p in phon:
        r = c.post(
            "/api/clip",
            files={"audio": ("c.wav", fake_wav(seconds=2.2), "audio/wav")},
            data={"session_id": sid, "prompt_text": p["text"], "prompt_id": p["prompt_id"]},
        ).json()
        phon_ids.append(r["clip_id"])
    check("8 phonetic clips stored", len(phon_ids) == 8)

    print("\n[3] review + correction loop")
    for cid, p_ in zip(clip_ids[:8], nxt[:8]):
        c.post(f"/api/clip/{cid}/review", data={"transcript": p_["text"], "status": "accepted"})
    for cid, p_ in zip(phon_ids, phon):
        c.post(f"/api/clip/{cid}/review", data={"transcript": p_["text"], "status": "accepted"})
    c.post(f"/api/clip/{clip_ids[8]}/review", data={"transcript": "yeah okay", "status": "accepted"})
    c.post(f"/api/clip/{clip_ids[9]}/review", data={"transcript": "", "status": "rejected"})
    t = c.get("/api/state").json()["totals"]
    check(f"17 accepted (got {t['accepted']})", t["accepted"] == 17)
    check(f"corrections logged (got {t['corrections']})", t["corrections"] >= 1)

    print("\n[4] coverage")
    cov = c.get("/api/coverage").json()
    check("coverage advanced", cov["diphones_covered"] > 0)
    check("phone counts complete", len(cov["phone_counts"]) == 39)
    check("recommendations present", len(cov["recommendations"]) > 0)

    print("\n[5] export + split hygiene")
    ex = c.post("/api/export", data={"dev_frac": 0.2, "test_frac": 0.2}).json()
    check(f"exported {sum(ex['counts'].values())} clips", sum(ex["counts"].values()) == 17)
    out = Path(ex["out_dir"])
    texts = {}
    for sp in ("train", "dev", "test"):
        for line in (out / f"{sp}.jsonl").read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                texts.setdefault(m["prompt_text"], set()).add(sp)
    leaked = [t for t, s in texts.items() if len(s) > 1]
    check(f"no prompt leaks across splits ({len(texts)} prompts)", not leaked)
    check("corrections exported", (out / "corrections.jsonl").exists())
    check("dev split filled (top-up)", ex["counts"]["dev"] >= 2)
    check("test split filled (top-up)", ex["counts"]["test"] >= 2)
    check(
        "critical clips all in train",
        all(
            json.loads(l)["prompt_set"] != "critical"
            for sp in ("dev", "test")
            for l in (out / f"{sp}.jsonl").read_text().splitlines()
            if l.strip()
        ),
    )
    first = json.loads((Path(tmp) / "split_assignments.json").read_text())
    ex2 = c.post("/api/export", data={"dev_frac": 0.2, "test_frac": 0.2}).json()
    second = json.loads((Path(tmp) / "split_assignments.json").read_text())
    check("sticky assignments survive re-export", first == second)
    check("re-export counts identical", ex2["counts"] == ex["counts"])

    print("\n[5b] backup")
    r = c.post("/api/backup").json()
    check("backup created", r["ok"] and r["bytes"] > 1000)
    import zipfile

    names = zipfile.ZipFile(Path(tmp) / "backups" / r["file"]).namelist()
    check("backup has database", "corpus.db" in names)
    check("backup has all audio incl. rejected", sum(1 for n in names if n.startswith("clips/")) == 21)
    st2 = c.get("/api/state").json()
    check("state reports backup", st2["backup"]["count"] >= 1 and st2["data_dir"])

    print("\n[6] eval harness")
    from willspeak import report as R
    from willspeak.evaluate import evaluate, phoneme_difficulty

    pairs = [
        {"id": 1, "ref": "I want a tomato sandwich", "hyp": "I want a tornado sandwich", "group": "functional"},
        {"id": 2, "ref": "I'm in pain", "hyp": "I am in pain", "group": "critical"},
        {
            "id": 3,
            "ref": "The birch canoe slid on the smooth planks",
            "hyp": "the birch canoe slid on the smooth plants",
            "group": "phonetic",
        },
        {"id": 4, "ref": "help", "hyp": "kelp", "group": "critical"},
        {"id": 5, "ref": "Turn the television on", "hyp": "", "group": "functional"},
    ]
    res = evaluate(pairs)
    check(f"WER computed ({res.wer * 100:.1f}%)", 0 < res.wer < 1)
    check("contraction normalized (I'm == I am)", res.by_group["critical"]["rate"] < 0.6)
    check("per-phoneme breakdown", len(res.per_phoneme) > 5)
    check("confusions found", len(res.confusions) > 0)
    diff = phoneme_difficulty(res, min_n=1)
    check(f"difficulty weights ({len(diff)} phones)", len(diff) > 0)
    check("weights in range", all(1.0 <= v <= 5.0 for v in diff.values()))

    rp = R.build(
        res,
        {"label": "smoke", "model": "synthetic", "date": "-", "split": "test", "difficulty": diff},
        tmp / "report.html",
    )
    html = rp.read_text()
    check(f"report written ({len(html) // 1024} KB)", len(html) > 4000)
    check("report is self-contained", "http://" not in html and "cdn" not in html.lower())

    print("\n[7] scheduler responds to difficulty")
    from willspeak.scheduler import Scheduler

    base = Scheduler(store).next_batch(8, prompt_set="phonetic")
    weighted = Scheduler(store, difficulty={"NG": 5.0, "ZH": 5.0}).next_batch(8, prompt_set="phonetic")
    check("weighted queue differs from unweighted", [p["text"] for p in base] != [p["text"] for p in weighted])

    print("\n[7b] difficulty hot-reload")
    r = c.post("/api/difficulty", json={"NG": 5.0, "ZH": 4.0})
    check("POST /api/difficulty accepted", r.status_code == 200 and r.json()["phones"] == 2)
    check("GET reflects it", c.get("/api/difficulty").json()["weights"].get("NG") == 5.0)
    r = c.post("/api/difficulty", json={"NG": "not a number"})
    check("bad weights rejected", r.status_code == 400)
    c.post("/api/difficulty", json={})

    print("\n[7c] live pipeline")
    st_live = c.get("/api/live/status").json()
    check("live status responds", "asr_installed" in st_live)
    board = c.get("/api/live/board").json()
    check("critical board populated", len(board["phrases"]) >= 20)
    if st_live["asr_installed"]:
        r = c.post("/api/live/transcribe", files={"audio": ("u.wav", fake_wav(seconds=2.0), "audio/wav")})
        check("live transcribe returns", r.status_code == 200 and "text" in r.json())
        lcid = r.json()["clip_id"]
        r = c.post("/api/live/confirm", data={"clip_id": lcid, "spoken_text": "corrected words", "was_edited": True})
        check("live confirm logs correction", r.status_code == 200)
    else:
        print("  SKIP  Parakeet model not installed")

    print("\n[8] pages render")
    from willspeak import __version__ as V

    for path in ("/", "/review", "/coverage", "/live"):
        r = c.get(path)
        check(f"GET {path}", r.status_code == 200 and "<html" in r.text.lower())
        check(f"{path} assets version-stamped", f"common.js?v={V}" in r.text)
        check(f"{path} no-cache header", r.headers.get("cache-control") == "no-cache")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}\n")
    return fails


def test_smoke():
    fails = run_smoke()
    assert not fails, "smoke checks failed: " + ", ".join(fails)


if __name__ == "__main__":
    raise SystemExit(1 if run_smoke() else 0)
