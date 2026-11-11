from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__
from . import audio as A
from .db import LIVE_PROMPT, Store
from .paths import DATA_DIR, PROMPT_DIR, STATIC_DIR
from .scheduler import Scheduler, difficulty_weights, seed_prompts

store = Store(DATA_DIR)
seed_prompts(store, str(PROMPT_DIR))


def _load_difficulty() -> dict:
    path = os.environ.get("WILLSPEAK_DIFFICULTY")
    if not path:
        return {}
    try:
        weights = difficulty_weights(json.loads(Path(path).read_text(encoding="utf-8")))
        print(f"[willspeak] loaded difficulty weights for {len(weights)} phones from {path}")
        return weights
    except (OSError, TypeError, ValueError) as e:
        print(f"[willspeak] could not load WILLSPEAK_DIFFICULTY={path}: {e}")
        return {}


sched = Scheduler(store, difficulty=_load_difficulty(),
                  redundancy=int(os.environ.get("WILLSPEAK_REDUNDANCY", "2")))

from .matcher.engine import Matcher

matcher = Matcher(store, DATA_DIR)

from .asr import asr

from .handsfree import Settings, decide

live_settings = Settings(DATA_DIR)
_last_spoken = {"at": None, "clip_id": None, "text": ""}
_live_state = {"awake": True}
_live_decode_lock = threading.Lock()

_recent_live: list[tuple[float, object]] = []
_DUPLICATE_WINDOW_S = 10.0


def _is_recent_duplicate(x, sr: int) -> bool:
    now = time.time()
    _recent_live[:] = [(t, y) for t, y in _recent_live if now - t < _DUPLICATE_WINDOW_S]
    return any(A.same_audio(x, y, sr) for _, y in _recent_live)


def _remember_live(x) -> None:
    _recent_live.append((time.time(), x))
    del _recent_live[:-5]


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].rstrip(".!?")


def _warm_asr_in_background() -> None:
    import threading

    def run():
        try:
            asr.warm_up()
        except Exception as e:  # noqa: BLE001
            print(f"[willspeak] asr: warm-up skipped: {e}")

    threading.Thread(target=run, name="asr-warm-up", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("WILLSPEAK_MATCH_EAGER", "1") != "0":
        matcher.start_background_build()
    if os.environ.get("WILLSPEAK_ASR_EAGER", "1") != "0" and asr.available():
        _warm_asr_in_background()
    yield


app = FastAPI(title="WillSpeak recorder", lifespan=lifespan)


@app.middleware("http")
async def no_stale_cache(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static", "/review", "/coverage", "/live", "/compare")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def _page(name: str) -> str:
    html = (STATIC_DIR / name).read_text(encoding="utf-8")
    for asset in ("style.css", "common.js"):
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={__version__}")
    return html


@app.get("/", response_class=HTMLResponse)
def index():
    return _page("index.html")


@app.get("/review", response_class=HTMLResponse)
def review_page():
    return _page("review.html")


@app.get("/coverage", response_class=HTMLResponse)
def coverage_page():
    return _page("coverage.html")


@app.get("/live", response_class=HTMLResponse)
def live_page():
    return _page("live.html")


@app.get("/api/difficulty")
def difficulty():
    return {"weights": sched.difficulty, "redundancy": sched.redundancy}


@app.post("/api/difficulty")
def set_difficulty(weights: Annotated[dict, Body()]):
    try:
        sched.difficulty = difficulty_weights(weights)
    except (TypeError, ValueError, AttributeError) as e:
        raise HTTPException(400, f"expected a JSON object of phone -> weight: {e}")
    sched.invalidate()
    return {"ok": True, "phones": len(sched.difficulty)}


@app.get("/api/state")
def state():
    from .backup import backup_stats
    session = store.open_session()
    return {
        "totals": store.totals(),
        "session": dict(session) if session else None,
        "data_dir": str(DATA_DIR.resolve()),
        "backup": backup_stats(DATA_DIR),
    }


def _rating(v: str | None) -> float | None:
    try:
        if v is None or not str(v).strip():
            return None
        value = float(v)
        return max(0.0, min(10.0, value)) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


@app.post("/api/session/start")
def session_start(label: str = Form(""), condition: str = Form("unknown"),
                  mic: str = Form(""), room: str = Form(""),
                  self_rating: str | None = Form(None), fatigue: str | None = Form(None),
                  rater: str = Form(""), rater_severity: str = Form("")):
    with store._lock:
        cur = store.open_session()
        if cur:
            return {"session_id": cur["id"], "resumed": True}
        sid = store.start_session(label, condition, mic, room,
                                 self_rating=_rating(self_rating), fatigue=_rating(fatigue),
                                 rater=rater.strip(), rater_severity=rater_severity.strip().lower())
        return {"session_id": sid, "resumed": False}


@app.post("/api/session/end")
def session_end(session_id: int = Form(...), notes: str = Form(""),
                self_rating: str | None = Form(None), fatigue: str | None = Form(None)):
    store.end_session(session_id, notes, self_rating=_rating(self_rating), fatigue=_rating(fatigue))
    return {"ok": True}


@app.get("/api/probe")
def probe_list():
    rows = store.prompts("probe")
    return {"prompts": [{"prompt_id": r["id"], "text": r["text"], "prompt_set": "probe", "kind": r["kind"]}
                        for r in sorted(rows, key=lambda r: r["id"])]}


@app.get("/api/next")
def next_prompts(n: int = 25, prompt_set: str | None = None):
    return {"prompts": sched.next_batch(n=n, prompt_set=prompt_set)}


@app.get("/api/coverage")
def coverage():
    return sched.coverage_report()


@app.post("/api/prompt")
def add_prompt(text: str = Form(...), prompt_set: str = Form("custom"),
               target_takes: int = Form(2)):
    from .g2p import text_phones
    from .scheduler import classify_kind
    ph, oov = text_phones(text)
    pid = store.add_prompt(text.strip(), prompt_set, classify_kind(text),
                           target_takes=target_takes, phones=ph, oov=oov)
    sched.invalidate()
    return {"prompt_id": pid, "oov": oov, "phones": ph,
            "duplicate": pid is None}


QUALITY_RETRIES = 1


def _apply_quality_gate(qc, x, retry: int) -> None:
    """Add the gate's verdict to a take's QC.

    A first failure is an error, so the take is stored for retake and the prompt
    comes back. After QUALITY_RETRIES more failures on the same prompt the take
    is kept with a warning instead: repeating a prompt costs the speaker real
    effort, and every take stays on disk with its scores either way.
    """
    from . import quality

    g = quality.gate()
    if g is None:
        return
    v = g(x).dict()
    if not v["keep"]:
        if retry < QUALITY_RETRIES:
            qc.errors.extend(v["messages"])
        else:
            v["kept_after_retry"] = True
            qc.warnings.extend(v["messages"])
            qc.warnings.append("Kept after another try. Check this take in Review.")
    qc.gate = v


@app.post("/api/clip")
async def upload_clip(
    audio: Annotated[UploadFile, File()],
    session_id: int = Form(...),
    prompt_text: str = Form(...),
    prompt_id: int | None = Form(None),
    quality_retry: int = Form(0),
):
    if store.session(session_id) is None:
        raise HTTPException(404, "no such session")
    if prompt_id is not None and store.con.execute("SELECT id FROM prompt WHERE id = ?", (prompt_id,)).fetchone() is None:
        raise HTTPException(404, "no such prompt")
    if not prompt_text.strip():
        raise HTTPException(400, "prompt text must contain words")
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        canon, sr, qc, orig, orig_sr = A.ingest(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}")
    _apply_quality_gate(qc, canon, quality_retry)

    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = f"s{session_id:04d}_{ts}_{uuid4().hex}"
    cpath = DATA_DIR / "clips" / f"{stem}.wav"
    A.write_wav(str(cpath), canon, sr, subtype="PCM_16")

    rpath = None
    if orig_sr != sr or (orig.ndim > 1):
        rpath = DATA_DIR / "raw" / f"{stem}.wav"
        A.write_wav(str(rpath), orig, orig_sr, subtype="FLOAT")

    cid = store.add_clip(
        session_id, prompt_id, prompt_text,
        str(cpath.relative_to(DATA_DIR)),
        str(rpath.relative_to(DATA_DIR)) if rpath else None,
        qc.duration_s, qc.dict(),
    )
    sched.invalidate()
    return {"clip_id": cid, "qc": qc.dict(), "totals": store.totals()}


@app.get("/api/clips")
def list_clips(status: str | None = None, session_id: int | None = None,
               limit: int = 200, offset: int = 0):
    rows = store.clips(status=status, session_id=session_id,
                       limit=limit, offset=offset)
    out = []
    for row in rows:
        clip = dict(row)
        clip["qc"] = json.loads(clip["qc"])
        out.append(clip)
    return {"clips": out}


@app.get("/api/clip/{cid}/audio")
def clip_audio(cid: int):
    row = store.clip(cid)
    if not row:
        raise HTTPException(404, "no such clip")
    return FileResponse(DATA_DIR / row["path"], media_type="audio/wav")


@app.post("/api/clip/{cid}/review")
def review(cid: int, transcript: str = Form(""), status: str = Form("accepted")):
    if status not in {"accepted", "rejected", "retake", "pending"}:
        raise HTTPException(400, "bad status")
    try:
        store.review_clip(cid, transcript, status)
    except KeyError:
        raise HTTPException(404, "no such clip")
    except ValueError as e:
        raise HTTPException(400, str(e))
    sched.invalidate()
    matcher.remove_exemplar(cid)
    return {"ok": True, "totals": store.totals()}


@app.get("/api/live/settings")
def get_live_settings():
    return live_settings.get()


@app.post("/api/live/settings")
async def set_live_settings(request: Request):
    form = await request.form()
    try:
        vals = live_settings.update(dict(form))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    special = [str(form[k]).strip() for k in ("stop_phrase", "sleep_phrase", "wake_phrase") if k in form and str(form[k]).strip()]
    if special:
        from .g2p import text_phones
        from .scheduler import classify_kind
        for sp in special:
            ph, oov = text_phones(sp)
            store.add_prompt(sp, "critical", classify_kind(sp), target_takes=15, priority=100, phones=ph, oov=oov)
        sched.invalidate()
    return vals


@app.get("/api/live/awake")
def get_awake():
    return {"awake": _live_state["awake"]}


@app.post("/api/live/awake")
def set_awake(awake: bool = Form(...)):
    _live_state["awake"] = bool(awake)
    return {"awake": _live_state["awake"]}


@app.get("/api/live/status")
def live_status():
    return {"asr_installed": asr.available(), "asr_backend": asr.name, "model": asr.model_name,
            "asr_ready": bool(getattr(asr, "warm", False)),
            "asr_eager": os.environ.get("WILLSPEAK_ASR_EAGER", "1") != "0",
            "asr_load_s": getattr(asr, "load_s", None),
            "device_requested": asr.device, "device_actual": asr.actual_device,
            "matcher": matcher.status(), "awake": _live_state["awake"]}


@app.post("/api/live/match/rebuild")
def match_rebuild():
    return matcher.rebuild()


@app.post("/api/live/match/new_session")
def match_new_session():
    matcher.new_session()
    return {"ok": True}


@app.get("/api/live/vocab")
def live_vocab():
    counts = matcher.take_counts()
    return {"labels": [{"text": t, "own_takes": counts.get(t, 0)} for t in matcher.vocabulary()]}


@app.post("/api/live/vocab")
def live_vocab_add(text: str = Form(...), prompt_set: str = Form("functional"),
                   target_takes: int = Form(4)):
    from .g2p import text_phones
    from .scheduler import classify_kind
    ph, oov = text_phones(text)
    pid = store.add_prompt(text.strip(), prompt_set, classify_kind(text),
                           target_takes=target_takes, phones=ph, oov=oov)
    sched.invalidate()
    return {"prompt_id": pid, "duplicate": pid is None, "oov": oov,
            "needs_takes": True}


@app.get("/api/live/board")
def live_board():
    rows = store.prompts("critical")
    return {"phrases": [r["text"] for r in rows]}


@app.post("/api/live/transcribe")
async def live_transcribe(audio: Annotated[UploadFile, File()],
                          session_id: int | None = Form(None),
                          preview_only: bool = Form(False),
                          pending_output: bool = Form(False),
                          gate_dbfs: float | None = Form(None)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if gate_dbfs is not None and not (math.isfinite(gate_dbfs) and -120 <= gate_dbfs <= 0):
        gate_dbfs = None
    return await run_in_threadpool(_decode_live, raw, session_id, preview_only, pending_output, gate_dbfs)


def _decode_live(raw: bytes, session_id: int | None, preview_only: bool, pending_output: bool,
                 gate_dbfs: float | None = None) -> dict:
    with _live_decode_lock:
        return _process_live(raw, session_id, preview_only, pending_output, gate_dbfs)


def _process_live(raw: bytes, session_id: int | None, preview_only: bool, pending_output: bool,
                  gate_dbfs: float | None = None) -> dict:
    if not asr.available() and not matcher.available():
        raise HTTPException(503, "neither a sentence recogniser nor the matcher is installed -"
                                 "run: python -m pip install -r requirements.txt && "
                                 "python scripts/fetch_models.py")
    if session_id is not None and store.session(session_id) is None:
        raise HTTPException(404, "no such session")
    try:
        canon, sr, qc, orig, orig_sr = A.ingest(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}")

    trimmed = A.trim_silence(canon, sr)
    dur = len(trimmed) / sr
    reason = None
    if dur < A.MIN_DURATION_S:
        reason = "no speech found in the take"
    elif dur > A.MAX_DURATION_S:
        reason = f"utterance too long (over {A.MAX_DURATION_S:.0f}s)"

    if reason is None and _is_recent_duplicate(trimmed, sr):
        return {"clip_id": None, "ok": False, "reason": "duplicate of the previous utterance "
                "(is another Speak page open on this microphone?)", "qc": qc.dict(),
                "text": ""}

    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = f"live_{ts}_{uuid4().hex}"
    cpath = DATA_DIR / "clips" / f"{stem}.wav"
    A.write_wav(str(cpath), canon, sr, subtype="PCM_16")
    rpath = None
    if orig_sr != sr or orig.ndim > 1:
        rpath = DATA_DIR / "raw" / f"{stem}.wav"
        A.write_wav(str(rpath), orig, orig_sr, subtype="FLOAT")

    sid = session_id
    if sid is None:
        with store._lock:
            cur = store.open_session()
            sid = cur["id"] if cur else store.start_session(label="live", condition="unknown")
    cid = store.add_clip(sid, None, LIVE_PROMPT, str(cpath.relative_to(DATA_DIR)),
                         str(rpath.relative_to(DATA_DIR)) if rpath else None,
                         round(len(canon) / sr, 3), qc.dict())
    if reason is None:
        _remember_live(trimmed)

    if reason:
        return {"clip_id": cid, "ok": False, "reason": reason, "qc": qc.dict(),
                "text": ""}

    if asr.available():
        try:
            res = asr.transcribe_array(trimmed, sr)
        except Exception as e:  # noqa: BLE001
            print(f"[willspeak] live clip {cid}: recogniser error: {e!r}")
            return {"clip_id": cid, "ok": False, "reason": f"recogniser error: {e}",
                    "qc": qc.dict(), "text": "", "match": None,
                    "decision": {"action": "ignore", "reason": "the recogniser failed; see the server log",
                                 "awake": _live_state["awake"]}}
    else:
        res = {"text": "", "latency_ms": 0, "model": "none", "device": None}
    match = None
    if matcher.available():
        try:
            match = matcher.match(trimmed, topk=3, asr_text=res.get("text") or None,
                                  update_session=not preview_only)
        except Exception as e:  # noqa: BLE001
            match = {"ok": False, "reason": f"matcher error: {e}", "candidates": []}
    hint = _first_sentence(qc.warnings[0]) if qc.warnings else None
    decision = decide(res.get("text", ""), match, live_settings.get(), _last_spoken["at"],
                      awake=_live_state["awake"])
    if pending_output and decision["action"] == "stop":
        decision.update(retract=False, retract_text="")
    if not preview_only and decision["action"] in ("sleep", "wake"):
        _live_state["awake"] = decision["action"] == "wake"
        store.suggest_transcript(cid, live_settings.get()[decision["action"] + "_phrase"])
    decision["awake"] = _live_state["awake"]
    print(f"[willspeak] live clip {cid}: dur={dur:.1f}s decode={res.get('latency_ms', 0)}ms "
          + (f"load={res['load_ms']}ms " if res.get("load_ms") else "")
          + f"snr={qc.est_snr_db:.0f}dB speech={qc.speech_dbfs:.0f}dBFS "
          + (f"gate={gate_dbfs:.0f}dBFS " if gate_dbfs is not None else "")
          + (f"logp={res['mean_logprob']:.2f} " if res.get("mean_logprob") is not None else "")
          + f"action={decision['action']}" + (f" by={decision['by']}" if decision.get("by") else "")
          + f" words={len((res.get('text') or '').split())} text={(res.get('text') or '')[:60]!r}")
    if not preview_only and decision["action"] == "stop":
        store.suggest_transcript(cid, live_settings.get()["stop_phrase"])
        if decision.get("retract") and _last_spoken["clip_id"]:
            store.review_clip(_last_spoken["clip_id"], "", "rejected")
            matcher.remove_exemplar(_last_spoken["clip_id"])
            sched.invalidate()
            _last_spoken.update(at=None, clip_id=None, text="")
    return {"clip_id": cid, "ok": True, "reason": None, "hint": hint,
            "qc": qc.dict(), "match": match, "decision": decision, **res}


@app.post("/api/live/confirm")
def live_confirm(clip_id: int = Form(...), spoken_text: str = Form(...), stage: str = Form("start")):
    if stage not in {"start", "end"}:
        raise HTTPException(400, "stage must be start or end")
    if stage == "end":
        if _last_spoken["clip_id"] == clip_id:
            _last_spoken["at"] = time.time()
        return {"ok": True, "status": "pending", "exemplar_added": False}
    try:
        store.suggest_transcript(clip_id, spoken_text)
    except KeyError:
        raise HTTPException(404, "no such clip")
    _last_spoken.update(at=time.time(), clip_id=clip_id, text=spoken_text)
    return {"ok": True, "status": "pending", "exemplar_added": False}


@app.post("/api/live/fix")
def live_fix(clip_id: int = Form(...), correct_text: str = Form("")):
    try:
        store.review_clip(clip_id, correct_text, "accepted" if correct_text.strip() else "rejected")
    except KeyError:
        raise HTTPException(404, "no such clip")
    except ValueError as e:
        raise HTTPException(400, str(e))
    sched.invalidate()
    matcher.remove_exemplar(clip_id)
    if not correct_text.strip():
        return {"ok": True, "status": "rejected"}
    added = False
    if matcher.available():
        row = store.clip(clip_id)
        try:
            import soundfile as sf
            x, sr = sf.read(str(DATA_DIR / str(row["path"]).replace("\\", "/")), dtype="float32")
            added = matcher.add_exemplar(clip_id, correct_text, A.to_mono_16k(x, sr), row["session_id"])
        except Exception as e:  # noqa: BLE001
            print(f"[willspeak] could not add exemplar for clip {clip_id}: {e}")
    return {"ok": True, "status": "accepted", "exemplar_added": added}


@app.get("/api/live/recent")
def live_recent(limit: int = 12):
    rows = store.con.execute(
        """SELECT id, prompt_text, transcript, status, duration_s, created_at FROM clip
           WHERE prompt_text = ? ORDER BY id DESC LIMIT ?""", (LIVE_PROMPT, limit)).fetchall()
    return {"clips": [dict(r) for r in rows], "last_spoken": _last_spoken}


@app.post("/api/backup")
def backup():
    from .backup import make_backup
    path = make_backup(store, DATA_DIR)
    return {"ok": True, "file": path.name, "path": str(path.resolve()),
            "bytes": path.stat().st_size}


@app.post("/api/export")
def export(dev_frac: float = Form(0.1), test_frac: float = Form(0.1),
           trim: bool = Form(True)):
    from .export import export_corpus
    try:
        res = export_corpus(store, DATA_DIR, dev_frac=dev_frac,
                            test_frac=test_frac, trim=trim)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(res)


from .compare import Comparison

comparison = Comparison(asr, _live_decode_lock)


@app.get("/compare", response_class=HTMLResponse)
def compare_page():
    return _page("compare.html")


@app.get("/api/compare/state")
def compare_state():
    return comparison.state()


@app.post("/api/compare/speaker")
async def compare_speaker(speaker: str = Form(...), model: str | None = Form(None)):
    def run():
        comparison.warm_stock()
        return comparison.select(speaker, model or None)
    try:
        return await run_in_threadpool(run)
    except KeyError as exc:
        raise HTTPException(404, f"unknown speaker {speaker}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/compare/audio/{clip_id}")
def compare_audio(clip_id: str):
    try:
        path = comparison.audio_path(clip_id)
    except KeyError as exc:
        raise HTTPException(404, f"no clip {clip_id} in the demo list") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/compare/release")
async def compare_release():
    return await run_in_threadpool(comparison.release)


@app.post("/api/compare/decode")
async def compare_decode(clip_id: str = Form(...)):
    try:
        return await run_in_threadpool(comparison.decode, clip_id)
    except KeyError as exc:
        raise HTTPException(404, f"no clip {clip_id} in the demo list") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/compare/try")
async def compare_try(audio: Annotated[UploadFile, File()]):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        canon, sr, qc, _orig, _orig_sr = A.ingest(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}") from e
    trimmed = A.trim_silence(canon, sr)
    dur = len(trimmed) / sr
    if qc.speech_ratio <= 0 or qc.peak_dbfs < -45 or dur < A.MIN_DURATION_S:
        raise HTTPException(422, "no speech found in the recording")
    if dur > A.MAX_DURATION_S:
        raise HTTPException(422, f"too long: keep it under {A.MAX_DURATION_S:.0f} s")
    try:
        return await run_in_threadpool(comparison.try_voice, trimmed, dur)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


from .pimic import PiMic

pi_mic = PiMic()


@app.get("/api/pimic/status")
def pimic_status():
    return pi_mic.status()


@app.get("/api/pimic/stream")
async def pimic_stream():
    try:
        q = await pi_mic.subscribe()
    except OSError as e:
        raise HTTPException(503, f"the box's microphone could not be opened: {e}") from e
    return StreamingResponse(pi_mic.stream(q), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                      "X-Accel-Buffering": "no", "X-PCM-Rate": str(pi_mic.status()["rate"])})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
