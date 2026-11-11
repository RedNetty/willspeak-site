from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from .. import audio as A
from ..paths import PROJECTION_DIR, WAVLM_DIR
from . import scoring as SC

MODEL_ID = "microsoft/wavlm-base-plus"
LAYER = 9
POOL = "mean+std"
HIDDEN = 768
DIM = 2 * HIDDEN
SESSION_WARMUP = 20


def norm_label(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_l2n = SC.l2_normalize


class Embedder:
    def __init__(self) -> None:
        self.device = os.environ.get("WILLSPEAK_MATCH_DEVICE", "auto")
        self._model = None
        self._fe = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure(self):
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoFeatureExtractor, AutoModel
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
                if not WAVLM_DIR.is_dir():
                    raise FileNotFoundError("WavLM is not installed; run python scripts/fetch_models.py")
                source = str(WAVLM_DIR)
                self._fe = AutoFeatureExtractor.from_pretrained(source, local_files_only=True)
                m = AutoModel.from_pretrained(source, local_files_only=True).eval()
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    m = m.to(self.device)
                except Exception:  # noqa: BLE001
                    self.device = "cpu"
                self._model = m
                print(f"[willspeak] matcher: {MODEL_ID} layer {LAYER} on {self.device}")
            return self._model

    def __call__(self, x16k: np.ndarray) -> np.ndarray:
        import torch
        m = self._ensure()
        y = prep_audio(x16k)
        inp = self._fe(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = m(**{k: v.to(self.device) for k, v in inp.items()}, output_hidden_states=True)
        return pool_frames(out.hidden_states[LAYER][0].float().cpu().numpy())


def pool_frames(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.float32)
    if h.ndim != 2 or h.shape[0] == 0:
        return np.zeros(DIM, np.float32)
    return np.concatenate([h.mean(0), h.std(0)]).astype(np.float32)


def prep_audio(x: np.ndarray, max_s: float = 12.0, min_s: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)[: int(16000 * max_s)]
    y = A.trim_silence(x, 16000, pad_s=0.05)
    if len(y) < 4800:
        y = x
    m = float(np.abs(y).max()) if len(y) else 0.0
    y = (y / m * 0.9) if m > 0 else y
    need = int(16000 * min_s)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    return y


class Projection:
    def __init__(self, path: str | Path | None):
        self.path = str(path) if path else None
        self.W = None
        self.T = None
        self.mean = np.zeros(DIM, np.float32)
        self.meta: dict = {"type": "identity"}
        if path and Path(path).exists():
            z = np.load(path, allow_pickle=True)
            self.W = z["W"].astype(np.float32)
            self.mean = z["mean"].astype(np.float32)
            if "T" in z.files:
                self.T = z["T"].astype(np.float32)
            try:
                self.meta = json.loads(str(z["meta"]))
            except Exception:  # noqa: BLE001
                self.meta = {}
        elif path:
            print(f"[willspeak] matcher: no projection file at {path}; scoring in the raw {DIM}-d space")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32) - self.mean
        if self.W is not None:
            x = x @ self.W
        if self.T is not None:
            x = x @ self.T
        return x

    @property
    def dim(self) -> int:
        return DIM if self.W is None else int(self.W.shape[1])

    @property
    def in_dim(self) -> int:
        return DIM if self.W is None else int(self.W.shape[0])

    def describe(self) -> str:
        if self.W is None:
            return "identity (no projection file)"
        kind = self.meta.get("session_transform") or ("T" if self.T is not None else "none")
        return (f"{self.meta.get('train', 'projection')} · {self.meta.get('pool', POOL)} · "
                f"alpha {self.meta.get('alpha', '?')} · session transform: {kind}")


class Matcher:
    def __init__(self, store, data_dir: Path) -> None:
        self.store = store
        self.data_dir = Path(data_dir)
        self.enabled = os.environ.get("WILLSPEAK_MATCH", "1") != "0"
        self.max_words = int(os.environ.get("WILLSPEAK_MATCH_MAXWORDS", "6"))
        self.warmup = float(os.environ.get("WILLSPEAK_MATCH_SESSION_WARMUP", str(SESSION_WARMUP)) or SESSION_WARMUP)
        proj_path = os.environ.get("WILLSPEAK_MATCH_PROJECTION") or str(PROJECTION_DIR / "projection.npz")
        self.proj = Projection(proj_path)
        self.emb = Embedder()
        if self.proj.W is not None and self.proj.in_dim != DIM:
            raise ValueError(
                f"{proj_path}: projection expects {self.proj.in_dim}-d input but the embedder produces {DIM}-d "
                f"({POOL}). Refit it: python scripts/matcher/train_projection.py --emb data/emb/torgo.npz "
                f"--out src/willspeak/models/projection.npz")
        self.cache_path = self.data_dir / "match_cache.npz"
        self._lock = threading.RLock()
        self.ids: list[str] = []
        self.labels: list[str] = []
        self.sessions: list[str] = []
        self.raw: list[np.ndarray] = []
        self.vecs: np.ndarray | None = None
        self.proto_labels: list[str] = []
        self.protos: np.ndarray | None = None
        self.sess_vecs: dict[str, np.ndarray] = {}
        self.bank_mean: np.ndarray | None = None
        self.live_sum: np.ndarray | None = None
        self.live_n = 0
        self.last_build: dict = {}
        self.building = False

    def available(self) -> bool:
        return self.enabled and WAVLM_DIR.is_dir() and Embedder.available()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "installed": Embedder.available(),
            "model_present": WAVLM_DIR.is_dir(),
            "projection": self.proj.describe(),
            "projection_dim": self.proj.dim,
            "pool": POOL,
            "session_warmup": self.warmup,
            "labels": len(set(self.labels)),
            "exemplars_own": len(self.ids),
            "sessions": len(self.sess_vecs),
            "device": self.emb.device,
            "building": self.building,
            "last_build": self.last_build,
        }

    def _candidate_rows(self) -> list[dict]:
        rows = self.store.con.execute(
            """SELECT c.id, c.path, COALESCE(c.transcript, c.prompt_text) label, c.session_id,
                      p.prompt_set, p.kind
               FROM clip c LEFT JOIN prompt p ON p.id = c.prompt_id
               WHERE c.status = 'accepted' ORDER BY c.id"""
        ).fetchall()
        out = []
        for r in rows:
            lab = norm_label(r["label"])
            if not lab or len(lab.split()) > self.max_words:
                continue
            if r["prompt_set"] in ("phonetic", "probe"):
                continue
            out.append({"id": int(r["id"]), "path": r["path"], "label": lab,
                        "session": f"s{r['session_id']}"})
        return out

    def vocabulary(self) -> list[str]:
        labs = set(self.labels)
        for r in self.store.con.execute(
                "SELECT text, kind FROM prompt WHERE active = 1 AND prompt_set IN ('critical','functional','custom')"):
            lab = norm_label(r["text"])
            if lab and len(lab.split()) <= self.max_words:
                labs.add(lab)
        return sorted(labs)

    def take_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lab in self.labels:
            counts[lab] = counts.get(lab, 0) + 1
        return counts

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        try:
            z = np.load(self.cache_path, allow_pickle=True)
            raw = z["raw"]
            if raw.ndim != 2 or raw.shape[1] != DIM:
                print(f"[willspeak] matcher: cache is {raw.shape[1] if raw.ndim == 2 else '?'}-d, embedder is {DIM}-d; re-embedding")
                return {}
            return {str(k): v for k, v in zip(z["ids"], raw)}
        except Exception:  # noqa: BLE001
            return {}

    def _save_cache(self) -> None:
        if self.ids:
            np.savez(self.cache_path, ids=np.array(self.ids), raw=np.stack(self.raw),
                     labels=np.array(self.labels), sessions=np.array(self.sessions))

    def start_background_build(self) -> None:
        if not self.available() or self.building:
            return
        def _go():
            try:
                self.building = True
                summary = self.rebuild()
                print(f"[willspeak] matcher bank ready: {summary}")
            except Exception as e:  # noqa: BLE001
                print(f"[willspeak] matcher build failed: {e}")
            finally:
                self.building = False
        threading.Thread(target=_go, daemon=True, name="matcher-build").start()

    def rebuild(self) -> dict:
        if not self.available():
            return {"ok": False, "reason": "torch/transformers not installed"}
        t0 = time.time()
        with self._lock:
            cached = self._load_cache()
            self.ids, self.labels, self.sessions, self.raw = [], [], [], []
            n_new = 0
            for row in self._candidate_rows():
                key = f"clip:{row['id']}"
                vec = cached.get(key)
                if vec is None:
                    path = self.data_dir / str(row["path"]).replace("\\", "/")
                    if not path.exists():
                        continue
                    import soundfile as sf
                    x, sr = sf.read(str(path), dtype="float32")
                    x = A.to_mono_16k(x, sr)
                    vec = self.emb(x)
                    n_new += 1
                self.ids.append(key); self.labels.append(row["label"]); self.sessions.append(row["session"])
                self.raw.append(np.asarray(vec, np.float32))
            self._finalize()
            self._save_cache()
            self.last_build = {"at": time.time(), "seconds": round(time.time() - t0, 1), "new_embeddings": n_new,
                               "exemplars": len(self.ids), "labels": len(set(self.labels)),
                               "sessions": len(self.sess_vecs)}
            return {"ok": True, **self.last_build}

    def _finalize(self) -> None:
        if not self.raw:
            self.vecs = self.protos = None
            self.proto_labels = []
            self.sess_vecs = {}
            self.bank_mean = None
            return
        R = np.stack(self.raw)
        Rn, self.sess_vecs = SC.center_by_session(R, np.array(self.sessions))
        self.vecs = _l2n(self.proj(Rn))
        self.proto_labels, self.protos = SC.label_prototypes(self.vecs, self.labels)
        self.bank_mean = R.mean(0).astype(np.float32)

    def remove_exemplar(self, clip_id: int) -> None:
        with self._lock:
            key = f"clip:{clip_id}"
            if key not in self.ids:
                return
            index = self.ids.index(key)
            for entries in (self.ids, self.labels, self.sessions, self.raw):
                entries.pop(index)
            self._finalize()
            if self.ids:
                self._save_cache()
            else:
                self.cache_path.unlink(missing_ok=True)

    def add_exemplar(self, clip_id: int, label: str, x16k: np.ndarray, session_id: int | None) -> bool:
        lab = norm_label(label)
        if not lab or len(lab.split()) > self.max_words or not self.available():
            return False
        v = self.emb(x16k)
        with self._lock:
            key = f"clip:{clip_id}"
            if key in self.ids:
                index = self.ids.index(key)
                self.labels[index] = lab
                self.raw[index] = v
                self._finalize()
                self._save_cache()
                return True
            self.ids.append(key); self.labels.append(lab); self.sessions.append(f"s{session_id}")
            self.raw.append(v)
            self._finalize()
            self._save_cache()
            return True

    def new_session(self) -> None:
        with self._lock:
            self.live_sum = None
            self.live_n = 0
            self._finalize()

    @property
    def live_mean(self) -> np.ndarray | None:
        if self.bank_mean is None:
            return None
        return SC.shrunk_session_mean(self.live_sum, self.live_n, self.bank_mean, self.warmup)

    def _live_norm(self, v: np.ndarray, update: bool) -> np.ndarray:
        if update:
            self.live_sum, self.live_n = SC.update_running_sum(self.live_sum, self.live_n, v)
        mu = self.live_mean
        return v if mu is None else v - mu

    def match(self, x16k: np.ndarray, topk: int = 3, asr_text: str | None = None,
              update_session: bool = True) -> dict:
        if not self.available():
            return {"ok": False, "reason": "matcher not installed", "candidates": []}
        if self.vecs is None:
            self.rebuild()
        if self.vecs is None:
            return {"ok": False, "reason": "no exemplars yet", "candidates": []}
        t0 = time.time()
        v = self.emb(x16k)
        with self._lock:
            if self.vecs is None:
                return {"ok": False, "reason": "no exemplars yet", "candidates": []}
            q = _l2n(self.proj(self._live_norm(v, update_session)))
            s = SC.cosine_scores(self.vecs, q)
            ps = SC.cosine_scores(self.protos, q)
            best = SC.score_labels(ps, self.proto_labels, s, self.labels, self.sessions)
        boosted = SC.apply_asr_vote(best, norm_label(asr_text) if asr_text else None)
        cands, top, margin = SC.rank_candidates(best, topk)
        for c in cands:
            c["source"] = "own"
        return {
            "ok": True,
            "candidates": cands,
            "top_score": top,
            "margin": margin,
            "asr_vote": boosted,
            "n_labels": len(best),
            "latency_ms": int((time.time() - t0) * 1000),
        }
