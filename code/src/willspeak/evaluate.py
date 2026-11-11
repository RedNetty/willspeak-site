from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .db import is_placeholder
from .g2p import SIL, text_phones
from .normalize import normalize


SUB, DEL, INS, OK = "sub", "del", "ins", "ok"


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])

    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append((OK, ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append((SUB, ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((DEL, ref[i - 1], None)); i -= 1
        else:
            ops.append((INS, None, hyp[j - 1])); j -= 1
    ops.reverse()
    return ops


def _rate(ops) -> dict:
    c = Counter(o[0] for o in ops)
    n_ref = c[OK] + c[SUB] + c[DEL]
    err = c[SUB] + c[DEL] + c[INS]
    return {
        "n_ref": n_ref, "sub": c[SUB], "dele": c[DEL], "ins": c[INS],
        "errors": err, "rate": (err / n_ref) if n_ref else float(err),
    }


@dataclass
class Result:
    n_utts: int = 0
    wer: float = 0.0
    cer: float = 0.0
    per: float = 0.0
    word: dict = field(default_factory=dict)
    char: dict = field(default_factory=dict)
    phone: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)
    per_phoneme: dict = field(default_factory=dict)
    confusions: list = field(default_factory=list)
    worst: list = field(default_factory=list)
    oov_words: list = field(default_factory=list)

    def dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def evaluate(pairs: Sequence[dict]) -> Result:
    w_ops, c_ops, p_ops = [], [], []
    grp_ops: dict[str, list] = defaultdict(list)
    ph_stats: dict[str, Counter] = defaultdict(Counter)
    conf: Counter = Counter()
    per_utt = []
    oov: Counter = Counter()

    for p in pairs:
        ref_n, hyp_n = normalize(p["ref"]), normalize(p.get("hyp", "") or "")
        rw, hw = ref_n.split(), hyp_n.split()
        ops_w = align(rw, hw)
        w_ops += ops_w
        grp_ops[p.get("group") or "all"] += ops_w
        c_ops += align(list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", "")))

        rp, r_oov = text_phones(ref_n)
        hp, _ = text_phones(hyp_n)
        oov.update(r_oov)
        rp = [x for x in rp if x != SIL]
        hp = [x for x in hp if x != SIL]
        ops_p = align(rp, hp)
        p_ops += ops_p
        for op, r, h in ops_p:
            if r is None:
                continue
            ph_stats[r][op] += 1
            if op == SUB and h:
                conf[(r, h)] += 1

        u = _rate(ops_w)
        per_utt.append({"id": p.get("id"), "ref": p["ref"], "hyp": p.get("hyp", ""),
                        "wer": round(u["rate"], 3), "n_words": u["n_ref"]})

    res = Result(n_utts=len(pairs))
    res.word, res.char, res.phone = _rate(w_ops), _rate(c_ops), _rate(p_ops)
    res.wer, res.cer, res.per = (round(res.word["rate"], 4),
                                 round(res.char["rate"], 4),
                                 round(res.phone["rate"], 4))
    res.by_group = {g: {**_rate(o), "rate": round(_rate(o)["rate"], 4)}
                    for g, o in grp_ops.items()}

    res.per_phoneme = {}
    for ph, c in ph_stats.items():
        n = c[OK] + c[SUB] + c[DEL]
        if n == 0:
            continue
        res.per_phoneme[ph] = {
            "n": n, "correct": c[OK], "sub": c[SUB], "del": c[DEL],
            "error_rate": round((c[SUB] + c[DEL]) / n, 4),
        }
    res.confusions = [{"ref": a, "hyp": b, "n": n} for (a, b), n in conf.most_common(25)]
    errs = [u for u in per_utt if u["n_words"] > 0 and u["wer"] > 0]
    res.worst = sorted(errs, key=lambda u: -u["wer"])[:25]
    res.oov_words = [w for w, _ in oov.most_common(30)]
    return res


def phoneme_difficulty(res: Result, lo: float = 1.0, hi: float = 5.0,
                       min_n: int = 5) -> dict[str, float]:
    rows = {p: v["error_rate"] for p, v in res.per_phoneme.items() if v["n"] >= min_n}
    if not rows:
        return {}
    lo_e, hi_e = min(rows.values()), max(rows.values())
    span = (hi_e - lo_e) or 1.0
    return {p: round(lo + (hi - lo) * (e - lo_e) / span, 3) for p, e in rows.items()}


def load_manifest(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if not isinstance(r, dict) or r.get("id") is None or not isinstance(r.get("text"), str):
                    raise ValueError(f"Manifest rows require id and text: {path}")
                if is_placeholder(r["text"]):
                    raise ValueError(f"Clip {r['id']} has no transcript, only the live placeholder: {path}")
                key = str(r["id"])
                if key in out:
                    raise ValueError(f"Duplicate manifest id {key}: {path}")
                out[key] = r
    return out


def pairs_from(manifest: dict[str, dict], hyps: dict[str, str],
               group_key: str = "prompt_set") -> list[dict]:
    return [
        {"id": k, "ref": v["text"], "hyp": hyps.get(k, ""),
         "group": v.get(group_key)}
        for k, v in manifest.items()
    ]


def transcribe_parakeet(manifest: dict[str, dict], root: str | Path,
                        model_dir: str | Path | None = None) -> dict[str, str]:
    from .asr.parakeet import ParakeetASR

    asr = ParakeetASR(Path(model_dir) if model_dir else None)
    if not asr.installed():
        raise ImportError("sherpa-onnx is not installed in this Python")
    if not asr.model_present():
        raise FileNotFoundError(f"no Parakeet model in {asr.model_dir} -run scripts/fetch_models.py")
    out = {}
    for i, (k, v) in enumerate(manifest.items()):
        out[k] = asr.transcribe(str(Path(root) / v["audio"]))["text"]
        if len(manifest) >= 20 and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(manifest)} clips…")
    return out
