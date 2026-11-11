#!/usr/bin/env python3
"""Choose among each clip's candidate sentences with a language model, then count what that choice did. Lab test 36.

Input: the pools ``candidate_pool.py`` wrote, one per speaker (or one pool whose manifest names the speaker of each clip),
and the manifests with the reference transcripts. For every candidate the total is

    total = acoustic + lambda x LM + bonus x words

where ``acoustic`` is the speech model's own log-likelihood of the sentence given the audio (already in the pool), ``LM`` is the
language model's log-probability of the sentence as text, and ``words`` is the word count. The highest total wins. The language
model never hears the audio, so it can only choose among what the speech model proposed.

``lambda`` and ``bonus`` are chosen on the other speakers' clips (fewest wrong words on a small grid) and applied to the speaker
being counted, so no speaker is scored with weights it influenced. One extra row per model uses the fixed pair (0.7, 3) from
round 7 with no fitting at all, and one row uses no language model (lambda 0, bonus fitted) so the pool's own effect is visible.

Counts per row, per speaker and pooled: exact sentences right, wrong words (all clips), wrong words on sentence clips only,
empty answers, each change versus greedy with a paired bootstrap over clips (2,000 resamples, seed 0). The oracle row picks
the candidate with the fewest wrong words and is the ceiling: the share of clips whose right sentence is in the pool at all.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import ci
import labkit as lab
from willspeak.artifacts import versions
from willspeak.evaluate import OK, align
from willspeak.normalize import normalize


@dataclass(frozen=True)
class LM:
    tag: str
    name: str
    params: str
    released: str
    note: str
    gated: bool = False


ROSTER: tuple[LM, ...] = (
    LM("gpt2", "openai-community/gpt2", "124M", "2019", "the weak control from round 7; MIT"),
    LM("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", "0.5B", "Sep 2024", "round 7's model; Apache-2.0"),
    LM("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", "0.6B", "Apr 2025", "Apache-2.0"),
    LM("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", "1.7B", "Apr 2025", "Apache-2.0"),
    LM("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B-Base", "0.8B", "Feb 2026", "hybrid attention; needs a transformers that knows qwen3_5; Apache-2.0"),
    LM("qwen3.5-2b", "Qwen/Qwen3.5-2B-Base", "2.3B", "Feb 2026", "the largest here that could still fit a Pi as a 4-bit file; Apache-2.0"),
    LM("minicpm5-1b", "openbmb/MiniCPM5-1B", "1.1B", "May 2026", "plain Llama architecture; Apache-2.0"),
    LM("lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Base", "1.2B", "Jan 2026", "needs transformers 4.54 or newer; LFM open license"),
    LM("granite4-350m", "ibm-granite/granite-4.0-h-350m-base", "350M", "Oct 2025", "hybrid Mamba; needs transformers 4.57 or newer; Apache-2.0"),
    LM("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "360M", "Nov 2024", "Apache-2.0"),
    LM("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", "1.7B", "Nov 2024", "Apache-2.0"),
    LM("gemma3-270m", "google/gemma-3-270m", "270M", "Aug 2025", "gated: accept Google's terms on Hugging Face and run huggingface-cli login first", True),
    LM("gemma3-1b", "google/gemma-3-1b-pt", "1B", "Mar 2025", "gated, as gemma3-270m", True),
    LM("llama3.2-1b", "meta-llama/Llama-3.2-1B", "1.2B", "Sep 2024", "gated: accept Meta's terms on Hugging Face and log in first", True),
)
BY_TAG = {m.tag: m for m in ROSTER}
assert len(BY_TAG) == len(ROSTER), "duplicate tag"

LAMBDAS = (0.3, 0.5, 0.7, 1.0, 1.5)
BONUSES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
FIXED = (0.7, 3.0)
RESAMPLES, SEED = 2000, 0
NO_LM: dict[str, float] = {}


# ----- pure parts: choosing, fitting, counting -----------------------------------------------------------------------


def parse_pairs(items: list[str], what: str) -> dict[str, str]:
    """SPK=PATH items into a dict; a single bare PATH is returned under the key ''."""
    out: dict[str, str] = {}
    for item in items:
        if "=" in item:
            speaker, path = item.split("=", 1)
        elif len(items) == 1:
            speaker, path = "", item
        else:
            raise ValueError(f"Several --{what} need SPK=PATH each, got {item!r}")
        if speaker in out or not path:
            raise ValueError(f"--{what} lists {speaker or 'the bare path'} twice or has no path: {item!r}")
        out[speaker] = path
    return out


def read_pool(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with lab.here(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in out or not row.get("candidates"):
                    raise ValueError(f"Duplicate clip or empty candidate list for {row.get('id')!r} in {path}")
                out[str(row["id"])] = row
    if not out:
        raise ValueError(f"Empty pool: {path}")
    return out


def split_by_speaker(manifest: dict[str, dict], pool: dict[str, dict]) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """One pool and one manifest covering several speakers, split by the manifest's speaker field."""
    manifests: dict[str, dict[str, dict]] = {}
    pools: dict[str, dict[str, dict]] = {}
    for clip_id, row in manifest.items():
        speaker = str(row.get("speaker") or "")
        if not speaker:
            raise ValueError(f"Clip {clip_id} has no speaker field; name the speaker with SPK=PATH instead")
        manifests.setdefault(speaker, {})[clip_id] = row
        if clip_id in pool:
            pools.setdefault(speaker, {})[clip_id] = pool[clip_id]
    return pools, manifests


def lm_score(lm: dict[str, float], text: str) -> float:
    """The language model's log-probability of a candidate; empty text scores 0 (the word bonus is what keeps empties down)."""
    if not normalize(text).split():
        return 0.0
    if text not in lm:
        raise KeyError(f"No language-model score for {text!r}")
    return lm[text]


def candidate_total(cand: dict, lam: float, bonus: float, lm: dict[str, float]) -> float | None:
    if cand.get("acoustic") is None:
        return None
    lm_part = lam * lm_score(lm, cand["text"]) if lam else 0.0
    return float(cand["acoustic"]) + lm_part + bonus * float(cand["n_words"])


def choose(row: dict, lam: float, bonus: float, lm: dict[str, float]) -> dict:
    """The candidate with the highest total; ties keep the earlier candidate (greedy comes first in every pool)."""
    best, best_total = None, None
    for cand in row["candidates"]:
        total = candidate_total(cand, lam, bonus, lm)
        if total is None:
            continue
        if best is None or total > best_total:
            best, best_total = cand, total
    return best if best is not None else row["candidates"][0]


def greedy_candidate(row: dict) -> dict:
    for cand in row["candidates"]:
        if "greedy" in cand["sources"]:
            return cand
    return row["candidates"][0]


def wrong_words(ref_norm: str, text: str) -> int:
    return sum(op != OK for op, _, _ in align(ref_norm.split(), normalize(text).split()))


def oracle_candidate(row: dict, ref_norm: str) -> dict:
    """The candidate with the fewest wrong words; ties keep the earlier one."""
    return min(row["candidates"], key=lambda c: wrong_words(ref_norm, c["text"]))


def fit_weights(train: list[tuple[dict, str]], lm: dict[str, float], lambdas=LAMBDAS, bonuses=BONUSES) -> tuple[float, float]:
    """The (lambda, bonus) with the fewest wrong words on the training clips; ties go to the smaller lambda, then the smaller bonus."""
    if not train:
        raise ValueError("No clips to fit weights on")
    best: tuple[int, float, float] | None = None
    for lam in lambdas:
        for bonus in bonuses:
            wrong = sum(wrong_words(ref, choose(row, lam, bonus, lm)["text"]) for row, ref in train)
            key = (wrong, lam, bonus)
            if best is None or key < best:
                best = key
    return best[1], best[2]


def leave_one_out(pools: dict[str, dict[str, dict]], refs: dict[str, dict[str, str]], lm: dict[str, float],
                  lambdas=LAMBDAS, bonuses=BONUSES) -> dict[str, tuple[float, float]]:
    """Weights for each speaker, fitted on every other speaker's clips."""
    if len(pools) < 2:
        raise ValueError("Leave-one-speaker-out needs at least two speakers")
    weights: dict[str, tuple[float, float]] = {}
    for held_out in pools:
        train = [(row, refs[other][clip_id]) for other in pools if other != held_out for clip_id, row in pools[other].items()]
        weights[held_out] = fit_weights(train, lm, lambdas, bonuses)
    return weights


def answers_with(pools: dict[str, dict[str, dict]], weights: dict[str, tuple[float, float]], lm: dict[str, float]) -> dict[str, dict[str, str]]:
    return {s: {clip_id: choose(row, *weights[s], lm)["text"] for clip_id, row in pool.items()} for s, pool in pools.items()}


@dataclass
class Count:
    clips: list[tuple[str, str]]  # (speaker, clip id), the order of every per-clip array
    errors: np.ndarray
    lengths: np.ndarray
    sentence: np.ndarray  # 1 where the clip is a sentence
    exact: np.ndarray  # 1 where normalised answer == normalised reference
    empty: np.ndarray

    def pooled(self) -> dict:
        s = self.sentence == 1
        return {"clips": int(len(self.clips)), "wrong": int(self.errors.sum()), "words": int(self.lengths.sum()),
                "sentence_clips": int(s.sum()), "sentence_wrong": int(self.errors[s].sum()), "sentence_words": int(self.lengths[s].sum()),
                "exact_sentences": int(self.exact[s].sum()), "exact_words": int(self.exact[~s].sum()), "word_clips": int((~s).sum()),
                "empty": int(self.empty.sum())}

    def by_speaker(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for speaker in dict.fromkeys(s for s, _ in self.clips):
            mask = np.array([s == speaker for s, _ in self.clips])
            sub = Count([c for c in self.clips if c[0] == speaker], self.errors[mask], self.lengths[mask], self.sentence[mask],
                        self.exact[mask], self.empty[mask])
            out[speaker] = sub.pooled()
        return out


def count(answers: dict[str, dict[str, str]], refs: dict[str, dict[str, str]], kinds: dict[str, dict[str, str]]) -> Count:
    clips, errors, lengths, sentence, exact, empty = [], [], [], [], [], []
    for speaker in answers:
        for clip_id in answers[speaker]:
            ref, text = refs[speaker][clip_id], answers[speaker][clip_id]
            hyp_norm = normalize(text)
            clips.append((speaker, clip_id))
            errors.append(wrong_words(ref, text))
            lengths.append(len(ref.split()))
            sentence.append(1 if kinds[speaker][clip_id] == "sentence" else 0)
            exact.append(1 if hyp_norm == ref else 0)
            empty.append(0 if hyp_norm.split() else 1)
    return Count(clips, np.array(errors), np.array(lengths), np.array(sentence), np.array(exact), np.array(empty))


def paired_count_interval(base: np.ndarray, new: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> list[float]:
    """95% interval for the change in a count (sum over clips of new - base), resampling clips with replacement."""
    if base.shape != new.shape or base.ndim != 1:
        raise ValueError("Paired interval needs two per-clip arrays of the same length")
    if base.size == 0:
        return [0.0, 0.0]
    diff = (new - base).astype(float)
    rng = np.random.RandomState(seed)
    draws = np.array([diff[rng.randint(0, diff.size, diff.size)].sum() for _ in range(resamples)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def compare(base: Count, new: Count) -> dict:
    """Changes of ``new`` against ``base`` on the same clips: words in WER points (the lab's gain), and counts with intervals."""
    if base.clips != new.clips:
        raise ValueError("The two systems were counted on different clips")
    s = base.sentence == 1
    draws = ci.bootstrap(base.errors, base.lengths, new.errors, n_resamples=RESAMPLES, seed=SEED)
    low, high = ci.interval(draws["delta"])
    words = int(base.lengths.sum())
    return {"wrong_change": int(new.errors.sum() - base.errors.sum()),
            "gain_points": float((base.errors.sum() - new.errors.sum()) / words * 100.0),
            "gain_ci_points": [float(low) * 100.0, float(high) * 100.0],
            "exact_sentences_change": int(new.exact[s].sum() - base.exact[s].sum()),
            "exact_sentences_ci": paired_count_interval(base.exact[s], new.exact[s]),
            "sentence_wrong_change": int(new.errors[s].sum() - base.errors[s].sum()),
            "sentence_wrong_ci": paired_count_interval(base.errors[s], new.errors[s])}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ----- the language models ------------------------------------------------------------------------------------------


def select_models(spec: str) -> list[LM]:
    if spec.strip() in ("", "all"):
        return list(ROSTER)
    tags = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tags if t not in BY_TAG]
    if unknown:
        raise KeyError(f"Unknown model tag(s) {unknown}; known: {', '.join(BY_TAG)}")
    return [BY_TAG[t] for t in tags]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(entry: LM, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(entry.name)
    try:
        model = AutoModelForCausalLM.from_pretrained(entry.name, dtype=dtype)
    except TypeError:  # transformers before 4.56 spell it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(entry.name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tok, model


def sentence_logprobs(tok, model, texts: list[str], device: str, batch: int = 16) -> list[float]:
    """Sum of log P(token | earlier tokens) over each text's tokens, after one start token; the text as written, not normalised."""
    import torch

    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if start is None:
        raise ValueError(f"{type(tok).__name__} has neither a start nor an end token to begin a sentence with")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out: list[float] = []
    with torch.no_grad():
        for first in range(0, len(texts), batch):
            chunk = texts[first:first + batch]
            seqs = [[start] + list(tok(t, add_special_tokens=False)["input_ids"]) for t in chunk]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            sums = (logp * mask[:, 1:]).sum(-1)
            out.extend(float(v) for v in sums.cpu())
    return out


def cached_scores(entry: LM, texts: list[str], cache_dir: Path, device: str, say) -> tuple[dict[str, float], dict]:
    """Scores for every text, from the cache file where present, computed and added otherwise."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{entry.tag}.json"
    cache: dict = {"model": entry.name, "scores": {}}
    if path.is_file():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache.get("model") != entry.name:
            raise ValueError(f"{path} holds scores from {cache.get('model')}, not {entry.name}; move it aside")
    scores: dict[str, float] = dict(cache["scores"])
    missing = [t for t in dict.fromkeys(texts) if t not in scores and normalize(t).split()]
    info: dict = {"name": entry.name, "params": entry.params, "released": entry.released, "note": entry.note,
                  "cached": len(texts) - len(missing), "scored": len(missing), "device": device}
    if missing:
        started = time.time()
        tok, model = load_lm(entry, device)
        info["load_seconds"] = round(time.time() - started, 1)
        started = time.time()
        for text, value in zip(missing, sentence_logprobs(tok, model, missing, device)):
            scores[text] = value
        info["score_seconds"] = round(time.time() - started, 1)
        import torch

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        path.write_text(json.dumps({"model": entry.name, "scores": scores}, indent=0, ensure_ascii=False), encoding="utf-8")
        say(f"  {entry.tag}: loaded in {info['load_seconds']} s, scored {len(missing)} sentences in {info['score_seconds']} s")
    else:
        say(f"  {entry.tag}: every sentence already in the cache")
    return scores, info


# ----- the run --------------------------------------------------------------------------------------------------------


def load_inputs(a) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    pool_paths, manifest_paths = parse_pairs(a.pool, "pool"), parse_pairs(a.manifest, "manifest")
    if "" in pool_paths or "" in manifest_paths:
        if set(pool_paths) != {""} or set(manifest_paths) != {""}:
            raise ValueError("Give one bare --pool and one bare --manifest (split by the speaker field), or SPK=PATH for both")
        pools, manifests = split_by_speaker(lab.read_manifest(manifest_paths[""]), read_pool(pool_paths[""]))
    else:
        if set(pool_paths) != set(manifest_paths):
            raise ValueError(f"--pool speakers {sorted(pool_paths)} and --manifest speakers {sorted(manifest_paths)} differ")
        pools = {s: read_pool(p) for s, p in pool_paths.items()}
        manifests = {s: lab.read_manifest(p) for s, p in manifest_paths.items()}
    for s, pool in pools.items():
        stray = sorted(set(pool) - set(manifests[s]))
        if stray:
            raise ValueError(f"{s}: {len(stray)} pooled clips are not in the manifest, e.g. {stray[:3]}")
        kinds = {clip_id for clip_id, row in manifests[s].items() if lab.clip_kind(row) in lab.KINDS}
        missing = sorted(kinds - set(pool))
        if missing and not a.allow_partial:
            raise ValueError(f"{s}: {len(missing)} manifest clips have no pool entry (e.g. {missing[:3]}); pass --allow-partial to count what is there")
        manifests[s] = {clip_id: manifests[s][clip_id] for clip_id in pool}
    return pools, manifests


def greedy_check(pools: dict[str, dict[str, dict]], items: list[str], say) -> dict:
    """How many pooled greedy answers differ from a saved decode of the same model (batching can move a few)."""
    out: dict = {}
    for s, path in parse_pairs(items, "check-greedy").items():
        if s not in pools:
            raise ValueError(f"--check-greedy names {s}, which has no pool")
        saved = lab.read_hyps(path)
        differ = [clip_id for clip_id, row in pools[s].items() if normalize(saved.get(clip_id, "")) != normalize(greedy_candidate(row)["text"])]
        out[s] = {"saved": str(path), "clips": len(pools[s]), "differ": len(differ), "examples": differ[:5]}
        say(f"greedy check {s}: {len(differ)} of {len(pools[s])} pooled greedy answers differ from {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", default=[], metavar="SPK=PATH", help="a speaker's pool from candidate_pool.py (repeat per speaker), or one bare path")
    ap.add_argument("--manifest", action="append", default=[], metavar="SPK=PATH", help="the matching manifest with references (repeat per speaker), or one bare path")
    ap.add_argument("--label", help="the system these pools came from, for the table, e.g. 'stock' or 'stage P adapter'")
    ap.add_argument("--models", default="all", help="comma-separated tags from the roster, or all (python lm_rescore.py --list shows them)")
    ap.add_argument("--list", action="store_true", help="print the roster and exit")
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS), help="grid of language-model weights")
    ap.add_argument("--bonuses", default=",".join(str(x) for x in BONUSES), help="grid of per-word bonuses")
    ap.add_argument("--fixed", default=f"{FIXED[0]},{FIXED[1]}", help="the untuned pair reported beside each model (round 7 used 0.7,3)")
    ap.add_argument("--cache", help="folder of cached language-model scores (default: lm-cache beside --out)")
    ap.add_argument("--check-greedy", action="append", default=[], metavar="SPK=HYPS", help="a saved decode of the same model to compare the pooled greedy answers with")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-partial", action="store_true", help="count a pool that covers only some of the manifest's clips")
    ap.add_argument("--out", help="folder for table.json and answers/ (refuses to overwrite table.json)")
    a = ap.parse_args()
    if a.list:
        for m in ROSTER:
            print(f"{m.tag:14} {m.params:>5}  {m.released:9} {m.name}  - {m.note}")
        return 0
    if not (a.pool and a.manifest and a.label and a.out):
        ap.error("--pool, --manifest, --label and --out are required (or --list)")
    lambdas = tuple(float(x) for x in a.lambdas.split(",") if x.strip())
    bonuses = tuple(float(x) for x in a.bonuses.split(",") if x.strip())
    fixed = tuple(float(x) for x in a.fixed.split(","))
    if not lambdas or not bonuses or len(fixed) != 2:
        ap.error("--lambdas and --bonuses need at least one value each; --fixed is LAMBDA,BONUS")
    models = select_models(a.models)
    out = lab.here(a.out)
    lab.refuse_existing(out / "table.json")
    cache_dir = lab.here(a.cache) if a.cache else out.parent / "lm-cache"
    say = print

    pools, manifests = load_inputs(a)
    speakers = list(pools)
    refs = {s: {clip_id: normalize(row["text"]) for clip_id, row in manifests[s].items()} for s in speakers}
    kinds = {s: {clip_id: lab.clip_kind(row) for clip_id, row in manifests[s].items()} for s in speakers}
    texts = sorted({c["text"] for s in speakers for row in pools[s].values() for c in row["candidates"]})
    n_cands = sum(len(row["candidates"]) for s in speakers for row in pools[s].values())
    n_clips = sum(len(pools[s]) for s in speakers)
    say(f"{a.label}: {n_clips} clips from {', '.join(speakers)}; {n_cands} candidates ({n_cands / n_clips:.1f} per clip), "
        f"{len(texts)} distinct sentences; grid {len(lambdas)} x {len(bonuses)}")
    checks = greedy_check(pools, a.check_greedy, say) if a.check_greedy else {}

    def row_for(label: str, answers: dict[str, dict[str, str]], weights: dict | None = None, **extra) -> dict:
        counted = count(answers, refs, kinds)
        row = {"label": label, "slug": slug(label), **counted.pooled(), "by_speaker": counted.by_speaker(), **extra}
        if weights is not None:
            row["weights"] = {s: {"lambda": w[0], "bonus": w[1]} for s, w in weights.items()}
        row["_count"] = counted
        row["_answers"] = answers
        return row

    greedy = row_for("greedy", {s: {clip_id: greedy_candidate(row)["text"] for clip_id, row in pools[s].items()} for s in speakers})
    rows = [greedy]
    none_weights = leave_one_out(pools, refs, NO_LM, (0.0,), bonuses)
    rows.append(row_for("acoustic only, bonus fitted", answers_with(pools, none_weights, NO_LM), none_weights))
    model_info: dict[str, dict] = {}
    device = None
    for entry in models:
        try:
            device = device or pick_device(a.device)
            scores, info = cached_scores(entry, texts, cache_dir, device, say)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            model_info[entry.tag] = {"name": entry.name, "status": "skipped", "reason": reason, "gated": entry.gated}
            say(f"  {entry.tag}: skipped ({reason})")
            continue
        model_info[entry.tag] = {"status": "scored", **info}
        fitted = leave_one_out(pools, refs, scores, lambdas, bonuses)
        rows.append(row_for(f"{entry.tag} fitted", answers_with(pools, fitted, scores), fitted, model=entry.tag))
        fixed_weights = {s: (fixed[0], fixed[1]) for s in speakers}
        rows.append(row_for(f"{entry.tag} fixed {fixed[0]:g},{fixed[1]:g}", answers_with(pools, fixed_weights, scores), fixed_weights, model=entry.tag))
    rows.append(row_for("oracle (best candidate per clip)", {s: {clip_id: oracle_candidate(row, refs[s][clip_id])["text"] for clip_id, row in pools[s].items()} for s in speakers}))

    base = greedy["_count"]
    for row in rows:
        if row is not greedy:
            row["vs_greedy"] = compare(base, row["_count"])
    say("")
    say(f"{'system':44} {'wrong/'+str(greedy['words']):>12} {'sent. wrong/'+str(greedy['sentence_words']):>18} "
        f"{'exact/'+str(greedy['sentence_clips']):>10} {'empty':>6}  change vs greedy")
    for row in rows:
        change = ""
        if "vs_greedy" in row:
            v = row["vs_greedy"]
            change = (f"words {v['wrong_change']:+d} ({v['gain_points']:+.1f} pts [{v['gain_ci_points'][0]:+.1f}, {v['gain_ci_points'][1]:+.1f}]), "
                      f"exact {v['exact_sentences_change']:+d} [{v['exact_sentences_ci'][0]:+.0f}, {v['exact_sentences_ci'][1]:+.0f}]")
        say(f"{row['label'][:44]:44} {row['wrong']:>12} {row['sentence_wrong']:>18} {row['exact_sentences']:>10} {row['empty']:>6}  {change}")
    for row in rows:
        if "weights" in row and row.get("model"):
            say(f"  {row['label']}: " + "; ".join(f"{s} lambda {w['lambda']:g} bonus {w['bonus']:g}" for s, w in row["weights"].items()))

    answers_dir = out / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for s, answers in row["_answers"].items():
            path = answers_dir / f"{row['slug']}-{s}.jsonl"
            path.write_text("".join(json.dumps({"id": clip_id, "hyp": text}) + "\n" for clip_id, text in answers.items()), encoding="utf-8")
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    inputs = {f"pool/{s}": p for s, p in parse_pairs(a.pool, "pool").items()}
    inputs.update({f"manifest/{s}": p for s, p in parse_pairs(a.manifest, "manifest").items()})
    lab.save_json(out / "table.json", {
        "format": "willspeak-lm-rescore-v1", "label": a.label, "speakers": speakers, "clips": n_clips, "candidates": n_cands,
        "distinct_sentences": len(texts), "grid": {"lambdas": lambdas, "bonuses": bonuses, "fixed": fixed},
        "resamples": RESAMPLES, "seed": SEED, "rows": clean_rows, "models": model_info, "roster": [asdict(m) for m in models],
        "greedy_check": checks, "packages": {k: v for k, v in versions().items() if k in ("torch", "transformers", "numpy")},
        "provenance": lab.provenance(__file__, inputs)})
    say(f"wrote {out / 'table.json'} and {len(rows) * len(speakers)} answer files under {answers_dir}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "lm_rescore"))
