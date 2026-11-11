#!/usr/bin/env python3
"""Score candidate sentences with a small text language model: how likely are these words as English? Lab test 36."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = "models/lm"
FORMAT = "willspeak-lm-scores-v1"
COMPLETIONS = "willspeak-lm-completions-v1"
CHECK = ("The sun rises in the east and sets in the west.", "She opened the door and walked into the kitchen.",
         "It was raining, so we stayed home and watched a movie.")
HEALTHY = -5.0
CLOSE = 0.3
SPARE_MB = 2000


@dataclass(frozen=True)
class LM:
    tag: str
    name: str
    released: str
    params: str
    kind: str
    backend: str
    repo: str
    file: str
    mb: float
    licence: str
    note: str


def _gguf(tag, name, released, params, kind, repo, file, mb, licence="Apache-2.0", note="") -> LM:
    return LM(tag, name, released, params, kind, "gguf", repo, file, mb, licence, note)


CATALOG: tuple[LM, ...] = (
    LM("qwen25-05b-fp32", "Qwen2.5-0.5B", "Sep 2024", "0.49B", "base", "hf", "Qwen/Qwen2.5-0.5B", "", 988.0, "Apache-2.0",
       "full-precision weights through transformers and torch; the reference the 4-bit and 8-bit files are compared with"),
    _gguf("qwen25-05b-q8", "Qwen2.5-0.5B", "Sep 2024", "0.49B", "base", "mradermacher/Qwen2.5-0.5B-GGUF", "Qwen2.5-0.5B.Q8_0.gguf", 531.1),
    _gguf("qwen25-05b-q4", "Qwen2.5-0.5B", "Sep 2024", "0.49B", "base", "mradermacher/Qwen2.5-0.5B-GGUF", "Qwen2.5-0.5B.Q4_K_M.gguf", 397.8),
    _gguf("qwen25-15b-q4", "Qwen2.5-1.5B", "Sep 2024", "1.54B", "base", "mradermacher/Qwen2.5-1.5B-GGUF", "Qwen2.5-1.5B.Q4_K_M.gguf", 986.0),
    _gguf("smollm2-135m-q8", "SmolLM2-135M", "Nov 2024", "0.13B", "base", "QuantFactory/SmolLM2-135M-GGUF", "SmolLM2-135M.Q8_0.gguf", 144.8),
    _gguf("smollm2-360m-q8", "SmolLM2-360M", "Nov 2024", "0.36B", "base", "QuantFactory/SmolLM2-360M-GGUF", "SmolLM2-360M.Q8_0.gguf", 386.4),
    _gguf("smollm2-17b-q4", "SmolLM2-1.7B", "Nov 2024", "1.71B", "base", "QuantFactory/SmolLM2-1.7B-GGUF", "SmolLM2-1.7B.Q4_K_M.gguf", 1055.6),
    _gguf("llama32-1b-q4", "Llama 3.2 1B", "Sep 2024", "1.24B", "base", "QuantFactory/Llama-3.2-1B-GGUF", "Llama-3.2-1B.Q4_K_M.gguf", 807.7,
          "Llama 3.2 Community"),
    _gguf("gemma3-1b-q4", "Gemma 3 1B", "Mar 2025", "1.00B", "base", "mradermacher/gemma-3-1b-pt-GGUF", "gemma-3-1b-pt.Q4_K_M.gguf", 806.1, "Gemma"),
    _gguf("qwen3-06b-q4", "Qwen3-0.6B", "Apr 2025", "0.60B", "base", "mradermacher/Qwen3-0.6B-Base-GGUF", "Qwen3-0.6B-Base.Q4_K_M.gguf", 396.7),
    _gguf("qwen3-17b-q4", "Qwen3-1.7B", "Apr 2025", "1.72B", "base", "mradermacher/Qwen3-1.7B-Base-GGUF", "Qwen3-1.7B-Base.Q4_K_M.gguf", 1107.4),
    _gguf("gemma3-270m-q8", "Gemma 3 270M", "Aug 2025", "0.27B", "base", "ggml-org/gemma-3-270m-GGUF", "gemma-3-270m-Q8_0.gguf", 291.5, "Gemma"),
    _gguf("granite4-350m-q4", "Granite 4.0 350M", "Oct 2025", "0.35B", "base", "ibm-granite/granite-4.0-350m-base-GGUF",
          "granite-4.0-350m-base-Q4_K_M.gguf", 237.0),
    _gguf("granite4-1b-q4", "Granite 4.0 1B", "Oct 2025", "1.63B", "base", "ibm-granite/granite-4.0-1b-base-GGUF",
          "granite-4.0-1b-base-Q4_K_M.gguf", 1023.6),
    _gguf("lfm25-12b-q4", "LFM2.5-1.2B", "Jan 2026", "1.17B", "base", "LiquidAI/LFM2.5-1.2B-Base-GGUF", "LFM2.5-1.2B-Base-Q4_K_M.gguf", 730.9,
          "LFM Open License v1.0"),
    _gguf("qwen35-08b-q8", "Qwen3.5-0.8B", "Mar 2026", "0.75B", "base", "mradermacher/Qwen3.5-0.8B-Base-GGUF", "Qwen3.5-0.8B-Base.Q8_0.gguf", 811.8),
    _gguf("qwen35-08b-q4", "Qwen3.5-0.8B", "Mar 2026", "0.75B", "base", "mradermacher/Qwen3.5-0.8B-Base-GGUF", "Qwen3.5-0.8B-Base.Q4_K_M.gguf", 529.3),
    _gguf("qwen35-2b-q4", "Qwen3.5-2B", "Mar 2026", "1.88B", "base", "mradermacher/Qwen3.5-2B-Base-GGUF", "Qwen3.5-2B-Base.Q4_K_M.gguf", 1274.4),
    _gguf("lfm25-350m-q4", "LFM2.5-350M", "Mar 2026", "0.35B", "instruct", "LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_K_M.gguf", 229.3,
          "LFM Open License v1.0", "no base file is published in this format, so this is the chat-tuned model"),
    _gguf("minicpm5-1b-q4", "MiniCPM5-1B", "May 2026", "1.08B", "base", "mradermacher/MiniCPM5-1B-Base-GGUF", "MiniCPM5-1B-Base.Q4_K_M.gguf", 688.1),
    _gguf("lfm25-230m-q4", "LFM2.5-230M", "Jun 2026", "0.23B", "instruct", "LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_K_M.gguf", 153.4,
          "LFM Open License v1.0", "no base file is published in this format, so this is the chat-tuned model"),
    _gguf("minicpm5-2b-q4", "MiniCPM5-2B", "Sep 2026", "2.52B", "base", "mradermacher/MiniCPM5-2B-Base-GGUF", "MiniCPM5-2B-Base.Q4_K_M.gguf", 1561.3),
    _gguf("qwen35-4b-q4", "Qwen3.5-4B", "Mar 2026", "4.21B", "base", "mradermacher/Qwen3.5-4B-Base-GGUF", "Qwen3.5-4B-Base.Q4_K_M.gguf", 2708.8),
    _gguf("gemma4-e2b-q4", "Gemma 4 E2B", "Mar 2026", "4.65B", "instruct", "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf", 3106.7,
          note="no base file is published in this format, so this is the chat-tuned model"),
    _gguf("nanbeige42-3b-q4", "Nanbeige4.2-3B", "Jul 2026", "4.17B", "base", "mradermacher/Nanbeige4.2-3B-Base-GGUF",
          "Nanbeige4.2-3B-Base.Q4_K_M.gguf", 2574.8),
    _gguf("lfm25-26b-q4", "LFM2.5-2.6B", "Jul 2026", "2.70B", "base", "mradermacher/LFM2.5-2.6B-Base-GGUF", "LFM2.5-2.6B-Base.Q4_K_M.gguf", 1674.5,
          "LFM Open License v1.0"),
    _gguf("granite42-3b-q4", "Granite 4.2 3B", "Aug 2026", "3.66B", "instruct", "ibm-granite/granite-4.2-3b-GGUF", "granite-4.2-3b-Q4_K_M.gguf",
          2244.0, note="no base file is published in this format, so this is the chat-tuned model"),
    _gguf("qwen38-2b-distill-q4", "Qwen3.8-2B distill", "Aug 2026", "1.94B", "instruct", "empero-ai/Qwen3.8-2B-Distill-GGUF",
          "Qwen3.8-2B-Q4_K_M.gguf", 1312.2,
          note="not an Alibaba release: Empero trained Qwen3.5-2B to copy the answers of Qwen3.8; Alibaba's own Qwen3.8 starts at 27B"),
    _gguf("spark26-17b-q4", "Spark-X2.6-1.7B", "Oct 2026", "1.71B", "base", "mradermacher/Spark-X2.6-1.7B-Base-GGUF",
          "Spark-X2.6-1.7B-Base.Q4_K_M.gguf", 1107.5),
    _gguf("spark26-4b-q4", "Spark-X2.6-4B", "Oct 2026", "4.11B", "base", "mradermacher/Spark-X2.6-4B-Base-GGUF", "Spark-X2.6-4B-Base.Q4_K_M.gguf",
          2600.2),
    _gguf("qwen38-27b-q4", "Qwen3.8-27B", "Aug 2026", "27.32B", "instruct", "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf", 16464.4,
          note="far too large for the Pi: a ceiling that shows what a much stronger text model would choose; needs about 18 GB of memory"),
)
BY_TAG = {m.tag: m for m in CATALOG}
assert len(BY_TAG) == len(CATALOG), "duplicate tag"
GROUPS = {
    "pi": ("qwen25-05b-fp32 qwen25-05b-q8 qwen25-05b-q4 smollm2-360m-q8 gemma3-270m-q8 granite4-350m-q4 qwen3-06b-q4 lfm25-230m-q4 "
           "lfm25-350m-q4 qwen35-08b-q8 qwen35-08b-q4 minicpm5-1b-q4 lfm25-12b-q4 gemma3-1b-q4 llama32-1b-q4 granite4-1b-q4 qwen25-15b-q4 "
           "qwen35-2b-q4 qwen38-2b-distill-q4 spark26-17b-q4 minicpm5-2b-q4 lfm25-26b-q4"),
    "large": "granite42-3b-q4 nanbeige42-3b-q4 spark26-4b-q4 qwen35-4b-q4 gemma4-e2b-q4",
    "ceiling": "qwen38-27b-q4",
}
DEFAULT_TAGS = GROUPS["pi"]


def get(tag: str) -> LM:
    if tag not in BY_TAG:
        raise KeyError(f"Unknown language model {tag!r}; python scripts/asr/lm_score.py list shows the tags")
    return BY_TAG[tag]


def expand(words: str | list[str]) -> list[str]:
    tags: list[str] = []
    for word in (words.split() if isinstance(words, str) else words) or ["pi"]:
        for tag in (GROUPS[word].split() if word in GROUPS else [word]):
            get(tag)
            if tag not in tags:
                tags.append(tag)
    return tags


def group_of(tag: str) -> str:
    return next((name for name, tags in GROUPS.items() if tag in tags.split()), "extra")


def free_mb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e6


def store(root: str | None) -> Path:
    path = Path(root) if root else ROOT / STORE
    return path if path.is_absolute() else ROOT / path


def model_path(lm: LM, root: str | None = None) -> Path:
    folder = store(root) / lm.tag
    return folder / lm.file if lm.backend == "gguf" else folder


def present(lm: LM, root: str | None = None) -> bool:
    path = model_path(lm, root)
    return path.is_file() if lm.backend == "gguf" else (path / "config.json").is_file()


def fetch(lm: LM, root: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    folder = store(root) / lm.tag
    folder.mkdir(parents=True, exist_ok=True)
    if lm.backend == "gguf":
        hf_hub_download(lm.repo, lm.file, local_dir=str(folder))
    else:
        snapshot_download(lm.repo, local_dir=str(folder), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    return model_path(lm, root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def log_softmax_pick(row, index: int) -> float:
    import numpy as np

    row = np.asarray(row, dtype=np.float64)
    top = float(row.max())
    return float(row[index]) - (top + math.log(float(np.exp(row - top).sum())))


class Scorer:
    start: list[int] = []
    settled: dict = {}

    def start_options(self) -> list[tuple[str, list[int]]]:
        bos, eos = self.specials()
        line = self.tokens("\n")
        options: list[tuple[str, list[int]]] = []
        if bos is not None:
            options.append(("start token", [bos]))
        if eos is not None and eos != bos:
            options.append(("end token", [eos]))
        if line:
            options.append(("new line", line))
        if bos is not None and line:
            options.append(("start token, new line", [bos, *line]))
        if eos is not None and eos != bos and line:
            options.append(("end token, new line", [eos, *line]))
        return options

    def score(self, text: str) -> tuple[float, int]:
        seq = [*self.start, *self.tokens(text)]
        values = self.logps(seq, len(self.start))
        return sum(values), len(values)

    def complete(self, text: str, max_tokens: int) -> str:
        return self.generate([*self.start, *self.tokens(text)], max_tokens)


class GgufScorer(Scorer):
    def __init__(self, path: Path, threads: int, ctx: int):
        import llama_cpp

        self.version = f"llama-cpp-python {llama_cpp.__version__}"
        self.llm = llama_cpp.Llama(model_path=str(path), n_ctx=ctx, n_batch=ctx, n_threads=threads, n_threads_batch=threads,
                                   n_gpu_layers=0, logits_all=True, verbose=False)
        self.ctx = ctx
        self.vocab = int(self.llm.n_vocab())
        self.architecture = str(self.llm.metadata.get("general.architecture", ""))

    def specials(self) -> tuple[int | None, int | None]:
        bos = self.llm.token_bos() if self.llm.metadata.get("tokenizer.ggml.bos_token_id") is not None else None
        eos = self.llm.token_eos()
        return (bos if bos is not None and bos >= 0 else None, eos if eos is not None and eos >= 0 else None)

    def tokens(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def piece(self, ids: list[int]) -> str:
        return self.llm.detokenize(ids, special=True).decode("utf-8", "replace")

    def logps(self, seq: list[int], first: int) -> list[float]:
        if len(seq) > self.ctx:
            raise ValueError(f"{len(seq)} tokens do not fit --ctx {self.ctx}")
        self.llm.reset()
        self.llm.eval(seq)
        scores = self.llm.scores
        return [log_softmax_pick(scores[i - 1], seq[i]) for i in range(first, len(seq))]

    def generate(self, seq: list[int], max_tokens: int) -> str:
        self.llm.reset()
        out: list[int] = []
        for token in self.llm.generate(seq, top_k=1, top_p=1.0, min_p=0.0, temp=0.0, repeat_penalty=1.0, reset=True):
            if token == self.llm.token_eos() or len(out) >= max_tokens:
                break
            out.append(token)
        return self.llm.detokenize(out).decode("utf-8", "replace")


class HfScorer(Scorer):
    def __init__(self, path: Path, threads: int, ctx: int):
        import torch
        import transformers

        torch.set_num_threads(threads)
        self.torch = torch
        self.version = f"transformers {transformers.__version__}, torch {torch.__version__}"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(path))
        self.model = transformers.AutoModelForCausalLM.from_pretrained(str(path)).float()
        self.model.eval()
        self.ctx = ctx
        self.vocab = int(self.model.config.vocab_size)
        self.architecture = str(getattr(self.model.config, "model_type", ""))

    def specials(self) -> tuple[int | None, int | None]:
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        return (None if bos is None else int(bos), None if eos is None else int(eos))

    def tokens(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def piece(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def logps(self, seq: list[int], first: int) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            logits = self.model(torch.tensor([seq])).logits[0, :-1].double()
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp.gather(1, torch.tensor(seq[1:]).unsqueeze(1))[:, 0]
        return [float(v) for v in picked[first - 1:]]

    def generate(self, seq: list[int], max_tokens: int) -> str:
        torch = self.torch
        with torch.no_grad():
            out = self.model.generate(torch.tensor([seq]), max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, len(seq):], skip_special_tokens=True)


def settle(scorer) -> dict:
    tried: list[tuple[str, list[int], float]] = []
    for label, ids in scorer.start_options():
        scorer.start = ids
        total, count = 0.0, 0
        for sentence in CHECK:
            logp, n = scorer.score(sentence)
            total, count = total + logp, count + n
        tried.append((label, ids, total / count))
    if not tried:
        raise ValueError("This model has no start token, end token or new line to put before the first word")
    best = max(mean for _label, _ids, mean in tried)
    chance = -math.log(scorer.vocab)
    if not best > HEALTHY:
        raise ValueError(f"This runtime does not read the model correctly: ordinary sentences score {best:.1f} per token at best "
                         f"(a working model is near -2; guessing at random is {chance:.1f})")
    label, ids, mean = next(t for t in tried if t[2] >= best - CLOSE)
    scorer.start = ids
    return {"before_the_first_word": label, "text": scorer.piece(ids), "ids": ids, "check_mean_logp": round(mean, 3),
            "tried": {name: round(value, 3) for name, _ids, value in tried}, "chance": round(chance, 2)}


def load(lm: LM, root: str | None, threads: int, ctx: int):
    if not present(lm, root):
        raise FileNotFoundError(f"{lm.tag} is not downloaded; run: python scripts/asr/lm_score.py fetch --models {lm.tag}")
    scorer = (GgufScorer if lm.backend == "gguf" else HfScorer)(model_path(lm, root), threads, ctx)
    scorer.settled = settle(scorer)
    return scorer


def read_pools(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for item in paths:
        path = Path(item) if Path(item).is_absolute() else ROOT / item
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "willspeak-lm-pool-v1":
            raise ValueError(f"{path} is not a candidate pool written by lm_pool.py")
        rows += data["rows"]
    if not rows:
        raise ValueError("The pools hold no clips")
    return rows


def pool_texts(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for cand in row["candidates"]:
            if cand["lm_text"]:
                seen.setdefault(cand["lm_text"])
    return list(seen)


def plain_words(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower().replace("’", "'")).split()


def halves(rows: list[dict], least: int = 4) -> list[tuple[str, list[str]]]:
    seen: dict[str, tuple[str, list[str]]] = {}
    for row in rows:
        words = row["ref"].split()
        if row.get("kind") != "sentence" or len(words) < least or row["ref"] in seen:
            continue
        cut = (len(words) + 1) // 2
        start = " ".join("I" if w == "i" else w for w in words[:cut])
        seen[row["ref"]] = (start[0].upper() + start[1:], words[cut:])
    return list(seen.values())


def pending(pools: list[str], scores: Path, completions: Path) -> bool:
    try:
        rows = read_pools(pools)
        have = json.loads(scores.read_text(encoding="utf-8"))["texts"]
        done = {r["start"] + " | " + r["rest"] for r in json.loads(completions.read_text(encoding="utf-8"))["rows"]}
    except (OSError, ValueError, KeyError):
        return True
    return any(t not in have for t in pool_texts(rows)) or any(start + " | " + " ".join(rest) not in done for start, rest in halves(rows))


def out_path(text: str, extend: bool = False) -> Path:
    path = Path(text) if Path(text).is_absolute() else ROOT / text
    if path.exists() and not extend:
        raise FileExistsError(f"Already exists; choose a new name, or pass --extend to add what is missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def earlier(path: Path, lm: LM, kind: str) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != kind or data.get("model", {}).get("tag") != lm.tag:
        raise ValueError(f"{path} was not written by this command for {lm.tag}; choose a new name")
    return data


def save(path: Path, value: dict) -> None:
    scratch = path.with_name(path.name + ".partial")
    scratch.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(scratch, path)


def header(lm: LM, scorer, path: Path, threads: int, load_seconds: float, before: float | None) -> dict:
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.suffix == ".safetensors")
    return {"model": asdict(lm), "file_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
            "file_sha256": {p.name: sha256(p) for p in files}, "runtime": scorer.version, "architecture": scorer.architecture,
            "threads": threads, "start": scorer.settled, "load_seconds": round(load_seconds, 2),
            "rss_before_load_mb": None if before is None else round(before, 1),
            "machine": {"node": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
                        "cpus": os.cpu_count()}}


def cmd_list(a) -> int:
    for lm in CATALOG:
        have = "downloaded" if present(lm, a.dir) else "-"
        print(f"{lm.tag:<21} {lm.name:<19} {lm.released:<9} {lm.params:<7} {lm.kind:<8} {lm.mb:>7.0f} MB  {group_of(lm.tag):<8} {have:<10} {lm.repo}"
              + (f"\n{'':<21} {lm.note}" if lm.note else ""))
    print("\ngroups (a group name works wherever a tag does):")
    for name, tags in GROUPS.items():
        print(f"  {name:<8} {len(tags.split()):>2} models, {sum(get(t).mb for t in tags.split()) / 1000:.1f} GB: {tags}")
    print("  pi: the 4-bit or 8-bit file is under 1.7 GB.  large: 2 to 3 GB, fits beside Parakeet on an 8 GB Pi, slow there.")
    print("  ceiling: not for the Pi.  extra: older or smaller relatives, in no group.")
    return 0


def cmd_fetch(a) -> int:
    failed = []
    for tag in expand(a.models):
        lm = get(tag)
        if present(lm, a.dir):
            print(f"{tag}: already here")
            continue
        free = free_mb(store(a.dir))
        if free < lm.mb + SPARE_MB:
            print(f"{tag}: FAILED no space left on device for it: {lm.mb:.0f} MB and {SPARE_MB} MB to spare are needed, {free:.0f} MB are free")
            failed.append(tag)
            continue
        print(f"{tag}: downloading {lm.repo} {lm.file} ({lm.mb:.0f} MB)", flush=True)
        try:
            fetch(lm, a.dir)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: FAILED {type(exc).__name__}: {exc}")
            failed.append(tag)
    if failed:
        print(f"not downloaded: {' '.join(failed)}")
    return 1 if failed else 0


def cmd_check(a) -> int:
    bad = []
    for tag in expand(a.models):
        try:
            scorer = load(get(tag), a.dir, a.threads, a.ctx)
            said = " ".join(scorer.complete("The capital of France is", 8).split())
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: FAILED {type(exc).__name__}: {exc}", flush=True)
            bad.append(tag)
            continue
        found = scorer.settled
        tried = ", ".join(f"{name} {value}" for name, value in found["tried"].items())
        print(f"{tag}: reads ordinary sentences at {found['check_mean_logp']} per token with {found['before_the_first_word']} before the first "
              f"word (tried: {tried}; guessing at random is {found['chance']}); The capital of France is -> {said!r}", flush=True)
        del scorer
    if bad:
        print(f"not usable here: {' '.join(bad)}")
    return 1 if bad else 0


def cmd_release(a) -> int:
    for tag in expand(a.models):
        folder = store(a.dir) / get(tag).tag
        if not folder.is_dir():
            print(f"{tag}: not here")
            continue
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        shutil.rmtree(folder)
        print(f"{tag}: removed {folder} ({size / 1e6:.0f} MB)")
    return 0


def cmd_score(a) -> int:
    lm = get(a.model)
    out = out_path(a.out, a.extend)
    before_run = earlier(out, lm, FORMAT) if a.extend else {}
    scored: dict[str, list] = dict(before_run.get("texts", {}))
    texts = [t for t in pool_texts(read_pools(a.pool)) if t not in scored]
    if before_run and not texts:
        print(f"{lm.tag}: every sentence in these pools already has a score in {out}")
        return 0
    before = rss_mb()
    t0 = time.monotonic()
    scorer = load(lm, a.dir, a.threads, a.ctx)
    load_seconds = time.monotonic() - t0
    head = header(lm, scorer, model_path(lm, a.dir), a.threads, load_seconds, before)
    if before_run and before_run.get("file_sha256") != head["file_sha256"]:
        raise ValueError(f"{out} was scored with a different model file; choose a new name")
    if before_run and before_run.get("start", {}).get("ids") != head["start"].get("ids"):
        raise ValueError(f"{out} was scored with something else before the first word; choose a new name")
    scorer.score("The warm-up sentence is not counted.")
    started = time.monotonic()
    tokens = 0
    for n, text in enumerate(texts):
        t1 = time.monotonic()
        logp, count = scorer.score(text)
        scored[text] = [round(logp, 4), count, round(time.monotonic() - t1, 5)]
        tokens += count
        if n % 500 == 499:
            print(f"  {n + 1}/{len(texts)} sentences", flush=True)
    seconds = time.monotonic() - started
    peak = peak_rss_mb()
    result = {"format": FORMAT, **head, "peak_rss_mb": None if peak is None else round(peak, 1), "sentences": len(scored),
              "scored_now": len(texts), "tokens": tokens, "seconds": round(seconds, 2),
              "tokens_per_second": round(tokens / seconds, 1) if seconds else None, "texts": scored}
    save(out, result)
    print(f"{lm.tag}: {len(texts)} sentences, {tokens} tokens in {seconds:.1f} s ({result['tokens_per_second']} tokens a second, "
          f"{a.threads} threads); peak memory {result['peak_rss_mb']} MB; wrote {out}")
    return 0


def cmd_complete(a) -> int:
    lm = get(a.model)
    out = out_path(a.out, a.extend)
    before_run = earlier(out, lm, COMPLETIONS) if a.extend else {}
    rows = list(before_run.get("rows", []))
    have = {r["start"] + " | " + r["rest"] for r in rows}
    prompts = [(start, rest) for start, rest in halves(read_pools(a.pool)) if start + " | " + " ".join(rest) not in have]
    if before_run and not prompts:
        print(f"{lm.tag}: every test sentence in these pools is already in {out}")
        return 0
    scorer = load(lm, a.dir, a.threads, a.ctx)
    for start, rest in prompts:
        said = scorer.complete(start, a.max_tokens)
        rows.append({"start": start, "rest": " ".join(rest), "model": said.strip(), "finished": plain_words(said)[: len(rest)] == rest})
    finished = [r for r in rows if r["finished"]]
    save(out, {"format": COMPLETIONS, "model": asdict(lm), "runtime": scorer.version, "prompts": len(rows), "finished": len(finished),
               "rows": rows})
    print(f"{lm.tag}: finished {len(finished)} of {len(rows)} test sentences from their first half"
          + ("".join(f"\n  {r['start']} | {r['rest']}" for r in finished[:10])))
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="the language models this script knows, and which are downloaded")
    p.add_argument("--dir", help=f"where the model files live (default {STORE})")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("fetch", help="download language models from Hugging Face")
    p.add_argument("--models", nargs="+", required=True, help="tags or group names (pi, large, ceiling)")
    p.add_argument("--dir", help=f"where the model files live (default {STORE})")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("check", help="load downloaded models and see that each reads ordinary sentences; shows what goes before the first word")
    p.add_argument("--models", nargs="+", required=True, help="tags or group names")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--dir", help=f"where the model files live (default {STORE})")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("release", help="delete downloaded model files (score files are kept; fetch brings a model back)")
    p.add_argument("--models", nargs="+", required=True, help="tags or group names")
    p.add_argument("--dir", help=f"where the model files live (default {STORE})")
    p.set_defaults(fn=cmd_release)
    for name, fn, text in (("score", cmd_score, "log-probability of every candidate sentence in the pools"),
                           ("complete", cmd_complete, "memorisation check: can the model finish each test sentence from its first half?")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--model", required=True, help="a tag from `list`")
        p.add_argument("--pool", nargs="+", required=True, help="candidate pools written by lm_pool.py")
        p.add_argument("--out", required=True, help="results .json to write (refuses to overwrite unless --extend)")
        p.add_argument("--extend", action="store_true", help="keep what --out already holds and add only what these pools still lack")
        p.add_argument("--threads", type=int, default=4, help="CPU threads (4 is what a Raspberry Pi 5 has)")
        p.add_argument("--ctx", type=int, default=96, help="longest sentence in tokens")
        p.add_argument("--dir", help=f"where the model files live (default {STORE})")
        if name == "complete":
            p.add_argument("--max-tokens", type=int, default=24)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    if getattr(a, "threads", 1) < 1 or getattr(a, "ctx", 8) < 8 or getattr(a, "max_tokens", 1) < 1:
        ap.error("--threads and --max-tokens must be positive, --ctx at least 8")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
