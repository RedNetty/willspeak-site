from __future__ import annotations

import os
from pathlib import Path


def _find_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


PKG_DIR: Path = Path(__file__).resolve().parent

if os.environ.get("WILLSPEAK_ROOT"):
    ROOT: Path = Path(os.environ["WILLSPEAK_ROOT"]).resolve()
else:
    ROOT = _find_root(PKG_DIR)

DATA_DIR: Path = Path(os.environ.get("WILLSPEAK_DATA", ROOT / "data")).resolve()
PROMPT_DIR: Path = Path(os.environ.get("WILLSPEAK_PROMPTS", ROOT / "prompts")).resolve()
STATIC_DIR: Path = Path(os.environ.get("WILLSPEAK_STATIC", ROOT / "static")).resolve()
MODELS_DIR: Path = Path(os.environ.get("WILLSPEAK_MODELS", ROOT / "models")).resolve()
WAVLM_DIR: Path = MODELS_DIR / "wavlm-base-plus"

PROJECTION_DIR: Path = PKG_DIR / "models"
