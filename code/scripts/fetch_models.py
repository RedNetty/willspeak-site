#!/usr/bin/env python3
"""Download or verify the models required by willspeak."""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _cli

WAVLM = "microsoft/wavlm-base-plus"
PARAKEET_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
)


def ok(message: str) -> None:
    print(f"  ok    {message}")


def missing(message: str) -> None:
    print(f"  MISS  {message}")


def check_projection() -> bool:
    from willspeak.paths import PROJECTION_DIR

    path = PROJECTION_DIR / "projection.npz"
    print("matcher projection")
    if not path.exists():
        missing(f"{path} is missing")
        return False
    try:
        import numpy as np

        from willspeak.matcher.engine import DIM, POOL

        with np.load(path, allow_pickle=True) as data:
            weights = data["W"]
        if weights.shape[0] != DIM:
            missing(f"{path.name} expects {weights.shape[0]}-d input; the matcher embeds {DIM}-d ({POOL}). "
                    f"Refit: python scripts/matcher/train_projection.py --emb data/emb/torgo.npz --out {path}")
            return False
        ok(f"{path.name} {weights.shape[0]}->{weights.shape[1]} ({POOL})")
        return True
    except Exception as exc:  # noqa: BLE001
        missing(f"{path} is unreadable: {exc}")
        return False


def check_wavlm(download: bool) -> bool:
    from willspeak.paths import WAVLM_DIR

    print(f"\n{WAVLM} (matcher encoder)")
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError:
        missing(f"transformers is not installed in {sys.executable}")
        return False
    try:
        source = str(WAVLM_DIR) if WAVLM_DIR.is_dir() else WAVLM
        extractor = AutoFeatureExtractor.from_pretrained(source, local_files_only=not download)
        model = AutoModel.from_pretrained(source, local_files_only=not download)
        if download and source == WAVLM:
            WAVLM_DIR.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".wavlm-", dir=WAVLM_DIR.parent) as staging:
                target = Path(staging) / "wavlm-base-plus"
                extractor.save_pretrained(target)
                model.save_pretrained(target)
                target.rename(WAVLM_DIR)
    except Exception as exc:  # noqa: BLE001
        message = "not in the local cache; re-run without --check" if not download else f"download failed: {exc}"
        missing(message)
        return False
    ok(f"loadable: {source}")
    return True


def _download_parakeet(destination: Path) -> bool:
    archive_path: Path | None = None
    print(f"  downloading {PARAKEET_URL}")
    try:
        with tempfile.NamedTemporaryFile(prefix="willspeak-parakeet-", suffix=".tar.bz2", delete=False) as archive:
            archive_path = Path(archive.name)
            with urllib.request.urlopen(PARAKEET_URL, timeout=120) as response:
                shutil.copyfileobj(response, archive)

        destination.mkdir(parents=True, exist_ok=True)
        (destination / "test_wavs").mkdir(exist_ok=True)
        with tarfile.open(archive_path, mode="r:bz2") as bundle:
            for member in bundle.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name.endswith(".wav"):
                    target_path = destination / "test_wavs" / name
                elif name.endswith((".onnx", ".txt", ".vocab", ".md")):
                    target_path = destination / name
                else:
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    continue
                with source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
    except Exception as exc:  # noqa: BLE001
        missing(f"download failed: {exc}")
        return False
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
    return True


def check_parakeet(download: bool) -> bool:
    from willspeak.asr.parakeet import default_model_dir, model_files

    destination = default_model_dir()
    print(f"\nParakeet-TDT-0.6B v2 int8 -> {destination}")
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        missing(f"sherpa-onnx is not installed in {sys.executable}")
        return False

    if model_files(destination) is None:
        if not download:
            missing("model directory is incomplete; re-run without --check")
            return False
        if not _download_parakeet(destination):
            return False
    files = model_files(destination)
    if files is None:
        missing("encoder, decoder, joiner, or tokens file is missing")
        return False
    ok(", ".join(path.name for path in files.values()))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify local files without downloading")
    parser.add_argument("--parakeet-only", action="store_true", help="skip matcher assets")
    args = parser.parse_args()
    download = not args.check

    results = {"parakeet": check_parakeet(download)}
    if not args.parakeet_only:
        results["projection"] = check_projection()
        results["wavlm"] = check_wavlm(download)

    print()
    if all(results.values()):
        print("All requested models are ready. Run scripts/asr/check.py for an end-to-end decode check.")
        return 0
    print("Setup is incomplete; resolve the MISS lines above.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _cli.PAUSE_ALWAYS = True
    raise SystemExit(_cli.run(main, "fetch_models"))
