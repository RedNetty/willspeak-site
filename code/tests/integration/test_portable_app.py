from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_application_runs_in_isolation(tmp_path):
    archive = tmp_path / "portable.zip"
    subprocess.run([sys.executable, str(ROOT / "package.py"), "--out", str(archive)], check=True)
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert "app/src/willspeak/models/projection.npz" in names
        assert "app/static/live.html" in names
        assert "app/prompts/critical.txt" in names
        for name in names:
            assert not set(Path(name).parts) & {
                "data", "docs", ".git", ".venv", "__pycache__", "reports", "logs",
            }
            assert not name.startswith("app/models/")
        bundle.extractall(tmp_path / "unpacked")
    copied = tmp_path / "unpacked/app"
    env = {k: v for k, v in os.environ.items() if not k.startswith("WILLSPEAK_") and k != "PYTHONPATH"}
    env.update(WILLSPEAK_ASR_EAGER="0", WILLSPEAK_MATCH_EAGER="0", HF_HUB_OFFLINE="1")
    script = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(copied / 'src')!r}); "
        "from willspeak.paths import ROOT, STATIC_DIR, PROMPT_DIR; "
        f"assert ROOT == Path({str(copied)!r}); "
        "assert (STATIC_DIR / 'live.html').is_file(); "
        "assert (PROMPT_DIR / 'critical.txt').is_file()"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, str(copied / "tests/integration/test_smoke.py")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, str(copied / "run.py"), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_includes_optional_models_and_refuses_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("portable_package", ROOT / "package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    copy = tmp_path / "app"
    copy.mkdir()
    for name in module.FILES:
        shutil.copy2(ROOT / name, copy / name)
    for name in module.DIRECTORIES:
        (copy / name).mkdir()
    (copy / "models/wavlm-base-plus").mkdir(parents=True)
    (copy / "models/wavlm-base-plus/config.json").write_text('{}')
    (copy / "data").mkdir()
    (copy / "data/private.txt").write_text('private')
    module.ROOT = copy
    archive = tmp_path / "models.zip"
    module.package(archive, with_models=True)
    with ZipFile(archive) as bundle:
        assert "app/models/wavlm-base-plus/config.json" in bundle.namelist()
        assert "app/data/private.txt" not in bundle.namelist()
    original = archive.read_bytes()
    with pytest.raises(FileExistsError):
        module.package(archive)
    assert archive.read_bytes() == original
