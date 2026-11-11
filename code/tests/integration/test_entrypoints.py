from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").rglob("*.py")
    if not p.name.startswith("_") and 'if __name__ == "__main__":' in p.read_text(encoding="utf-8")
)
LAUNCHER = ROOT / ("willspeak.bat" if os.name == "nt" else "willspeak.sh")
LAUNCH_COMMAND = [str(LAUNCHER)] if os.name == "nt" else ["bash", str(LAUNCHER)]
LAUNCHER_COMMANDS = ("run", "check", "fetch-models", "bench", "evaluate", "train", "export", "match-eval")


@pytest.mark.parametrize("command", COMMANDS)
def test_script_help_from_other_directory(command, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / command), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.skipif(os.name != "nt" and shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize("name", LAUNCHER_COMMANDS)
def test_launcher_forwards_help(name, tmp_path):
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(Path(sys.executable).parent.parent)
    result = subprocess.run(
        [*LAUNCH_COMMAND, name, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.skipif(os.name != "nt" and shutil.which("bash") is None, reason="bash is not installed")
def test_launcher_preserves_failure_exit(tmp_path):
    result = subprocess.run(
        [*LAUNCH_COMMAND, "run", "--not-a-real-option"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2


@pytest.mark.skipif(os.name != "nt" and shutil.which("bash") is None, reason="bash is not installed")
def test_launcher_rejects_unknown_command(tmp_path):
    result = subprocess.run(
        [*LAUNCH_COMMAND, "no-such-command"],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 2
    assert "unknown command" in (result.stdout + result.stderr).lower()


def test_windows_launcher_targets_exist():
    content = (ROOT / "willspeak.bat").read_text()
    targets = re.findall(r'"%PY%" (scripts\\[\w\\]+\.py|run\.py)', content)
    targets += re.findall(r'set "SCRIPT=(scripts\\[\w\\]+\.py|run\.py)"', content)
    assert targets
    for t in targets:
        assert (ROOT / t.replace("\\", "/")).is_file(), t
    assert "pause" not in content.lower()
