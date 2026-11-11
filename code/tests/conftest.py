import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("WILLSPEAK_LOG_DIR", tempfile.mkdtemp(prefix="willspeak-test-logs-"))
# The synthetic tones in tests/helpers.py are not speech, so the recording-quality
# gate's verdict on them means nothing. Tests that need the gate switch it on.
os.environ.setdefault("WILLSPEAK_QUALITY_GATE", "0")
