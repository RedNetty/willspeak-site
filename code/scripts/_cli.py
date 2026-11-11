from __future__ import annotations

import os
import re
import sys
import traceback
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAUSE_ALWAYS = False


class _Tee:
    def __init__(self, stream, f, console_filter=None):
        self.stream, self.f = stream, f
        self.console_filter = console_filter
        self.pending = ""

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        if self.console_filter is None:
            with suppress(OSError, ValueError):
                self.stream.write(s)
        else:
            self.pending += s
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self._emit(line + "\n")
        return len(s)

    def _emit(self, line):
        if self.console_filter(line):
            with suppress(OSError, ValueError):
                self.stream.write(line)

    def flush(self):
        if self.pending:
            self._emit(self.pending)
            self.pending = ""
        with suppress(OSError, ValueError):
            self.stream.flush()
        self.f.flush()

    def isatty(self):
        try:
            return self.stream.isatty()
        except (OSError, ValueError):
            return False


class NemoConsole:
    def __init__(self):
        self.skip_indented = False

    def __call__(self, line: str) -> bool:
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        if self.skip_indented and (not line.strip() or line.startswith((" ", "\t"))):
            return False
        self.skip_indented = False
        notices = (
            "[NeMo I ",
            "If you intend to do training or fine-tuning, please call",
            "If you intend to do validation, please call",
            "NumbaPerformanceWarning: Grid size",
            "OneLogger: Setting error_handling_strategy",
            "No exporters were provided.",
            "Seed set to ",
            "Using bfloat16 Automatic Mixed Precision",
            "Using 16bit Automatic Mixed Precision",
            "GPU available:", "TPU available:", "HPU available:",
            "LOCAL_RANK:", "`Trainer(val_check_interval=",
            "To properly utilize them, you should set `torch.set_float32_matmul_precision",
        )
        if any(notice in line for notice in notices):
            self.skip_indented = True
            return False
        return True


def run(main, name: str, console_filter_factory=None) -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        try:
            return main() or 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    logdir = Path(os.environ.get("WILLSPEAK_LOG_DIR") or (ROOT / "logs"))
    logdir.mkdir(parents=True, exist_ok=True)
    logpath = logdir / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(logpath, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"# {name} argv={sys.argv[1:]} cwd={os.getcwd()}\n")
        sys.stdout = _Tee(original_stdout, lf, console_filter_factory() if console_filter_factory else None)
        sys.stderr = _Tee(original_stderr, lf, console_filter_factory() if console_filter_factory else None)
        try:
            code = main() or 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except KeyboardInterrupt:
            code = 130
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr

    print(f"\n(full log saved to {logpath})")
    if os.name == "nt" and sys.stdin.isatty() and (code != 0 or PAUSE_ALWAYS):
        with suppress(EOFError, OSError):
            input("\nPress Enter to close this window…")
    return code
