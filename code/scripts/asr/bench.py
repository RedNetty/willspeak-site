#!/usr/bin/env python3
"""Measure how fast this machine runs the Parakeet sentence recogniser."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

WORKER_MARKER = "@@BENCH_RESULT@@"
VERSION = "1.0.0"

BANDS = ((1.0, "GOOD"), (2.0, "USABLE"), (4.0, "SLUGGISH"), (float("inf"), "TOO SLOW"))

PI_DEMO_BOX_RTF = 0.7


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip("\x00 \n\t")
    except OSError:
        return ""


def run_cmd(args: list[str], timeout: int = 10) -> str:
    if shutil.which(args[0]) is None:
        return ""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def fmt_s(seconds: float) -> str:
    return "n/a" if math.isnan(seconds) else f"{seconds:.3f} s"


def verdict_for(decode_3s: float) -> str:
    for ceiling, name in BANDS:
        if decode_3s < ceiling:
            return name
    return "TOO SLOW"


FLAGS_OF_INTEREST = {
    "x86_64": ["avx512f", "avx2", "fma", "sse4_2", "sse4_1"],
    "aarch64": ["asimd", "asimdhp", "asimddp", "i8mm", "bf16", "sve", "crc32"],
}

ARM_PARTS = {
    "0xd0b": "Cortex-A76", "0xd08": "Cortex-A72", "0xd03": "Cortex-A53",
    "0xd05": "Cortex-A55", "0xd09": "Cortex-A73", "0xd0d": "Cortex-A77",
    "0xd41": "Cortex-A78", "0xd44": "Cortex-X1", "0xd4b": "Cortex-A78C",
}


def cpu_info() -> dict:
    info: dict = {"arch": platform.machine(), "logical_cores": os.cpu_count()}
    cpuinfo = read_text("/proc/cpuinfo")

    name = ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            name = line.split(":", 1)[1].strip()
            break
    if not name:
        part = ""
        for line in cpuinfo.splitlines():
            if line.lower().startswith("cpu part"):
                part = line.split(":", 1)[1].strip().lower()
                break
        name = ARM_PARTS.get(part, "")
    info["model"] = name or platform.processor() or "unknown"

    board = read_text("/proc/device-tree/model")
    if board:
        info["board"] = board

    flags: set[str] = set()
    for key in ("flags", "features"):
        match = re.search(rf"^{key}\s*:(.*)$", cpuinfo, re.MULTILINE | re.IGNORECASE)
        if match:
            flags |= set(match.group(1).split())
    wanted = FLAGS_OF_INTEREST.get(info["arch"], sorted(flags)[:12])
    info["simd"] = [f for f in wanted if f in flags]
    info["simd_checked"] = wanted

    lscpu = run_cmd(["lscpu"])
    for key, field in (("Core(s) per socket", "cores_per_socket"), ("Socket(s)", "sockets"),
                       ("Thread(s) per core", "threads_per_core"), ("CPU max MHz", "max_mhz")):
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", lscpu, re.MULTILINE)
        if match:
            value = match.group(1).strip()
            info[field] = float(value) if field == "max_mhz" else int(float(value))
    if "max_mhz" not in info:
        khz = read_text("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if khz.isdigit():
            info["max_mhz"] = int(khz) / 1000.0
    if "cores_per_socket" in info and "sockets" in info:
        info["physical_cores"] = info["cores_per_socket"] * info["sockets"]

    info["governor"] = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "unknown"
    return info


def memory_info() -> dict:
    out: dict = {}
    meminfo = read_text("/proc/meminfo")
    for key, field in (("MemTotal", "total_gb"), ("MemAvailable", "available_gb"),
                       ("SwapTotal", "swap_total_gb")):
        match = re.search(rf"^{key}:\s+(\d+) kB$", meminfo, re.MULTILINE)
        if match:
            out[field] = round(int(match.group(1)) / 1024 / 1024, 2)
    return out


def temperatures() -> dict:
    out: dict = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        raw = read_text(str(zone / "temp"))
        if raw.lstrip("-").isdigit():
            label = read_text(str(zone / "type")) or zone.name
            out[label] = round(int(raw) / 1000.0, 1)
    vc = run_cmd(["vcgencmd", "measure_temp"])
    match = re.search(r"([\d.]+)", vc)
    if match:
        out["vcgencmd"] = float(match.group(1))
    return out


THROTTLE_BITS = {
    0: "under-voltage NOW", 1: "arm frequency capped NOW", 2: "currently throttled NOW",
    3: "soft temperature limit NOW", 16: "under-voltage occurred since boot",
    17: "arm frequency capping occurred since boot", 18: "throttling occurred since boot",
    19: "soft temperature limit occurred since boot",
}


def throttle_state() -> dict:
    raw = run_cmd(["vcgencmd", "get_throttled"])
    match = re.search(r"0x([0-9a-fA-F]+)", raw)
    if not match:
        return {}
    value = int(match.group(1), 16)
    return {"raw": f"0x{value:x}",
            "flags": [text for bit, text in THROTTLE_BITS.items() if value & (1 << bit)]}


def storage_info(scratch: Path, megabytes: int) -> dict:
    out: dict = {}
    source = run_cmd(["findmnt", "-no", "SOURCE", "/"])
    if source:
        out["root_device"] = source
        out["root_media"] = ("SD card" if "mmcblk" in source else
                             "USB/SATA/NVMe" if re.search(r"/dev/(sd|nvme)", source) else "unknown")
    blob = os.urandom(1024 * 1024)
    path = scratch / "diskbench.bin"
    try:
        start = time.perf_counter()
        with open(path, "wb") as handle:
            handle.writelines(blob for _ in range(megabytes))
            handle.flush()
            os.fsync(handle.fileno())
        out["write_mb_s"] = round(megabytes / (time.perf_counter() - start), 1)

        start = time.perf_counter()
        with open(path, "rb") as handle:
            while handle.read(4 * 1024 * 1024):
                pass
        out["read_mb_s"] = round(megabytes / (time.perf_counter() - start), 1)
    except OSError as exc:
        out["error"] = str(exc)
    finally:
        path.unlink(missing_ok=True)
    return out


def compute_scores(quick: bool) -> dict:
    out: dict = {}
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy is not installed"}

    n = 512 if quick else 1024
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b
    best = min(_time_matmul(a, b) for _ in range(3))
    out["matmul_gflops"] = round(2 * n ** 3 / best / 1e9, 2)

    iterations = 1_000_000 if quick else 3_000_000
    start = time.perf_counter()
    total = 0
    for i in range(iterations):
        total += i ^ (i >> 3)
    elapsed = time.perf_counter() - start
    out["scalar_mops"] = round(iterations / elapsed / 1e6, 2)
    out["_scalar_checksum"] = total % 1000
    return out


def _time_matmul(a, b) -> float:
    start = time.perf_counter()
    a @ b
    return time.perf_counter() - start


def accelerator_info() -> dict:
    usb = run_cmd(["lsusb"]).lower()
    present = ("18d1:9302" in usb or "1a6e:089a" in usb or "google" in usb and "coral" in usb
               or Path("/dev/apex_0").exists())
    return {"edge_tpu_present": bool(present)}


def hardware_baseline(scratch: Path, quick: bool) -> dict:
    disk_mb = 64 if quick else 256
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "distro": _distro(),
        "python": platform.python_version(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "storage": storage_info(scratch, disk_mb),
        "compute": compute_scores(quick),
        "accelerator": accelerator_info(),
        "temps_idle": temperatures(),
        "throttled_before": throttle_state(),
    }


def _distro() -> str:
    text = read_text("/etc/os-release")
    match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.MULTILINE)
    return match.group(1) if match else ""


def find_reference_audio(model_dir: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit if explicit.is_file() else None
    candidates = sorted(model_dir.glob("test_wavs/*.wav")) + sorted(model_dir.glob("*.wav"))
    return candidates[0] if candidates else None


def build_offset_clips(reference: Path, seconds: float, count: int, scratch: Path) -> list[Path]:
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(reference), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        length = round(len(mono) * 16000 / rate)
        mono = np.interp(np.linspace(0, len(mono) - 1, length), np.arange(len(mono)), mono).astype("float32")
        rate = 16000

    want = int(seconds * rate)
    tiled = mono
    while len(tiled) < want + count * rate:
        tiled = np.concatenate([tiled, np.zeros(int(0.25 * rate), dtype="float32"), mono])

    out: list[Path] = []
    for index in range(count):
        start = int(index * 0.37 * rate) % max(len(tiled) - want, 1)
        path = scratch / f"post-{index}.wav"
        sf.write(str(path), tiled[start:start + want], rate, subtype="PCM_16")
        out.append(path)
    return out


def build_clips(reference: Path, durations: list[float], scratch: Path) -> dict[str, Path]:
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(reference), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        length = round(len(mono) * 16000 / rate)
        mono = np.interp(np.linspace(0, len(mono) - 1, length), np.arange(len(mono)), mono).astype("float32")
        rate = 16000

    gap = np.zeros(int(0.25 * rate), dtype="float32")
    out: dict[str, Path] = {}
    for seconds in durations:
        want = int(seconds * rate)
        buffer = mono
        while len(buffer) < want:
            buffer = np.concatenate([buffer, gap, mono])
        clip = buffer[:want]
        path = scratch / f"clip-{seconds:g}s.wav"
        sf.write(str(path), clip, rate, subtype="PCM_16")
        out[f"{seconds:g}"] = path
    return out


def worker_main(spec: dict) -> int:
    result: dict = {"threads": spec["threads"], "runs": {}, "texts": {}, "words": {},
                    "engine": "", "error": None}
    try:
        os.environ["WILLSPEAK_ASR_THREADS"] = str(spec["threads"])
        recognizer, engine, load_seconds = _build_recognizer(Path(spec["model_dir"]), spec["threads"])
        result["engine"] = engine
        result["load_s"] = round(load_seconds, 3)

        for label, path in spec["clips"].items():
            times: list[float] = []
            text = ""
            for index in range(spec["repeats"] + spec["warmup"]):
                start = time.perf_counter()
                text = recognizer(path)
                elapsed = time.perf_counter() - start
                if index >= spec["warmup"]:
                    times.append(round(elapsed, 4))
            result["runs"][label] = times
            result["words"][label] = len(text.split())
            result["texts"][label] = text[:200]
    except Exception as exc:                                    # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["peak_rss_mb"] = peak_rss_mb()
    print(WORKER_MARKER + json.dumps(result))
    return 0


def _build_recognizer(model_dir: Path, threads: int):
    try:
        from willspeak.asr.parakeet import ParakeetASR

        asr = ParakeetASR(model_dir)
        start = time.perf_counter()
        asr._ensure()
        load = time.perf_counter() - start
        return (lambda path: asr.transcribe(path)["text"]), "willspeak.asr.parakeet", load
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] app recognizer failed ({exc}); trying sherpa-onnx directly", file=sys.stderr)

    import sherpa_onnx
    files = {}
    for part in ("encoder", "decoder", "joiner"):
        for candidate in (model_dir / f"{part}.int8.onnx", model_dir / f"{part}.onnx"):
            if candidate.is_file():
                files[part] = str(candidate)
                break
    start = time.perf_counter()
    model = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=files["encoder"], decoder=files["decoder"], joiner=files["joiner"],
        tokens=str(model_dir / "tokens.txt"), num_threads=threads, sample_rate=16000,
        feature_dim=128, model_type="nemo_transducer", provider="cpu",
        decoding_method="greedy_search",
    )
    load = time.perf_counter() - start

    def decode(path: str) -> str:
        import soundfile as sf
        samples, _ = sf.read(path, dtype="float32")
        stream = model.create_stream()
        stream.accept_waveform(16000, samples)
        model.decode_stream(stream)
        return (stream.result.text or "").strip()

    return decode, "sherpa-onnx (standalone fallback)", load


def peak_rss_mb() -> float:
    try:
        import resource
        kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kilobytes / 1024, 1)
    except Exception:                                           # noqa: BLE001
        return float("nan")


def run_worker(spec: dict, timeout: int) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", json.dumps(spec)]
    env = dict(os.environ, WILLSPEAK_ASR_THREADS=str(spec["threads"]))
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              check=False, env=env, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s", "threads": spec["threads"], "runs": {}}
    for line in proc.stdout.splitlines():
        if line.startswith(WORKER_MARKER):
            return json.loads(line[len(WORKER_MARKER):])
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    return {"error": "worker produced no result:\n" + "\n".join(tail),
            "threads": spec["threads"], "runs": {}}


def proc_rss_mb(pid: int) -> float:
    match = re.search(r"^VmRSS:\s+(\d+) kB$", read_text(f"/proc/{pid}/status"), re.MULTILINE)
    return round(int(match.group(1)) / 1024, 1) if match else float("nan")


def proc_cpu_ticks(pid: int) -> float:
    fields = read_text(f"/proc/{pid}/stat").rsplit(")", 1)[-1].split()
    return float(fields[11]) + float(fields[12]) if len(fields) > 12 else float("nan")


class AppUnderTest:
    def __init__(self, port: int, scratch: Path, use_real_data: bool, threads: int,
                 model_dir: Path | None = None) -> None:
        self.port, self.scratch, self.use_real_data, self.threads = port, scratch, use_real_data, threads
        self.model_dir = model_dir
        self.process: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.log = scratch / "app.log"

    def start(self, timeout: int = 240) -> dict:
        env = dict(os.environ, WILLSPEAK_ASR_THREADS=str(self.threads), PORT=str(self.port))
        if self.model_dir is not None:
            env["WILLSPEAK_PARAKEET_DIR"] = str(self.model_dir)
        if not self.use_real_data:
            data_dir = self.scratch / "app-data"
            data_dir.mkdir(parents=True, exist_ok=True)
            env["WILLSPEAK_DATA"] = str(data_dir)
        with open(self.log, "w") as handle:
            self.process = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", str(self.port)],
                                            cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                return {"ok": False, "reason": f"server exited early; see {self.log}"}
            if self._get("/api/state") is not None:
                break
            time.sleep(0.5)
        else:
            self.stop()
            return {"ok": False, "reason": f"server did not answer within {timeout}s"}
        for page in ("/", "/live"):
            self._get(page)
        return {"ok": True, "pid": self.process.pid}

    def warm_model(self, wav: Path) -> bool:
        try:
            import urllib.request
            body, headers = _multipart_wav(wav)
            request = urllib.request.Request(self.base + "/api/live/transcribe", data=body,
                                             headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=300):
                return True
        except Exception:                                       # noqa: BLE001
            return False

    def sample_start(self) -> None:
        if self.process is None:
            return
        pid = self.process.pid
        self._samples = [proc_rss_mb(pid)]
        self._stop = threading.Event()
        self._t0 = time.time()
        self._ticks0 = proc_cpu_ticks(pid)

        def poll() -> None:
            while not self._stop.wait(0.4):
                self._samples.append(proc_rss_mb(pid))

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def sample_stop(self) -> dict:
        if self.process is None or not getattr(self, "_thread", None):
            return {}
        self._stop.set()
        self._thread.join(timeout=3)
        ticks = proc_cpu_ticks(self.process.pid) - self._ticks0
        hertz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        elapsed = max(time.time() - self._t0, 1e-6)
        clean = [value for value in self._samples if not math.isnan(value)]
        return {"rss_mb": max(clean) if clean else float("nan"),
                "cpu_percent_of_one_core": round(100 * ticks / hertz / elapsed, 1),
                "window_s": round(elapsed, 1)}

    def transcribe(self, wav: Path) -> dict:
        import urllib.request
        body, headers = _multipart_wav(wav)
        request = urllib.request.Request(self.base + "/api/live/transcribe", data=body,
                                         headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
        except Exception as exc:                                # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"wall_s": round(time.perf_counter() - start, 3),
                "server_latency_ms": payload.get("latency_ms"),
                "text": (payload.get("text") or "")[:120],
                "ok": payload.get("ok", True), "reason": payload.get("reason")}

    def _get(self, path: str):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as response:
                return response.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


def _multipart_wav(path: Path) -> tuple[bytes, dict]:
    boundary = "----willspeakbench" + os.urandom(8).hex()
    payload = path.read_bytes()
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="{path.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}",
                  "Content-Length": str(len(body))}


def summarise(runs: dict) -> dict:
    out: dict = {}
    for label, times in (runs or {}).items():
        if not times:
            continue
        seconds = float(label)
        out[label] = {
            "median_s": round(median(times), 3),
            "min_s": round(min(times), 3),
            "max_s": round(max(times), 3),
            "rtf": round(median(times) / seconds, 3),
            "drift_percent": round(100 * (times[-1] - times[0]) / times[0], 1) if times[0] else None,
            "repeats": len(times),
        }
    return out


def render(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    hardware, asr = report["hardware"], report["asr"]
    cpu = hardware["cpu"]

    add("=" * 72)
    add(f"willspeak Parakeet benchmark {VERSION}   label: {report['label']}")
    add(f"{report['started_at']}   host: {hardware['hostname']}")
    add("=" * 72)

    add("\nHARDWARE")
    if cpu.get("board"):
        add(f"  board          {cpu['board']}")
    add(f"  cpu            {cpu.get('model', '?')}  "
        f"({cpu.get('physical_cores') or cpu.get('logical_cores')} cores, "
        f"{cpu.get('logical_cores')} threads"
        + (f", max {cpu['max_mhz']:.0f} MHz" if cpu.get("max_mhz") else "") + ")")
    present = ", ".join(cpu.get("simd") or []) or "none of the ones checked"
    absent = [f for f in cpu.get("simd_checked", []) if f not in (cpu.get("simd") or [])]
    add(f"  simd           {present}")
    if absent:
        add(f"  simd missing   {', '.join(absent)}")
    add(f"  governor       {cpu.get('governor')}")
    memory = hardware["memory"]
    add(f"  memory         {memory.get('total_gb', '?')} GB total, "
        f"{memory.get('available_gb', '?')} GB available at start")
    add(f"  os             {hardware.get('distro') or hardware['os']}  (python {hardware['python']})")
    storage = hardware["storage"]
    add(f"  storage        {storage.get('root_device', '?')} ({storage.get('root_media', '?')}), "
        f"write {storage.get('write_mb_s', '?')} MB/s, read {storage.get('read_mb_s', '?')} MB/s")
    compute = hardware["compute"]
    add(f"  compute        matmul {compute.get('matmul_gflops', '?')} GFLOP/s, "
        f"scalar {compute.get('scalar_mops', '?')} Mops/s")
    add(f"  edge tpu       {'present' if hardware['accelerator']['edge_tpu_present'] else 'not present'}")
    if hardware.get("temps_idle"):
        add("  temp before    " + ", ".join(f"{k} {v} C" for k, v in hardware["temps_idle"].items()))
    if report.get("temps_after"):
        add("  temp after     " + ", ".join(f"{k} {v} C" for k, v in report["temps_after"].items()))
    for when, key in (("before", "throttled_before"), ("after", "throttled_after")):
        state = hardware.get(key) or report.get(key) or {}
        if state.get("flags"):
            add(f"  THROTTLE {when}  {state['raw']}: " + "; ".join(state["flags"]))
        elif state:
            add(f"  throttle {when}  {state['raw']} (clean)")

    if asr.get("skipped"):
        add("\nASR: SKIPPED -- " + (asr.get("error") or "--skip-asr was given"))
        return "\n".join(lines) + "\n"

    add("\nMODEL")
    add(f"  directory      {asr['model_dir']}")
    add(f"  engine         {asr.get('engine', '?')}")
    add(f"  audio          {asr['reference_audio']}")
    add(f"  transcript     \"{asr.get('sample_text', '')}\"")

    add("\nTHREAD SWEEP  (3 s of speech, median of the repeats)")
    add("  threads |  median  |    rtf  |  load  | peak rss")
    add("  --------+----------+---------+--------+---------")
    for entry in asr["thread_sweep"]:
        if entry.get("error"):
            add(f"  {entry['threads']:>7} |  ERROR: {entry['error'].splitlines()[0][:48]}")
            continue
        stats = entry["summary"].get(asr["headline_label"], {})
        add(f"  {entry['threads']:>7} | {stats.get('median_s', 0):7.3f}s | "
            f"{stats.get('rtf', 0):6.3f}x | {entry.get('load_s', 0):5.2f}s | "
            f"{entry.get('peak_rss_mb', 0):6.0f} MB")
    add(f"  -> best: {asr['best_threads']} threads. Set WILLSPEAK_ASR_THREADS={asr['best_threads']}.")
    if asr.get("more_threads_slower"):
        add("  -> more threads made it SLOWER. Past the point where the cores can be fed,")
        add("     they contend for memory bandwidth instead of sharing work.")

    add(f"\nLENGTH SWEEP  (at {asr['best_threads']} threads)")
    add("  audio |  median  |     rtf  | drift | transcript check")
    add("  ------+----------+----------+-------+------------------")
    for label in sorted(asr["length_sweep"], key=float):
        stats = asr["length_sweep"][label]
        drift = stats.get("drift_percent")
        add(f"  {label:>4}s | {stats['median_s']:7.3f}s | {stats['rtf']:7.3f}x | "
            f"{(f'{drift:+.1f}%' if drift is not None else '   n/a'):>6} | "
            f"{asr.get('words', {}).get(label, 0)} words")
    add("  rtf under 1.0 means the box transcribes faster than a person speaks.")
    add("  rtf should improve with length as the fixed per-utterance cost amortises.")

    if asr.get("loaded"):
        loaded = asr["loaded"]
        add("\nIDLE vs LOADED  (the same 3 s clip, with the willspeak server running)")
        add(f"  idle           {fmt_s(asr['headline_idle_s'])}")
        if loaded.get("error"):
            add(f"  loaded         ERROR: {loaded['error']}")
        else:
            add(f"  loaded         {fmt_s(loaded['median_s'])}   "
                f"penalty {loaded['penalty_percent']:+.0f}%")
            add(f"  server memory  {loaded.get('app_rss_mb', '?')} MB resident "
                f"(its own copy of the model -- a worst case, both processes hold one)")
            add(f"  server cpu     {loaded.get('app_cpu_percent_of_one_core', '?')}% of one core "
                "while it was only sitting there")
            if loaded.get("http_median_s"):
                add("")
                add(f"  end to end     {fmt_s(loaded['http_median_s'])} through the server's own HTTP path:")
                add(f"                 upload, ingest, QC, trim, decode and the database write "
                    f"({loaded['http_overhead_s']:+.3f}s")
                add("                 against the bare decode above). This is the number the device")
                add("                 delivers, and the decode alone is its floor.")
                if loaded.get("http_server_reported_median_s"):
                    add(f"                 of which the server called "
                        f"{loaded['http_server_reported_median_s']:.3f}s its own decode")
                add(f"  server cpu     {loaded.get('http_cpu_percent_of_one_core', '?')}% of one core "
                    "while it was doing that work")
                add("                 (over 100% means it used more than one core)")
            elif loaded.get("http_error"):
                add(f"  end to end     FAILED: {loaded['http_error']}")
            else:
                if (loaded.get("app_cpu_percent_of_one_core") or 0) < 5:
                    add("  NOTE: the server used under 5% of a core, so it was nearly idle. Use")
                    add("        --app-hold-seconds and open the Speak page during the pause.")
        add("  The browser on the Speak page is still not measured, and on a board this")
        add("  size it is often the bigger cost.")

    add("\nVERDICT")
    add(f"  headline       {fmt_s(asr['headline_s'])} for a 3 s utterance "
        f"({asr['headline_source']}, {asr['best_threads']} threads)")
    add(f"  band           {asr['verdict']}    "
        "(<1s GOOD | 1-2s USABLE | 2-4s SLUGGISH | >4s TOO SLOW)")
    add(f"  demo-box rule  idle rtf {asr['headline_idle_rtf']:.3f} -- "
        f"{'PASS' if asr['pi_demo_box'] else 'FAIL'} "
        f"(rule: under {PI_DEMO_BOX_RTF} idle and this box is the demo box)")
    for note in asr.get("notes", []):
        add(f"  note           {note}")

    add("\nWHAT THIS DOES NOT MEASURE")
    add("  Accuracy (that is WER, from scripts/asr/evaluate.py -- a box can be fast and")
    add("  useless), microphone and speaker latency, the matcher and text-to-speech, and")
    add("  a long thermal soak. The number here is the floor, not the total.")
    return "\n".join(lines) + "\n"


def render_compare(reports: list[dict]) -> str:
    lines = ["", f"COMPARISON of {len(reports)} runs", ""]
    header = ["metric"] + [r["label"][:22] for r in reports]
    rows = [header]

    def row(name, getter):
        rows.append([name] + [str(getter(r)) for r in reports])

    row("host", lambda r: r["hardware"]["hostname"])
    row("cpu", lambda r: (r["hardware"]["cpu"].get("board")
                          or r["hardware"]["cpu"].get("model", "?"))[:22])
    row("simd", lambda r: ",".join(r["hardware"]["cpu"].get("simd", []))[:22] or "-")
    row("governor", lambda r: r["hardware"]["cpu"].get("governor", "?"))
    row("matmul GFLOP/s", lambda r: r["hardware"]["compute"].get("matmul_gflops", "?"))
    row("best threads", lambda r: r["asr"].get("best_threads", "?"))
    row("3 s decode", lambda r: fmt_s(r["asr"].get("headline_s", float("nan"))))
    row("3 s rtf", lambda r: f"{r['asr'].get('headline_idle_rtf', float('nan')):.3f}")
    row("model load", lambda r: fmt_s(r["asr"].get("load_s", float("nan"))))
    row("peak rss MB", lambda r: r["asr"].get("peak_rss_mb", "?"))
    row("verdict", lambda r: r["asr"].get("verdict", "?"))

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for index, row_values in enumerate(rows):
        lines.append("  " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row_values)))
        if index == 0:
            lines.append("  " + "-+-".join("-" * width for width in widths))
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, nargs="+",
                        help="thread counts to sweep (default: 1 2 4, capped at the core count)")
    parser.add_argument("--durations", type=float, nargs="+", default=[1, 3, 5, 10, 30],
                        help="clip lengths in seconds for the length sweep (default: 1 3 5 10 30)")
    parser.add_argument("--repeats", type=int, default=5, help="timed decodes per configuration (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="untimed decodes first (default 1)")
    parser.add_argument("--model-dir", type=Path, help="Parakeet ONNX directory (default: the app's)")
    parser.add_argument("--audio", type=Path, help="reference speech WAV (default: the model's test audio)")
    parser.add_argument("--out", type=Path, default=Path("reports/bench-pi5"), help="report directory")
    parser.add_argument("--label", help="name for this run (default: host-timestamp)")
    parser.add_argument("--quick", action="store_true", help="short run: fewer repeats and lengths")
    parser.add_argument("--skip-asr", action="store_true", help="hardware baseline only")
    parser.add_argument("--with-app", action="store_true",
                        help="re-measure with the willspeak server running alongside")
    parser.add_argument("--app-port", type=int, default=8123, help="port for the server under test")
    parser.add_argument("--app-hold-seconds", type=float, default=0,
                        help="pause after the server is up so you can open the Speak page")
    parser.add_argument("--app-real-data", action="store_true",
                        help="let the server use the real data/ directory (default: a throwaway copy)")
    parser.add_argument("--compare", type=Path, nargs="+", help="print a table comparing saved .json reports")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


parse_args.QUICK_DEFAULT_DURATIONS = [1, 3, 5, 10, 30]


def main() -> int:
    args = parse_args()

    if args.worker_spec:
        return worker_main(json.loads(args.worker_spec))

    if args.compare:
        reports = []
        for path in args.compare:
            if not path.is_file():
                print(f"no such report: {path}", file=sys.stderr)
                return 1
            reports.append(json.loads(path.read_text()))
        print(render_compare(reports))
        return 0

    if args.quick:
        args.repeats = min(args.repeats, 2)
        if args.durations == parse_args.QUICK_DEFAULT_DURATIONS:
            args.durations = [1, 3, 10]

    cores = os.cpu_count() or 4
    threads = args.threads or sorted({t for t in (1, 2, 4, cores) if t <= max(cores, 1)})
    durations = sorted(set(args.durations))
    headline_label = "3" if 3 in durations else f"{durations[len(durations) // 2]:g}"
    label = args.label or f"{socket.gethostname()}-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"

    scratch = Path(tempfile.mkdtemp(prefix="willspeak-bench-"))
    report: dict = {"tool": "willspeak/scripts/asr/bench.py", "version": VERSION, "label": label,
                    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "argv": sys.argv[1:]}
    try:
        print(f"[bench] hardware baseline ({label}) ...")
        report["hardware"] = hardware_baseline(scratch, args.quick)

        if args.skip_asr:
            report["asr"] = {"skipped": True}
        else:
            report["asr"] = run_asr_suite(args, scratch, threads, durations, headline_label)

        report["temps_after"] = temperatures()
        report["throttled_after"] = throttle_state()
        report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

        text = render(report)
        print("\n" + text)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{label}.txt").write_text(text, encoding="utf-8")
        (args.out / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved {args.out / (label + '.txt')}")
        print(f"saved {args.out / (label + '.json')}")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_asr_suite(args, scratch: Path, threads: list[int], durations: list[float],
                  headline_label: str) -> dict:
    from willspeak.asr.parakeet import default_model_dir

    model_dir = (args.model_dir or default_model_dir()).resolve()
    asr: dict = {"model_dir": str(model_dir), "headline_label": headline_label, "notes": []}

    from willspeak.asr.parakeet import model_files

    if model_files(model_dir) is None:
        asr["error"] = (f"no Parakeet model in {model_dir}. Run: "
                        "python scripts/fetch_models.py --parakeet-only "
                        "(or bash deploy/pi5/setup.sh)")
        print("[bench] " + asr["error"], file=sys.stderr)
        asr["skipped"] = True
        return asr

    reference = find_reference_audio(model_dir, args.audio)
    if reference is None:
        asr["error"] = (f"no test audio found. Put a WAV in {model_dir / 'test_wavs'} or pass --audio. "
                        "python scripts/fetch_models.py --parakeet-only downloads the model's own "
                        "test recordings.")
        print("[bench] " + asr["error"], file=sys.stderr)
        asr["skipped"] = True
        return asr
    asr["reference_audio"] = str(reference)

    clips = build_clips(reference, durations, scratch)
    clips_str = {label: str(path) for label, path in clips.items()}
    budget = int(60 + args.repeats * sum(durations) * 8)

    print(f"[bench] thread sweep {threads} on {headline_label}s of speech ...")
    sweep = []
    for count in threads:
        spec = {"model_dir": str(model_dir), "threads": count, "repeats": args.repeats,
                "warmup": args.warmup, "clips": {headline_label: clips_str[headline_label]}}
        result = run_worker(spec, timeout=budget)
        result["summary"] = summarise(result.get("runs", {}))
        median_s = result["summary"].get(headline_label, {}).get("median_s")
        print(f"[bench]   {count} threads: {median_s if median_s is not None else result.get('error')}")
        sweep.append(result)
    asr["thread_sweep"] = sweep

    scored = [(e["summary"][headline_label]["median_s"], e["threads"]) for e in sweep
              if not e.get("error") and headline_label in e.get("summary", {})]
    if not scored:
        asr["error"] = "every thread configuration failed; see the thread sweep above"
        asr["best_threads"] = threads[0]
        return asr
    best_time, best_threads = min(scored)
    asr["best_threads"] = best_threads
    asr["more_threads_slower"] = best_threads < max(t for _, t in scored)

    print(f"[bench] length sweep at {best_threads} threads ...")
    spec = {"model_dir": str(model_dir), "threads": best_threads, "repeats": args.repeats,
            "warmup": args.warmup, "clips": clips_str}
    long_run = run_worker(spec, timeout=budget * 3)
    asr["engine"] = long_run.get("engine", "")
    asr["load_s"] = long_run.get("load_s", float("nan"))
    asr["peak_rss_mb"] = long_run.get("peak_rss_mb", float("nan"))
    asr["texts"] = long_run.get("texts", {})
    asr["words"] = long_run.get("words", {})
    asr["sample_text"] = asr["texts"].get(headline_label, "")
    asr["length_sweep"] = summarise(long_run.get("runs", {}))
    asr["raw_runs"] = long_run.get("runs", {})
    if long_run.get("error"):
        asr["notes"].append(f"length sweep error: {long_run['error']}")

    headline_idle = asr["length_sweep"].get(headline_label, {}).get("median_s", best_time)
    asr["headline_idle_s"] = headline_idle
    asr["headline_idle_rtf"] = headline_idle / float(headline_label)
    asr["headline_s"] = headline_idle
    asr["headline_source"] = "idle"

    if args.with_app:
        asr["loaded"] = measure_with_app(args, scratch, clips, headline_label, best_threads,
                                         model_dir, headline_idle, budget, reference)
        if not asr["loaded"].get("error"):
            asr["headline_s"] = asr["loaded"]["median_s"]
            asr["headline_source"] = "with the server running"

    asr["verdict"] = verdict_for(asr["headline_s"])
    asr["pi_demo_box"] = asr["headline_idle_rtf"] < PI_DEMO_BOX_RTF
    _add_notes(asr)
    return asr


def _max_rss(*samples: dict) -> float:
    values = [s.get("rss_mb", float("nan")) for s in samples]
    values = [v for v in values if v is not None and not math.isnan(v)]
    return max(values) if values else float("nan")


def measure_with_app(args, scratch: Path, clips: dict, headline_label: str, best_threads: int,
                     model_dir: Path, headline_idle: float, budget: int, reference: Path) -> dict:
    print(f"[bench] starting the willspeak server on port {args.app_port} ...")
    app = AppUnderTest(args.app_port, scratch, args.app_real_data, best_threads, model_dir)
    started = app.start()
    if not started.get("ok"):
        return {"error": started.get("reason", "could not start the server")}
    try:
        print("[bench]   server up; sending one utterance so it loads its own model ...")
        warmed = app.warm_model(clips[headline_label])
        if args.app_hold_seconds:
            print(f"[bench]   holding {args.app_hold_seconds:g}s -- open "
                  f"http://127.0.0.1:{args.app_port}/live now")
            time.sleep(args.app_hold_seconds)

        spec = {"model_dir": str(model_dir), "threads": best_threads, "repeats": args.repeats,
                "warmup": args.warmup, "clips": {headline_label: str(clips[headline_label])}}
        app.sample_start()
        result = run_worker(spec, timeout=budget)
        usage = app.sample_stop()
        if result.get("error"):
            return {"error": result["error"]}
        summary = summarise(result["runs"])[headline_label]

        print("[bench]   timing the same audio through the server's /api/live/transcribe ...")
        posts = []
        app.sample_start()
        for wav in build_offset_clips(reference, float(headline_label), args.repeats, scratch):
            posts.append(app.transcribe(wav))
        http_usage = app.sample_stop()
        accepted = [p for p in posts if p.get("ok") and p.get("wall_s")]

        loaded = {
            "median_s": summary["median_s"], "rtf": summary["rtf"],
            "penalty_percent": round(100 * (summary["median_s"] - headline_idle) / headline_idle, 1),
            "app_rss_mb": _max_rss(usage, http_usage),
            "app_cpu_percent_of_one_core": usage.get("cpu_percent_of_one_core"),
            "model_warmed_in_server": warmed, "peak_rss_mb": result.get("peak_rss_mb"),
            "http_posts": posts,
        }
        if accepted:
            loaded["http_median_s"] = round(median([p["wall_s"] for p in accepted]), 3)
            server_ms = [p["server_latency_ms"] for p in accepted if p.get("server_latency_ms")]
            if server_ms:
                loaded["http_server_reported_median_s"] = round(median(server_ms) / 1000, 3)
            loaded["http_overhead_s"] = round(loaded["http_median_s"] - summary["median_s"], 3)
            loaded["http_cpu_percent_of_one_core"] = http_usage.get("cpu_percent_of_one_core")
        else:
            loaded["http_error"] = "; ".join(
                str(p.get("reason") or p.get("error") or "rejected") for p in posts[:3]) or "no response"
        return loaded
    finally:
        app.stop()


def _add_notes(asr: dict) -> None:
    notes = asr["notes"]
    lengths = asr.get("length_sweep", {})
    drifts = [s.get("drift_percent") for s in lengths.values() if s.get("drift_percent") is not None]
    if drifts and max(drifts) > 10:
        notes.append(f"drift up to {max(drifts):+.1f}% across repeats -- the box slows down under "
                     "sustained work. Check temperature, power supply and airflow.")
    ordered = sorted(lengths.items(), key=lambda kv: float(kv[0]))
    if len(ordered) >= 2 and ordered[-1][1]["rtf"] > ordered[0][1]["rtf"] * 1.15:
        notes.append("real-time factor got WORSE on longer clips, which points at memory or heat "
                     "rather than compute.")
    if asr.get("more_threads_slower"):
        notes.append("the fastest thread count is not the highest one -- do not set "
                     "WILLSPEAK_ASR_THREADS to the core count by assumption.")


if __name__ == "__main__":
    raise SystemExit(main())
