"""The box's own microphone, streamed to a browser on another computer.

A page opened from another machine (a laptop on the same hotspot) cannot use
that machine's microphone over plain http, and should not: the device's
microphone is the lavalier on the box. This module runs one capture process on
the box and fans its raw PCM (16-bit little-endian, mono, RATE Hz) out to every
connected page. The page turns it back into the same messages the recorder
worklet sends, so the Speak and Compare pages work unchanged.

The capture command is, in order of preference:
  WILLSPEAK_PI_MIC_CMD   a whole command line that writes raw s16le mono PCM at RATE
                       (for tests: ffmpeg -re -i clip.wav -f s16le -ar 48000 -ac 1 -)
  WILLSPEAK_PI_MIC       an ALSA device for arecord (e.g. plughw:CARD=Device,DEV=0)
  pulse                when the desktop's PipeWire/Pulse socket is reachable, so the
                       kiosk browser and this stream can share the microphone
  the first capture card arecord lists
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

RATE = 48000
CHUNK_BYTES = RATE * 2 // 50  # 20 ms
_QUEUE_CHUNKS = 50  # one second of backlog per page before it starts dropping


def _runtime_dir() -> Path | None:
    for candidate in (os.environ.get("XDG_RUNTIME_DIR"), f"/run/user/{os.getuid()}" if hasattr(os, "getuid") else None):
        if candidate and Path(candidate, "pulse", "native").exists():
            return Path(candidate)
    return None


def capture_cards() -> list[dict]:
    """USB/other capture cards from `arecord -l`, without the HDMI outputs."""
    if not shutil.which("arecord"):
        return []
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for m in re.finditer(r"^card (\d+): (\S+) \[([^\]]*)\], device (\d+): [^\[]*\[([^\]]*)\]", out, re.M):
        cards.append({"card": int(m[1]), "id": m[2], "name": m[3], "device": int(m[4]), "description": m[5]})
    return cards


def capture_command() -> tuple[list[str], dict, str]:
    """(argv, extra environment, human description) for the capture process."""
    custom = os.environ.get("WILLSPEAK_PI_MIC_CMD")
    if custom:
        return shlex.split(custom), {}, "custom command"
    base = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw",
            "--buffer-time=100000", "--period-time=20000"]
    device = os.environ.get("WILLSPEAK_PI_MIC")
    if device:
        return base + ["-D", device], {}, device
    runtime = _runtime_dir()
    if runtime is not None:
        return base + ["-D", "pulse"], {"XDG_RUNTIME_DIR": str(runtime)}, "pulse (shared with the desktop)"
    cards = capture_cards()
    if cards:
        c = cards[0]
        return base + ["-D", f"plughw:CARD={c['id']},DEV={c['device']}"], {}, f"{c['name']} (plughw)"
    return base, {}, "default"


class PiMic:
    """One capture process, shared by every page that is listening."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.error: str | None = None
        self.source: str | None = None

    def status(self) -> dict:
        cards = capture_cards()
        _, _, source = capture_command()
        return {"rate": RATE, "format": "s16le", "channels": 1,
                "running": self._proc is not None and self._proc.returncode is None,
                "listeners": len(self._subscribers), "source": self.source or source,
                "microphones": [c["name"] for c in cards], "error": self.error}

    async def _start(self) -> None:
        argv, extra, source = capture_command()
        env = {**os.environ, **extra}
        self.source, self.error = source, None
        self._proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        self._reader = asyncio.create_task(self._pump(self._proc))
        print(f"[willspeak] pi mic: started ({source}), pid {self._proc.pid}")

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while True:
                data = await proc.stdout.read(CHUNK_BYTES)
                if not data:
                    break
                for q in list(self._subscribers):
                    if q.full():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    q.put_nowait(data)
        finally:
            err = b""
            if proc.stderr is not None:
                try:
                    err = await asyncio.wait_for(proc.stderr.read(), 1)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    pass
            rc = await proc.wait()
            if self._proc is proc:
                self._proc = None
                if self._subscribers:
                    msg = err.decode(errors="replace").strip() or f"capture exited with code {rc}"
                    self.error = msg
                    print(f"[willspeak] pi mic: stopped: {msg}")
                for q in list(self._subscribers):
                    q.put_nowait(None)

    async def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            print("[willspeak] pi mic: stopped (no one listening)")
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_CHUNKS)
        async with self._lock:
            self._subscribers.add(q)
            if self._proc is None or self._proc.returncode is not None:
                try:
                    await self._start()
                except OSError as e:
                    self._subscribers.discard(q)
                    self.error = f"could not start the capture: {e}"
                    raise
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)
            if not self._subscribers:
                await self._stop()

    async def stream(self, q: asyncio.Queue):
        try:
            while True:
                data = await q.get()
                if data is None:
                    return
                yield data
        finally:
            await self.unsubscribe(q)
