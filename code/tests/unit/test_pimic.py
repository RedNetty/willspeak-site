from __future__ import annotations

import asyncio
import shlex
import sys

from willspeak import pimic


def _command(script: str) -> str:
    return " ".join(shlex.quote(a) for a in (sys.executable, "-c", script))


def test_stream_fans_out_one_capture_and_stops_with_the_last_listener(monkeypatch):
    # Writes 0.2 s of PCM, then keeps the process alive like a real microphone.
    script = "import sys,time; sys.stdout.buffer.write(b'\\x01\\x00'*9600); sys.stdout.flush(); time.sleep(30)"
    monkeypatch.setenv("WILLSPEAK_PI_MIC_CMD", _command(script))

    async def run():
        mic = pimic.PiMic()
        a = await mic.subscribe()
        b = await mic.subscribe()
        proc = mic._proc
        assert proc is not None and mic.status()["listeners"] == 2
        got = {"a": 0, "b": 0}

        async def take(name, q):
            gen = mic.stream(q)  # the server closes it when the page disconnects
            async for chunk in gen:
                got[name] += len(chunk)
                if got[name] >= 19200:
                    break
            await gen.aclose()

        await asyncio.wait_for(asyncio.gather(take("a", a), take("b", b)), 10)
        assert got == {"a": 19200, "b": 19200}
        assert mic._proc is None and proc.returncode is not None  # killed when the last page left
        assert mic.status()["listeners"] == 0

    asyncio.run(run())


def test_a_capture_that_fails_ends_the_stream_and_reports_why(monkeypatch):
    script = "import sys; sys.stderr.write('audio open error: Device or resource busy'); sys.exit(1)"
    monkeypatch.setenv("WILLSPEAK_PI_MIC_CMD", _command(script))

    async def run():
        mic = pimic.PiMic()
        q = await mic.subscribe()
        chunks = [c async for c in mic.stream(q)]
        assert chunks == []
        assert "busy" in mic.status()["error"]

    asyncio.run(asyncio.wait_for(run(), 10))


def test_capture_command_prefers_an_explicit_device(monkeypatch):
    monkeypatch.delenv("WILLSPEAK_PI_MIC_CMD", raising=False)
    monkeypatch.setenv("WILLSPEAK_PI_MIC", "plughw:CARD=Device,DEV=0")
    argv, env, source = pimic.capture_command()
    assert argv[0] == "arecord" and argv[-2:] == ["-D", "plughw:CARD=Device,DEV=0"]
    assert str(pimic.RATE) in argv and source == "plughw:CARD=Device,DEV=0"
