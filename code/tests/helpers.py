import io

import numpy as np
import soundfile as sf


def fake_wav(seconds=1.6, sr=48000, level=0.09, noise=6e-4, clip=False):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    env = (np.abs(np.sin(2 * np.pi * 3.1 * t)) > 0.28).astype(float)
    sig = (
        level
        * env
        * (np.sin(2 * np.pi * 130 * t) + 0.5 * np.sin(2 * np.pi * 720 * t) + 0.25 * np.sin(2 * np.pi * 1240 * t))
        / 1.75
    )
    sig += noise * np.random.randn(len(t))
    if clip:
        sig = np.clip(sig * 14, -1, 1)
    b = io.BytesIO()
    sf.write(b, sig.astype("float32"), sr, format="WAV", subtype="FLOAT")
    return b.getvalue()
