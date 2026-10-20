"""Make clean synthesized speech sound like F1 team radio.

Team radio is narrow-band (roughly 300 Hz to 3.4 kHz), a bit compressed and
overdriven, has a bed of static, and opens and closes with a short beep. This is
done with numpy only, so it adds no dependency beyond what the project already uses.
"""

from __future__ import annotations

import numpy as np

LOW_HZ = 300.0
HIGH_HZ = 3400.0


def _tone(freq, seconds, sr, level):
    t = np.arange(int(sr * seconds)) / sr
    tone = np.sin(2 * np.pi * freq * t) * level
    fade = min(len(tone) // 4, int(sr * 0.005))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return tone


def radio_filter(samples, sample_rate: int, *, drive: float = 2.2, noise_level: float = 0.012,
                 beeps: bool = True, seed: int | None = None) -> np.ndarray:
    """Return a float32 mono signal in [-1, 1] that sounds like a pit-to-car radio call."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.astype(np.float32)
    sr = float(sample_rate)

    # Band-pass: smooth Butterworth-like magnitude response applied in the frequency domain.
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    safe = np.maximum(freqs, 1.0)
    mask = 1.0 / np.sqrt(1.0 + (LOW_HZ / safe) ** 8) / np.sqrt(1.0 + (freqs / HIGH_HZ) ** 8)
    y = np.fft.irfft(spectrum * mask, n=x.size)

    # Compress and lightly overdrive, like a radio mic driven hard.
    peak = np.max(np.abs(y)) or 1.0
    y = np.tanh(drive * y / peak) / np.tanh(drive)

    rng = np.random.default_rng(seed)
    y = y * 0.85 + rng.normal(0.0, noise_level, y.size)

    if beeps:
        gap = np.zeros(int(sr * 0.04))
        open_beep = np.concatenate([_tone(1250, 0.06, sr, 0.35), _tone(1650, 0.06, sr, 0.35)])
        close_beep = _tone(1250, 0.07, sr, 0.3)
        y = np.concatenate([open_beep, gap, y, gap, close_beep])
    return np.clip(y, -1.0, 1.0).astype(np.float32)
