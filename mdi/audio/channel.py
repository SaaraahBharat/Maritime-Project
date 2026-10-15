"""Phase 1/2: simulate a marine VHF radio channel on clean (TTS) speech.

Chain: resample to 8 kHz -> 300-3400 Hz band-pass -> optional background noise
(engine/wind recordings, e.g. MUSAN) -> AWGN at a target SNR -> soft clipping
(over-driven mic) -> random dropouts (fading) -> squelch tail burst.

Every parameter is recorded in the returned `ChannelParams`, so ASR word error
rate can later be plotted against SNR and the other degradations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

VHF_SR = 8000


@dataclass
class ChannelParams:
    snr_db: float
    clip_gain: float
    dropout_count: int
    squelch_tail: bool
    background_snr_db: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def resample(x: np.ndarray, sr: int, target_sr: int = VHF_SR) -> np.ndarray:
    if sr == target_sr:
        return x
    g = np.gcd(sr, target_sr)
    return signal.resample_poly(x, target_sr // g, sr // g)


def bandpass(x: np.ndarray, sr: int = VHF_SR, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    sos = signal.butter(6, [lo, min(hi, sr / 2 - 1)], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix `noise` into `x` so the result has the requested SNR."""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    scale = _rms(x) / (_rms(noise) * 10 ** (snr_db / 20))
    return x + scale * noise


def soft_clip(x: np.ndarray, gain: float) -> np.ndarray:
    return np.tanh(gain * x) / np.tanh(gain)


def dropouts(x: np.ndarray, rng: np.random.Generator, count: int, sr: int = VHF_SR) -> np.ndarray:
    y = x.copy()
    for _ in range(count):
        length = int(rng.uniform(0.03, 0.25) * sr)
        start = int(rng.integers(0, max(1, len(y) - length)))
        y[start : start + length] *= rng.uniform(0.0, 0.15)
    return y


def squelch_tail(rng: np.random.Generator, sr: int = VHF_SR) -> np.ndarray:
    """Short burst of hiss heard when the transmitter un-keys."""
    n = int(rng.uniform(0.08, 0.2) * sr)
    return bandpass(rng.normal(0, 0.3, n), sr) * np.linspace(1, 0, n)


def simulate_vhf(
    x: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    snr_db: float | None = None,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, ChannelParams]:
    """Degrade mono float audio `x` (any sample rate). Returns 8 kHz audio."""
    y = resample(np.asarray(x, dtype=np.float64), sr)
    y = y / (np.max(np.abs(y)) + 1e-9)

    bg_snr = None
    if background is not None:
        bg_snr = float(rng.uniform(0, 15))
        y = add_noise(y, resample(background, sr), bg_snr)

    y = bandpass(y)
    snr = float(rng.uniform(-5, 20)) if snr_db is None else float(snr_db)
    y = add_noise(y, rng.normal(0, 1, len(y)), snr)
    y = bandpass(y)  # radio filters the noise too

    gain = float(rng.uniform(1.0, 6.0))
    y = soft_clip(y / (np.max(np.abs(y)) + 1e-9), gain)
    n_drop = int(rng.integers(0, 4))
    y = dropouts(y, rng, n_drop)

    tail = bool(rng.random() < 0.7)
    if tail:
        y = np.concatenate([y, squelch_tail(rng)])

    y = 0.95 * y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32), ChannelParams(snr, gain, n_drop, tail, bg_snr)
