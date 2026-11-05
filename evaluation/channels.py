"""Controlled degradations applied to 16 kHz audio before embedding.

Each condition changes one thing, so comparing two conditions on the same
trials isolates its effect. These are simulations: a real phone network adds
packet loss, echo cancellation, AGC and handset effects that are not modelled.
"""

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000
Condition = Callable[[np.ndarray], np.ndarray]


def narrowband(x: np.ndarray) -> np.ndarray:
    """Telephone band: 16 kHz -> 8 kHz, 300-3400 Hz band-pass, back to 16 kHz.

    The embedding model expects 16 kHz, which is what a deployed service would
    also do with phone audio; the upsampling restores the rate, not the lost band.
    """
    x8 = resample_poly(x, 1, 2)
    sos = butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")
    return resample_poly(sosfiltfilt(sos, x8), 2, 1).astype(np.float32)


def mulaw_8bit(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 mu-law companding with 8-bit quantisation (continuous-curve approximation)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return (np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu).astype(np.float32)


def g711(x: np.ndarray) -> np.ndarray:
    """Narrowband channel followed by mu-law coding: a classic landline call."""
    return mulaw_8bit(narrowband(x))


def add_noise(
    x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Mix ``noise`` into ``x`` at ``snr_db`` (power over the whole clip).

    A random excerpt of ``noise`` is used; it is tiled if shorter than ``x``.
    """
    if noise.size < x.size:
        noise = np.tile(noise, int(np.ceil(x.size / noise.size)))
    start = rng.integers(0, noise.size - x.size + 1)
    n = noise[start : start + x.size]
    p_signal = np.mean(x.astype(np.float64) ** 2)
    p_noise = np.mean(n.astype(np.float64) ** 2)
    if p_noise == 0:
        return x.copy()
    scale = np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    return (x + scale * n).astype(np.float32)


CONDITIONS: dict[str, Condition] = {
    "clean": lambda x: x,
    "narrowband": narrowband,
    "g711": g711,
}
