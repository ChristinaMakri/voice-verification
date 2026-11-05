"""Channel bandwidth: detection (which threshold applies) and simulation.

A telephone line passes roughly 300-3400 Hz. Audio that went through one has
almost no energy above 4 kHz, even after it is resampled to 16 kHz, while
microphone speech always has some (fricatives, breath, room noise). The
detector measures that high-band share. Its cut-off comes from this physical
argument and was fixed before it was measured on any data.
"""

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

WIDEBAND = "wideband"
NARROWBAND = "narrowband"
CHANNELS = (WIDEBAND, NARROWBAND)

HIGH_BAND_HZ = 4000.0
# Wideband speech puts roughly -10 to -30 dB of its energy above 4 kHz;
# band-limited audio leaves only filter leakage, typically below -50 dB.
NARROWBAND_MAX_DB = -35.0


def high_band_ratio_db(samples: np.ndarray, sample_rate: int) -> float:
    """Share of signal energy above 4 kHz, in dB (-inf if there is none)."""
    if sample_rate <= 2 * HIGH_BAND_HZ:
        return float("-inf")
    power = np.abs(np.fft.rfft(samples.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    total = power[freqs > 0].sum()
    high = power[freqs >= HIGH_BAND_HZ].sum()
    if total == 0 or high == 0:
        return float("-inf")
    return float(10 * np.log10(high / total))


def detect_channel(
    samples: np.ndarray, sample_rate: int, source_sample_rate: int | None = None
) -> tuple[str, float]:
    """Return (channel, high-band ratio in dB).

    A file recorded at 8 kHz or less cannot contain anything above 4 kHz, so it
    is narrowband by definition. Measuring it after upsampling would be
    unreliable: the resampling filter rolls off around 4 kHz, not at it, and
    lets some energy just below 4 kHz leak above.
    """
    if source_sample_rate is not None and source_sample_rate <= 2 * HIGH_BAND_HZ:
        return NARROWBAND, float("-inf")
    ratio = high_band_ratio_db(samples, sample_rate)
    return (NARROWBAND if ratio < NARROWBAND_MAX_DB else WIDEBAND), ratio


def simulate_narrowband(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Telephone band: 16 kHz -> 8 kHz, 300-3400 Hz band-pass, back to 16 kHz.

    The upsampling restores the rate the embedding model expects, not the lost
    band. A real phone network also adds codecs, packet loss and echo
    cancellation, which are not modelled here.
    """
    if sample_rate != 16000:
        raise ValueError("simulate_narrowband expects 16 kHz audio")
    x8 = resample_poly(samples, 1, 2)
    sos = butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")
    return resample_poly(sosfiltfilt(sos, x8), 2, 1).astype(np.float32)
