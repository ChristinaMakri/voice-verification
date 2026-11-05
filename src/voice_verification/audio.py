"""Audio decoding, normalisation and a lightweight speech-duration estimate."""

import io
from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import AudioError


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32, mono
    sample_rate: int
    source_sample_rate: int  # rate of the uploaded file, before resampling

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(data: bytes, target_sr: int, max_seconds: float) -> Audio:
    """Decode any format libsndfile supports (WAV incl. mu-law/A-law, FLAC, OGG),
    downmix to mono and resample to ``target_sr``.

    Telephony audio (8 kHz) is upsampled; that does not restore the missing
    bandwidth, which is why the threshold must be calibrated per channel.
    """
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise AudioError(f"unsupported or corrupt audio: {exc}") from exc

    if samples.size == 0:
        raise AudioError("audio is empty")
    if samples.shape[0] / sr > max_seconds:
        raise AudioError(f"audio longer than {max_seconds:.0f}s")

    mono = samples.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // g, sr // g)
    return Audio(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sample_rate=target_sr,
        source_sample_rate=sr,
    )


def speech_seconds(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    rel_db: float = 35.0,
    floor_db: float = -50.0,
) -> float:
    """Estimate how many seconds contain speech-level energy.

    A frame counts if its RMS is within ``rel_db`` of the loudest frame and above
    an absolute floor. Crude on purpose: it only gates out silent or near-silent
    uploads. Swap in a neural VAD (e.g. Silero) for noisy channels.
    """
    frame = int(sample_rate * frame_ms / 1000)
    n = len(samples) // frame
    if n == 0:
        return 0.0
    frames = samples[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-12)
    threshold = max(db.max() - rel_db, floor_db)
    return float(np.count_nonzero(db > threshold) * frame / sample_rate)
