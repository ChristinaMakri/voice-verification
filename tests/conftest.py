"""Test fixtures. All audio is synthetic: no real voices are used or shipped."""

import io

import numpy as np
import pytest
import soundfile as sf

from voice_verification.config import Settings
from voice_verification.crypto import VoiceprintCipher, generate_key
from voice_verification.scoring import l2_normalize
from voice_verification.service import VerificationService
from voice_verification.storage import InMemoryStore


def synth_voice(
    pitch_hz: float, seconds: float, sr: int = 16000, seed: int = 0, silence_s: float = 0.0
) -> np.ndarray:
    """A harmonic tone + noise as a stand-in "voice"; pitch acts as the speaker identity."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * pitch_hz * k * t) / k for k in range(1, 5))
    x = 0.3 * x / np.max(np.abs(x)) + 0.01 * rng.standard_normal(t.size)
    if silence_s:
        x = np.concatenate([x, np.zeros(int(silence_s * sr))])
    return x.astype(np.float32)


def to_wav(samples: np.ndarray, sr: int = 16000, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


class FakeEmbedder:
    """Log-band magnitude spectrum: same pitch -> high cosine, different pitch -> low."""

    model_id = "fake-spectrum-v1"
    sample_rate = 16000

    def embed(self, samples: np.ndarray) -> np.ndarray:
        power = np.abs(np.fft.rfft(samples)) ** 2
        freqs = np.fft.rfftfreq(samples.size, 1 / self.sample_rate)
        edges = np.geomspace(50, 4000, 65)
        bands = [
            power[(freqs >= lo) & (freqs < hi)].sum()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        return l2_normalize(np.sqrt(np.array(bands, dtype=np.float32)))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        encryption_key=generate_key(),
        api_key="test-key",
        threshold=0.8,
        min_enroll_speech_s=6.0,
        min_verify_speech_s=2.0,
        max_samples_per_user=3,
    )


@pytest.fixture
def service(settings: Settings) -> VerificationService:
    return VerificationService(
        settings=settings,
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        cipher=VoiceprintCipher(settings.encryption_key),
    )
