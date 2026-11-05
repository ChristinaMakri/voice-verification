import numpy as np
import pytest

from voice_verification.audio import load_audio, speech_seconds
from voice_verification.errors import AudioError

from .conftest import synth_voice, to_wav


def test_telephony_8k_is_resampled_to_16k():
    audio = load_audio(to_wav(synth_voice(200, 2.0, sr=8000), sr=8000), 16000, 60)
    assert audio.sample_rate == 16000
    assert audio.samples.dtype == np.float32
    assert audio.duration_s == pytest.approx(2.0, abs=0.01)


def test_mulaw_wav_decodes():
    audio = load_audio(to_wav(synth_voice(200, 1.0, sr=8000), sr=8000, subtype="ULAW"), 16000, 60)
    assert audio.duration_s == pytest.approx(1.0, abs=0.01)


def test_stereo_is_downmixed():
    mono = synth_voice(200, 1.0)
    audio = load_audio(to_wav(np.stack([mono, mono], axis=1)), 16000, 60)
    assert audio.samples.ndim == 1


def test_corrupt_audio_rejected():
    with pytest.raises(AudioError):
        load_audio(b"definitely not audio", 16000, 60)


def test_too_long_audio_rejected():
    with pytest.raises(AudioError, match="longer"):
        load_audio(to_wav(synth_voice(200, 3.0)), 16000, max_seconds=2)


def test_speech_seconds_ignores_trailing_silence():
    x = synth_voice(200, 3.0, silence_s=2.0)
    assert speech_seconds(x, 16000) == pytest.approx(3.0, abs=0.1)


def test_speech_seconds_of_silence_is_zero():
    assert speech_seconds(np.zeros(16000, dtype=np.float32), 16000) == 0.0
