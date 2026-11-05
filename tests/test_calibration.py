"""Channel detection, cohort files, calibration, and the fail-safe service behaviour."""

import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voice_verification.api import create_app
from voice_verification.calibration import (
    Calibration,
    CalibrationError,
    calibrate,
    load_scoring_config,
)
from voice_verification.channel import (
    NARROWBAND,
    WIDEBAND,
    detect_channel,
    high_band_ratio_db,
    simulate_narrowband,
)
from voice_verification.cohort import CohortError, file_sha256, load_cohort, save_cohort
from voice_verification.config import Settings
from voice_verification.crypto import VoiceprintCipher, generate_key
from voice_verification.service import VerificationService
from voice_verification.storage import InMemoryStore

from .conftest import FakeEmbedder, synth_voice, to_wav

SR = 16000
AUTH = {"X-API-Key": "test-key"}


def broadband_voice(pitch, seconds=3.0, seed=0):
    """Synthetic voice plus broadband noise, so it has energy above 4 kHz like real speech."""
    rng = np.random.default_rng(seed)
    x = synth_voice(pitch, seconds, seed=seed)
    return (x + 0.03 * rng.standard_normal(x.size)).astype(np.float32)


# --- channel detection ----------------------------------------------------------
def test_wideband_and_simulated_narrowband_are_told_apart():
    x = broadband_voice(200)
    assert detect_channel(x, SR)[0] == WIDEBAND
    assert detect_channel(simulate_narrowband(x), SR)[0] == NARROWBAND


def test_8khz_source_upsampled_is_narrowband():
    from voice_verification.audio import load_audio

    x8 = broadband_voice(200)[::2]  # crude 8 kHz signal
    audio = load_audio(to_wav(x8, sr=8000), SR, 60)
    assert audio.source_sample_rate == 8000
    assert detect_channel(audio.samples, SR, audio.source_sample_rate)[0] == NARROWBAND


def test_high_band_ratio_of_silence_is_minus_inf():
    assert high_band_ratio_db(np.zeros(SR, dtype=np.float32), SR) == float("-inf")


# --- cohort file -------------------------------------------------------------------
def test_cohort_roundtrip_and_model_binding(tmp_path):
    cipher = VoiceprintCipher(generate_key())
    cohort = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)
    path = tmp_path / "cohort.npz"
    save_cohort(path, cohort, "model-a", cipher)
    np.testing.assert_array_equal(load_cohort(path, "model-a", cipher), cohort)
    with pytest.raises(CohortError):
        load_cohort(path, "model-b", cipher)
    assert cohort.tobytes() not in path.read_bytes()  # stored encrypted


# --- calibration -------------------------------------------------------------------
def make_speakers(root, pitches, n_files=6):
    speakers = {}
    for i, pitch in enumerate(pitches):
        d = root / f"spk{i:02d}"
        d.mkdir(parents=True)
        files = []
        for k in range(n_files):
            f = d / f"{k}.wav"
            sf.write(f, broadband_voice(pitch, seed=100 * i + k), SR)
            files.append(f)
        speakers[d.name] = files
    return speakers


@pytest.fixture
def cohort():
    emb = FakeEmbedder()
    return np.stack([emb.embed(broadband_voice(p, seed=7)) for p in np.linspace(130, 900, 12)])


def test_calibrate_produces_per_channel_asnorm_thresholds(tmp_path, cohort):
    speakers = make_speakers(tmp_path, np.linspace(150, 850, 8))
    cal, report = calibrate(speakers, FakeEmbedder(), cohort, "abc", top_k=5, simulate=True)
    assert cal.score == "asnorm" and cal.cohort_top_k == 5
    assert set(cal.thresholds) == {WIDEBAND, NARROWBAND}
    assert cal.sources == {WIDEBAND: "recorded", NARROWBAND: "simulated"}
    assert report[WIDEBAND]["speakers"] == 8
    assert "warning" in report[WIDEBAND]  # fewer than 40 speakers


def test_calibrate_without_cohort_uses_raw_scores(tmp_path):
    speakers = make_speakers(tmp_path, [200, 500, 800])
    cal, _ = calibrate(speakers, FakeEmbedder(), None)
    assert cal.score == "raw" and cal.cohort_sha256 is None
    assert set(cal.thresholds) == {WIDEBAND}


def test_calibration_file_roundtrip_and_validation(tmp_path):
    cal = Calibration(
        model_id="m",
        score="raw",
        target_far=0.01,
        thresholds={WIDEBAND: 0.4},
        speakers={WIDEBAND: 50},
        sources={WIDEBAND: "recorded"},
    )
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal
    with pytest.raises(CalibrationError):
        Calibration(
            model_id="m",
            score="asnorm",
            target_far=0.01,
            thresholds={WIDEBAND: 1.0},
            speakers={},
            sources={},
        )  # asnorm without cohort info
    with pytest.raises(CalibrationError):
        Calibration(
            model_id="m",
            score="raw",
            target_far=0.01,
            thresholds={"satellite": 1.0},
            speakers={},
            sources={},
        )


def test_load_scoring_config_rejects_a_different_cohort(tmp_path, cohort):
    key = generate_key()
    cipher = VoiceprintCipher(key)
    cpath = tmp_path / "cohort.npz"
    save_cohort(cpath, cohort, FakeEmbedder.model_id, cipher)
    cal = Calibration(
        model_id=FakeEmbedder.model_id,
        score="asnorm",
        target_far=0.01,
        thresholds={WIDEBAND: 1.0},
        speakers={WIDEBAND: 50},
        sources={WIDEBAND: "recorded"},
        cohort_sha256=file_sha256(cpath),
        cohort_top_k=5,
    )
    cal.save(tmp_path / "cal.json")
    settings = Settings(
        encryption_key=key,
        api_key="k",
        calibration_path=str(tmp_path / "cal.json"),
        cohort_path=str(cpath),
    )
    loaded_cohort, loaded_cal = load_scoring_config(settings, cipher, FakeEmbedder.model_id)
    assert loaded_cal == cal and loaded_cohort.shape == cohort.shape

    save_cohort(cpath, cohort[:-1], FakeEmbedder.model_id, cipher)  # someone swaps the file
    with pytest.raises(CalibrationError, match="differs"):
        load_scoring_config(settings, cipher, FakeEmbedder.model_id)


# --- service behaviour -------------------------------------------------------------
def build_service(tmp_path, cohort, calibrated=True, channels=(WIDEBAND, NARROWBAND)):
    key = generate_key()
    settings = Settings(encryption_key=key, api_key="test-key", min_enroll_speech_s=6.0)
    cal = None
    if calibrated:
        speakers = make_speakers(tmp_path / "calib", np.linspace(150, 850, 8))
        cal, _ = calibrate(speakers, FakeEmbedder(), cohort, "sha", top_k=5, simulate=True)
        cal = Calibration(
            **{
                **json.loads(json.dumps(cal.__dict__)),
                "thresholds": {c: cal.thresholds[c] for c in channels},
            }
        )
    return VerificationService(
        settings,
        FakeEmbedder(),
        InMemoryStore(),
        VoiceprintCipher(key),
        calibration=cal,
        cohort=cohort,
    )


def enroll_files(pitch):
    return [
        ("files", (f"{k}.wav", to_wav(broadband_voice(pitch, 4.0, seed=k)), "audio/wav"))
        for k in range(2)
    ]


def probe(pitch, narrow=False):
    x = broadband_voice(pitch, 3.0, seed=42)
    return {"file": ("p.wav", to_wav(simulate_narrowband(x) if narrow else x), "audio/wav")}


def test_uncalibrated_service_refuses_to_verify(tmp_path, cohort):
    with TestClient(create_app(build_service(tmp_path, cohort, calibrated=False))) as c:
        assert c.get("/health").json()["calibrated"] is False
        assert c.post("/v1/voiceprints/a", headers=AUTH, files=enroll_files(300)).status_code == 201
        r = c.post("/v1/voiceprints/a/verify", headers=AUTH, files=probe(300))
        assert r.status_code == 503
        assert "not calibrated" in r.json()["detail"]


def test_calibrated_service_uses_asnorm_and_channel_threshold(tmp_path, cohort):
    svc = build_service(tmp_path, cohort)
    with TestClient(create_app(svc)) as c:
        assert c.get("/health").json()["channels"] == [NARROWBAND, WIDEBAND]
        c.post("/v1/voiceprints/a", headers=AUTH, files=enroll_files(300))

        wide = c.post("/v1/voiceprints/a/verify", headers=AUTH, files=probe(300)).json()
        assert wide["channel"] == WIDEBAND and wide["score_type"] == "asnorm"
        assert wide["calibrated"] is True
        assert wide["threshold"] == round(svc.calibration.thresholds[WIDEBAND], 4)
        assert wide["verified"] is True

        narrow = c.post("/v1/voiceprints/a/verify", headers=AUTH, files=probe(300, True)).json()
        assert narrow["channel"] == NARROWBAND
        assert narrow["threshold"] == round(svc.calibration.thresholds[NARROWBAND], 4)

        impostor = c.post("/v1/voiceprints/a/verify", headers=AUTH, files=probe(800)).json()
        assert impostor["verified"] is False


def test_channel_without_threshold_is_refused(tmp_path, cohort):
    svc = build_service(tmp_path, cohort, channels=(WIDEBAND,))
    with TestClient(create_app(svc)) as c:
        c.post("/v1/voiceprints/a", headers=AUTH, files=enroll_files(300))
        r = c.post("/v1/voiceprints/a/verify", headers=AUTH, files=probe(300, narrow=True))
        assert r.status_code == 503
        assert "narrowband" in r.json()["detail"]


def test_service_rejects_calibration_for_another_model(cohort):
    cal = Calibration(
        model_id="other",
        score="raw",
        target_far=0.01,
        thresholds={WIDEBAND: 0.4},
        speakers={WIDEBAND: 50},
        sources={WIDEBAND: "recorded"},
    )
    with pytest.raises(CalibrationError):
        VerificationService(
            Settings(encryption_key=generate_key(), api_key="k"),
            FakeEmbedder(),
            InMemoryStore(),
            VoiceprintCipher(generate_key()),
            calibration=cal,
        )
