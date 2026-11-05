import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .audio import Audio, load_audio, speech_seconds
from .calibration import Calibration, CalibrationError
from .channel import detect_channel
from .config import Settings
from .crypto import VoiceprintCipher
from .embedding import Embedder
from .errors import (
    EnrollmentLimit,
    InsufficientSpeech,
    InvalidUserId,
    NotCalibrated,
    NotEnrolled,
)
from .scoring import asnorm, centroid, cohort_stats, cosine
from .spoofing import NoSpoofDetector, SpoofDetector
from .storage import VoiceprintStore

# Opaque, pseudonymous ids only. Do not use national ids, phone numbers or
# names: the voiceprint table should not be enough to identify anyone.
_USER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class EnrollResult:
    user_id: str
    samples_added: int
    samples_total: int
    speech_seconds: float


@dataclass(frozen=True)
class VerifyResult:
    user_id: str
    verified: bool
    score: float  # the score compared with the threshold (AS-norm or raw)
    threshold: float
    score_type: str  # "asnorm" | "raw"
    raw_score: float  # cosine similarity, for diagnostics
    channel: str  # detected: "wideband" | "narrowband"
    calibrated: bool
    speech_seconds: float
    spoof_check: str
    spoof_bonafide: bool


@dataclass(frozen=True)
class Decision:
    score: float
    raw_score: float
    threshold: float
    score_type: str


class VerificationService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VoiceprintStore,
        cipher: VoiceprintCipher,
        spoof_detector: SpoofDetector | None = None,
        calibration: Calibration | None = None,
        cohort: np.ndarray | None = None,
    ):
        if calibration is not None:
            if calibration.model_id != embedder.model_id:
                raise CalibrationError(
                    f"calibration is for '{calibration.model_id}', "
                    f"embedder is '{embedder.model_id}'"
                )
            if calibration.score == "asnorm" and cohort is None:
                raise CalibrationError("AS-norm calibration needs its cohort")
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.cipher = cipher
        self.spoof = spoof_detector or NoSpoofDetector()
        self.calibration = calibration
        self.cohort = cohort

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def _load(self, data: bytes) -> tuple[Audio, float]:
        audio = load_audio(data, self.embedder.sample_rate, self.settings.max_audio_s)
        return audio, speech_seconds(audio.samples, audio.sample_rate)

    @staticmethod
    def _check_user_id(user_id: str) -> None:
        if not _USER_ID.fullmatch(user_id):
            raise InvalidUserId("user_id must be 1-128 chars of [A-Za-z0-9_.-]")

    def _require_calibration(self) -> None:
        if not self.calibrated and not self.settings.allow_uncalibrated:
            raise NotCalibrated(
                "service is not calibrated: set VV_CALIBRATION_PATH (see vv-calibrate), "
                "or VV_ALLOW_UNCALIBRATED=true for development only"
            )

    def decide(self, reference: np.ndarray, probe: np.ndarray, channel: str) -> Decision:
        """Score a probe embedding against a voiceprint and pick the threshold."""
        raw = cosine(probe, reference)
        cal = self.calibration
        if cal is None:
            return Decision(raw, raw, self.settings.threshold, "raw")
        if channel not in cal.thresholds:
            raise NotCalibrated(f"no threshold calibrated for {channel} audio")
        score = raw
        if cal.score == "asnorm":
            k = cal.cohort_top_k
            score = float(
                asnorm(
                    np.array([raw]),
                    cohort_stats(reference, self.cohort, k),
                    cohort_stats(probe, self.cohort, k),
                )[0]
            )
        return Decision(score, raw, cal.thresholds[channel], cal.score)

    def enroll(self, user_id: str, recordings: Sequence[bytes]) -> EnrollResult:
        self._check_user_id(user_id)
        if not recordings:
            raise InsufficientSpeech("at least one recording is required")

        model_id = self.embedder.model_id
        existing = self.store.count(user_id, model_id)
        if existing + len(recordings) > self.settings.max_samples_per_user:
            raise EnrollmentLimit(
                f"at most {self.settings.max_samples_per_user} samples per user "
                f"({existing} already stored)"
            )

        loaded = [self._load(r) for r in recordings]
        for i, (_, speech) in enumerate(loaded):
            if speech < self.settings.min_verify_speech_s:
                raise InsufficientSpeech(f"recording {i} has only {speech:.1f}s of speech")
        total_speech = sum(s for _, s in loaded)
        if existing == 0 and total_speech < self.settings.min_enroll_speech_s:
            raise InsufficientSpeech(
                f"first enrollment needs {self.settings.min_enroll_speech_s:.0f}s of speech, "
                f"got {total_speech:.1f}s"
            )

        # Embed everything before writing, so a failure stores nothing.
        blobs = [
            self.cipher.encrypt(user_id, self.embedder.embed(audio.samples)) for audio, _ in loaded
        ]
        self.store.add_many(user_id, model_id, blobs)
        return EnrollResult(
            user_id=user_id,
            samples_added=len(blobs),
            samples_total=existing + len(blobs),
            speech_seconds=round(total_speech, 2),
        )

    def verify(self, user_id: str, recording: bytes) -> VerifyResult:
        self._check_user_id(user_id)
        self._require_calibration()
        stored = self.store.list(user_id, self.embedder.model_id)
        if not stored:
            raise NotEnrolled(f"no voiceprint for user '{user_id}'")

        audio, speech = self._load(recording)
        if speech < self.settings.min_verify_speech_s:
            raise InsufficientSpeech(
                f"need {self.settings.min_verify_speech_s:.0f}s of speech, got {speech:.1f}s"
            )

        channel, _ = detect_channel(audio.samples, audio.sample_rate, audio.source_sample_rate)
        spoof = self.spoof.check(audio.samples, audio.sample_rate)
        reference = centroid([self.cipher.decrypt(user_id, b) for b in stored])
        d = self.decide(reference, self.embedder.embed(audio.samples), channel)
        return VerifyResult(
            user_id=user_id,
            verified=spoof.is_bonafide and d.score >= d.threshold,
            score=round(d.score, 4),
            threshold=round(d.threshold, 4),
            score_type=d.score_type,
            raw_score=round(d.raw_score, 4),
            channel=channel,
            calibrated=self.calibrated,
            speech_seconds=round(speech, 2),
            spoof_check=spoof.detector,
            spoof_bonafide=spoof.is_bonafide,
        )

    def status(self, user_id: str) -> int:
        self._check_user_id(user_id)
        return self.store.count(user_id, self.embedder.model_id)

    def delete(self, user_id: str) -> int:
        self._check_user_id(user_id)
        return self.store.delete(user_id)
