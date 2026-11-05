import re
from collections.abc import Sequence
from dataclasses import dataclass

from .audio import Audio, load_audio, speech_seconds
from .config import Settings
from .crypto import VoiceprintCipher
from .embedding import Embedder
from .errors import EnrollmentLimit, InsufficientSpeech, InvalidUserId, NotEnrolled
from .scoring import centroid, cosine
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
    score: float
    threshold: float
    speech_seconds: float
    spoof_check: str
    spoof_bonafide: bool


class VerificationService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VoiceprintStore,
        cipher: VoiceprintCipher,
        spoof_detector: SpoofDetector | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.cipher = cipher
        self.spoof = spoof_detector or NoSpoofDetector()

    def _load(self, data: bytes) -> tuple[Audio, float]:
        audio = load_audio(data, self.embedder.sample_rate, self.settings.max_audio_s)
        return audio, speech_seconds(audio.samples, audio.sample_rate)

    @staticmethod
    def _check_user_id(user_id: str) -> None:
        if not _USER_ID.fullmatch(user_id):
            raise InvalidUserId("user_id must be 1-128 chars of [A-Za-z0-9_.-]")

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
        stored = self.store.list(user_id, self.embedder.model_id)
        if not stored:
            raise NotEnrolled(f"no voiceprint for user '{user_id}'")

        audio, speech = self._load(recording)
        if speech < self.settings.min_verify_speech_s:
            raise InsufficientSpeech(
                f"need {self.settings.min_verify_speech_s:.0f}s of speech, got {speech:.1f}s"
            )

        spoof = self.spoof.check(audio.samples, audio.sample_rate)
        reference = centroid([self.cipher.decrypt(user_id, b) for b in stored])
        score = cosine(self.embedder.embed(audio.samples), reference)
        threshold = self.settings.threshold
        return VerifyResult(
            user_id=user_id,
            verified=spoof.is_bonafide and score >= threshold,
            score=round(score, 4),
            threshold=threshold,
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
