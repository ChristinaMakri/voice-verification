class VerificationError(Exception):
    """Base class for errors that map to a client-facing HTTP status."""

    status_code = 400


class InvalidUserId(VerificationError):
    status_code = 422


class AudioError(VerificationError):
    status_code = 422


class InsufficientSpeech(VerificationError):
    status_code = 422


class NotEnrolled(VerificationError):
    status_code = 404


class EnrollmentLimit(VerificationError):
    status_code = 409


class NotCalibrated(VerificationError):
    status_code = 503
