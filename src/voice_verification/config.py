from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set via an env var prefixed with ``VV_``."""

    model_config = SettingsConfigDict(env_prefix="VV_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://vv:vv@localhost:5432/voiceprints"
    # base64-encoded 32-byte key; generate one with `vv-keygen`.
    encryption_key: str
    # Clients must send this in the X-API-Key header.
    api_key: str

    # Cosine-similarity acceptance threshold. The default suits ECAPA on clean
    # wideband speech; calibrate it on your own channel with `vv-calibrate`.
    threshold: float = 0.25

    sample_rate: int = 16000
    min_enroll_speech_s: float = 8.0
    min_verify_speech_s: float = 2.0
    max_audio_s: float = 60.0
    max_upload_bytes: int = 10 * 1024 * 1024
    max_samples_per_user: int = 10

    model_cache_dir: str = "./.models"
    device: str = "cpu"
