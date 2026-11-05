import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from . import __version__
from .config import Settings
from .crypto import VoiceprintCipher
from .errors import AudioError, VerificationError
from .service import VerificationService

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _build_default_service(settings: Settings) -> VerificationService:
    from .embedding import EcapaEmbedder
    from .storage import PostgresStore

    return VerificationService(
        settings=settings,
        embedder=EcapaEmbedder(settings.model_cache_dir, settings.device),
        store=PostgresStore(settings.database_url),
        cipher=VoiceprintCipher(settings.encryption_key),
    )


def create_app(service: VerificationService | None = None) -> FastAPI:
    """Build the app. Pass ``service`` to inject fakes (tests); otherwise the
    ECAPA model and Postgres store are created from env settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or _build_default_service(Settings())
        yield
        close = getattr(app.state.service.store, "close", None)
        if close:
            close()

    app = FastAPI(title="voice-verification", version=__version__, lifespan=lifespan)

    def get_service(request: Request) -> VerificationService:
        return request.app.state.service

    def require_api_key(
        key: str | None = Depends(_api_key_header),
        svc: VerificationService = Depends(get_service),
    ) -> None:
        if key is None or not secrets.compare_digest(key, svc.settings.api_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def read_upload(upload: UploadFile, limit: int) -> bytes:
        data = upload.file.read(limit + 1)
        if len(data) > limit:
            raise AudioError(f"file '{upload.filename}' exceeds {limit} bytes")
        return data

    @app.exception_handler(VerificationError)
    async def _verification_error(_: Request, exc: VerificationError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/voiceprints/{user_id}", status_code=201, dependencies=[Depends(require_api_key)])
    def enroll(
        user_id: str,
        files: list[UploadFile] = File(..., description="One or more recordings of the user"),
        svc: VerificationService = Depends(get_service),
    ) -> dict:
        limit = svc.settings.max_upload_bytes
        return asdict(svc.enroll(user_id, [read_upload(f, limit) for f in files]))

    @app.post("/v1/voiceprints/{user_id}/verify", dependencies=[Depends(require_api_key)])
    def verify(
        user_id: str,
        file: UploadFile = File(..., description="Recording to check against the voiceprint"),
        svc: VerificationService = Depends(get_service),
    ) -> dict:
        return asdict(svc.verify(user_id, read_upload(file, svc.settings.max_upload_bytes)))

    @app.get("/v1/voiceprints/{user_id}", dependencies=[Depends(require_api_key)])
    def status(user_id: str, svc: VerificationService = Depends(get_service)) -> dict:
        samples = svc.status(user_id)
        return {"user_id": user_id, "enrolled": samples > 0, "samples": samples}

    @app.delete("/v1/voiceprints/{user_id}", dependencies=[Depends(require_api_key)])
    def delete(user_id: str, svc: VerificationService = Depends(get_service)) -> dict:
        return {"user_id": user_id, "deleted": svc.delete(user_id)}

    return app
