FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install '.[ecapa]'

RUN useradd --create-home --uid 10001 app && mkdir /models && chown app /models
USER app

EXPOSE 8000
CMD ["uvicorn", "voice_verification.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
