"""Compute embeddings once per (model, condition, recording) and cache them.

Embedding is the only slow step (~0.1 s per clip on CPU), so every experiment
reuses the cache and re-scoring takes seconds. The cache holds voiceprints of
real people: it lives under ``data/`` and must never be published.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from voice_verification.audio import load_audio
from voice_verification.embedding import Embedder

from .channels import CONDITIONS


class EmbeddingCache:
    def __init__(self, cache_dir: Path, embedder: Embedder):
        self.embedder = embedder
        slug = embedder.model_id.replace("/", "__")
        self.dir = Path(cache_dir) / slug
        self.dir.mkdir(parents=True, exist_ok=True)

    def _file(self, condition: str) -> Path:
        return self.dir / f"{condition}.npz"

    def _load(self, condition: str) -> dict[str, np.ndarray]:
        f = self._file(condition)
        if not f.exists():
            return {}
        with np.load(f) as z:
            return dict(zip(z["ids"].tolist(), z["emb"], strict=True))

    def _save(self, condition: str, table: dict[str, np.ndarray]) -> None:
        ids = np.array(sorted(table))
        emb = np.stack([table[i] for i in ids]) if len(ids) else np.zeros((0, 0), np.float32)
        tmp = self._file(condition).with_suffix(".tmp.npz")
        np.savez(tmp, ids=ids, emb=emb)
        tmp.replace(self._file(condition))

    def get(
        self, condition: str, items: Iterable[tuple[str, str]], save_every: int = 200
    ) -> dict[str, np.ndarray]:
        """``items`` are (utt_id, audio_path). Returns utt_id -> embedding."""
        if condition not in CONDITIONS:
            raise KeyError(f"unknown condition '{condition}'")
        degrade = CONDITIONS[condition]
        table = self._load(condition)
        todo = [(u, p) for u, p in dict(items).items() if u not in table]
        for i, (utt, path) in enumerate(todo, 1):
            audio = load_audio(Path(path).read_bytes(), self.embedder.sample_rate, 600)
            table[utt] = self.embedder.embed(degrade(audio.samples))
            if i % save_every == 0:
                self._save(condition, table)
                print(f"  [{condition}] {i}/{len(todo)}", file=sys.stderr)
        if todo:
            self._save(condition, table)
        return table
