from collections.abc import Sequence

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("cannot normalise a zero or non-finite vector")
    return (v / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of the normalised enrollment embeddings, re-normalised."""
    if not embeddings:
        raise ValueError("no embeddings")
    return l2_normalize(np.mean([l2_normalize(e) for e in embeddings], axis=0))
