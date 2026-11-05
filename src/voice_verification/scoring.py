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


def cohort_stats(embeddings: np.ndarray, cohort: np.ndarray, k: int) -> np.ndarray:
    """Mean and std of each embedding's top-k cosine scores against the cohort.

    ``embeddings`` is (d,) or (n, d), L2-normalised; ``cohort`` is (m, d).
    Returns an (n, 2) array of [mean, std].
    """
    if cohort.shape[0] < k:
        raise ValueError(f"cohort has {cohort.shape[0]} entries, fewer than k={k}")
    s = np.atleast_2d(embeddings) @ cohort.T
    top = -np.sort(-s, axis=1)[:, :k]
    return np.stack([top.mean(axis=1), top.std(axis=1)], axis=1)


def asnorm(score: np.ndarray, model_stats: np.ndarray, probe_stats: np.ndarray) -> np.ndarray:
    """Adaptive symmetric score normalisation (AS-norm).

    Expresses a raw score as "how far above what this voiceprint and this
    probe usually score against strangers", averaged over both sides.
    """
    return 0.5 * (
        (score - model_stats[:, 0]) / model_stats[:, 1]
        + (score - probe_stats[:, 0]) / probe_stats[:, 1]
    )
