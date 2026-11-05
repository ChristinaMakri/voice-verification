"""Error-rate metrics for threshold calibration.

Decision rule everywhere: accept when ``score >= threshold``.
  FAR(t) = P(non-target score >= t)   -- impostor accepted
  FRR(t) = P(target score < t)        -- genuine user rejected
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    far: float
    frr: float


def error_rates(target: np.ndarray, nontarget: np.ndarray, threshold: float) -> OperatingPoint:
    target, nontarget = np.asarray(target), np.asarray(nontarget)
    return OperatingPoint(
        threshold=float(threshold),
        far=float(np.mean(nontarget >= threshold)),
        frr=float(np.mean(target < threshold)),
    )


def compute_eer(target: np.ndarray, nontarget: np.ndarray) -> OperatingPoint:
    """Equal error rate: the candidate threshold where FAR and FRR are closest."""
    target, nontarget = np.sort(np.asarray(target)), np.sort(np.asarray(nontarget))
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("need both target and non-target scores")
    candidates = np.unique(np.concatenate([target, nontarget, [np.inf]]))
    frr = np.searchsorted(target, candidates, side="left") / target.size
    far = 1.0 - np.searchsorted(nontarget, candidates, side="left") / nontarget.size
    i = int(np.argmin(np.abs(far - frr)))
    return OperatingPoint(threshold=float(candidates[i]), far=float(far[i]), frr=float(frr[i]))


def threshold_at_far(nontarget: np.ndarray, max_far: float) -> float:
    """Lowest threshold whose FAR does not exceed ``max_far``."""
    if not 0.0 <= max_far <= 1.0:
        raise ValueError("max_far must be in [0, 1]")
    desc = np.sort(np.asarray(nontarget))[::-1]
    if desc.size == 0:
        raise ValueError("need non-target scores")
    allowed = int(np.floor(max_far * desc.size))
    if allowed >= desc.size:
        return float(desc[-1])
    return float(np.nextafter(desc[allowed], np.inf))
