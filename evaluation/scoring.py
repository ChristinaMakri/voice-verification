"""Score trials with the service's own logic and summarise with uncertainty.

Scores use ``voice_verification.scoring`` (centroid of the enrollment
embeddings, cosine similarity), so the evaluation measures the deployed
decision rule, not a re-implementation of it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from voice_verification.metrics import compute_eer, error_rates, threshold_at_far
from voice_verification.scoring import centroid

from .protocol import Protocol

# A statistic maps (target_scores, nontarget_scores) -> float.
Statistic = Callable[[np.ndarray, np.ndarray], float]


def score_trials(
    protocol: Protocol,
    enroll_emb: dict[str, np.ndarray],
    probe_emb: dict[str, np.ndarray],
) -> pd.DataFrame:
    models = {
        spk: centroid([enroll_emb[u] for u in g.utt_id])
        for spk, g in protocol.enroll.groupby("speaker")
    }
    t = protocol.trials
    model_mat = np.stack([models[m] for m in t.model])
    probe_mat = np.stack([probe_emb[u] for u in t.probe_utt])
    # Embeddings are L2-normalised, so the row-wise dot product is the cosine.
    return t.assign(score=np.einsum("ij,ij->i", model_mat, probe_mat))


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float

    def as_dict(self, digits: int = 4) -> dict:
        return {k: round(v, digits) for k, v in vars(self).items()}


def _split_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df.score[df.target].to_numpy(), df.score[~df.target].to_numpy()


def bootstrap(
    scored: pd.DataFrame,
    statistic: Statistic,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Speaker-level bootstrap.

    Trials of one speaker are correlated (same voiceprint, same voice), so
    resampling individual trials would understate the uncertainty. We resample
    *model speakers* with replacement and keep all of their trials.
    """
    groups = [_split_scores(g) for _, g in scored.groupby("model", sort=True)]
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        tgt = np.concatenate([groups[i][0] for i in pick])
        non = np.concatenate([groups[i][1] for i in pick])
        values[b] = statistic(tgt, non)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return Interval(value=float(statistic(*_split_scores(scored))), low=float(lo), high=float(hi))


def eer_rate(tgt: np.ndarray, non: np.ndarray) -> float:
    op = compute_eer(tgt, non)
    return (op.far + op.frr) / 2


def summarise(scored: pd.DataFrame, target_far: float = 0.01, n_boot: int = 1000) -> dict:
    """Pick thresholds on calib, report everything on test."""
    calib = scored[scored.split == "calib"]
    test = scored[scored.split == "test"]
    c_tgt, c_non = _split_scores(calib)

    thresholds = {
        "eer_calib": compute_eer(c_tgt, c_non).threshold,
        f"far{target_far:g}_calib": threshold_at_far(c_non, target_far),
    }

    def at(threshold: float) -> dict:
        return {
            "threshold": round(float(threshold), 4),
            "far": bootstrap(test, lambda t, n: error_rates(t, n, threshold).far, n_boot).as_dict(),
            "frr": bootstrap(test, lambda t, n: error_rates(t, n, threshold).frr, n_boot).as_dict(),
        }

    t_tgt, t_non = _split_scores(test)
    return {
        "test_trials": {"target": int(t_tgt.size), "nontarget": int(t_non.size)},
        "test_eer": bootstrap(test, eer_rate, n_boot).as_dict(),
        "test_score_means": {
            "target": round(float(t_tgt.mean()), 4),
            "nontarget": round(float(t_non.mean()), 4),
        },
        "operating_points": {name: at(t) for name, t in thresholds.items()},
    }
