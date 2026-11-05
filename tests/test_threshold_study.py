import numpy as np
import pandas as pd
import pytest

from evaluation import threshold_study as ts
from voice_verification.metrics import threshold_at_far as reference_threshold


@pytest.mark.parametrize("alpha", [0.0, 0.001, 0.01, 0.2, 1.0])
def test_fast_threshold_matches_reference(alpha):
    rng = np.random.default_rng(0)
    for scores in [rng.normal(size=5000), np.round(rng.normal(size=500), 1)]:  # with ties
        assert ts.threshold_at_far(scores, alpha) == reference_threshold(scores, alpha)


def test_conservative_threshold_is_not_below_empirical():
    rng = np.random.default_rng(1)
    model = np.repeat(np.arange(40), 100)
    non = rng.normal(size=model.size) + rng.normal(scale=0.3, size=40)[model]
    emp = ts.threshold_at_far(non, 0.01)
    assert ts.conservative_threshold(non, model, 0.01, np.random.default_rng(0)) >= emp


def test_asnorm_is_zero_at_cohort_mean_and_scale_free():
    stats = np.array([[0.2, 0.05], [0.4, 0.1]])
    s = np.array([0.2, 0.4])
    np.testing.assert_allclose(ts.asnorm(s, stats, stats), [0.0, 0.0])
    # a probe one cohort-std above both means scores 1 regardless of the scale
    np.testing.assert_allclose(ts.asnorm(np.array([0.25, 0.5]), stats, stats), [1.0, 1.0])


def test_cohort_stats_uses_top_k():
    cohort = np.eye(4)[:, :4]
    emb = np.array([[1.0, 0.0, 0.0, 0.0]])
    mean, std = ts.cohort_stats(emb, cohort, k=2)[0]
    assert mean == pytest.approx(0.5) and std == pytest.approx(0.5)


def _summary(viol, frr, m0=0.4):
    rows = []
    for cond in ts.CONDITIONS:
        rows.append(
            dict(
                exp="E1",
                fit_cond=cond,
                eval_cond=cond,
                method="M0",
                n=118,
                alpha=ts.ALPHA,
                violation_rate=m0,
                median_frr=0.0,
            )
        )
        for m in ts.METHODS:
            for n in ts.N_FIT:
                rows.append(
                    dict(
                        exp="E1",
                        fit_cond=cond,
                        eval_cond=cond,
                        method=m,
                        n=n,
                        alpha=ts.ALPHA,
                        violation_rate=viol[m] + (80 - n) / 400,
                        median_frr=frr[m],
                    )
                )
        other = [c for c in ts.CONDITIONS if c != cond][0]
        for m in ts.METHODS:
            rows.append(
                dict(
                    exp="E3",
                    fit_cond=cond,
                    eval_cond=other,
                    method=m,
                    n=80,
                    alpha=ts.ALPHA,
                    violation_rate=0.9,
                    median_frr=0.1,
                )
            )
    return pd.DataFrame(rows)


def test_decide_picks_cheapest_safe_method():
    viol = {"M1": 0.6, "M2": 0.45, "M3": 0.48, "M4": 0.3}
    frr = {"M1": 0.03, "M2": 0.06, "M3": 0.04, "M4": 0.08}
    d = ts.decide(_summary(viol, frr))
    assert d["meets_safety"]["M1"]["clean"] is False  # 0.6 > 0.4 + 0.1
    assert d["recommended"] == "M3"  # safe and cheaper than M2, M4
    assert d["min_calibration_speakers"] == 60  # (80-60)/400 = 0.05 <= tolerance
    assert d["recalibrate_per_channel"] is True


def test_decide_recommends_nothing_when_none_safe():
    viol = dict.fromkeys(ts.METHODS, 0.9)
    d = ts.decide(_summary(viol, dict.fromkeys(ts.METHODS, 0.05)))
    assert d["recommended"] is None


def test_summarise_aggregates_per_group():
    rows = [
        dict(
            exp="E1",
            fit_cond="clean",
            eval_cond="clean",
            method="M1",
            n=80,
            alpha=0.01,
            rep=r,
            threshold=0.4,
            far=f,
            frr=0.05,
        )
        for r, f in enumerate([0.005, 0.009, 0.012, 0.02])
    ]
    s = ts.summarise(pd.DataFrame(rows))
    assert len(s) == 1
    assert s.violation_rate.iat[0] == 0.5
    assert s.median_frr.iat[0] == pytest.approx(0.05)
    assert s.bias_median_far_over_alpha.iat[0] == pytest.approx(1.05)
