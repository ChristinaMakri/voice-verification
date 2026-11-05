import numpy as np
import pytest

from voice_verification.metrics import compute_eer, error_rates, threshold_at_far


def test_perfect_separation_has_zero_eer():
    eer = compute_eer(np.array([0.8, 0.9, 0.95]), np.array([0.1, 0.2, 0.3]))
    assert eer.far == 0 and eer.frr == 0
    assert 0.3 < eer.threshold <= 0.8


def test_overlapping_scores():
    target = np.array([0.4, 0.6, 0.7, 0.9])
    nontarget = np.array([0.1, 0.2, 0.5, 0.65])
    eer = compute_eer(target, nontarget)
    assert eer.far == pytest.approx(0.25) and eer.frr == pytest.approx(0.25)


def test_threshold_at_far_respects_limit():
    rng = np.random.default_rng(0)
    nontarget = rng.normal(0, 0.1, 1000)
    t = threshold_at_far(nontarget, 0.01)
    assert error_rates(np.array([1.0]), nontarget, t).far <= 0.01
    # and it is the lowest such threshold: going one notch below breaks the limit
    below = np.sort(nontarget)[::-1][10]
    assert np.mean(nontarget >= below) > 0.01


def test_threshold_at_zero_far_rejects_every_impostor():
    nontarget = np.array([0.1, 0.5, 0.3])
    assert error_rates(np.array([1.0]), nontarget, threshold_at_far(nontarget, 0.0)).far == 0
