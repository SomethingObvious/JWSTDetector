from __future__ import annotations

import numpy as np

from jwstdetector.significance import bh_q, conformal_p


def test_conformal_p_counts_ties_and_the_test_point() -> None:
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    p = conformal_p(np.array([0.5, 2.0, 4.0, 9.0, np.nan]), cal)
    assert np.allclose(p, [5 / 5, 4 / 5, 2 / 5, 1 / 5, 1.0]), p


def test_conformal_p_is_uniform_for_exchangeable_scores() -> None:
    # Over many draws, P(p <= a) is at most a when test and calibration share a law.
    rng = np.random.default_rng(0)
    p = np.concatenate([
        conformal_p(rng.normal(size=1), rng.normal(size=99)) for _ in range(4000)
    ])
    for a in (0.05, 0.1, 0.5):
        assert np.mean(p <= a) <= a + 0.015, (a, np.mean(p <= a))


def test_bh_q_matches_the_step_up_rule() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.005])
    q = bh_q(p)
    # Sorted, p is 0.005 0.01 0.03 0.04 0.2 and m p / i is 0.025 0.025 0.05 0.05 0.2.
    assert np.allclose(q, [0.025, 0.05, 0.05, 0.2, 0.025]), q
    for alpha in (0.02, 0.025, 0.05, 0.1):
        ranked = np.sort(p)
        passing = np.nonzero(ranked <= alpha * np.arange(1, 6) / 5)[0]
        k = int(passing[-1]) + 1 if passing.size else 0
        assert (q <= alpha).sum() == k, alpha
