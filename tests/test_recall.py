from __future__ import annotations

import numpy as np

from jwstdetector.recall import group_interval, separation


def testseparation_is_great_circle_arcseconds() -> None:
    assert np.isclose(separation(10.0, -72.0, (10.0, -73.0)), 3600.0)
    # A degree of RA at Dec 60 is half a degree on the sky.
    assert np.isclose(separation(11.0, 60.0, (10.0, 60.0)), 1800.0, rtol=1e-3)


def testgroup_interval_draws_whole_groups() -> None:
    # Two groups whose members agree, so every draw's mean is 10, 20 or 30.
    lo, hi = group_interval([("a", 10.0), ("a", 10.0), ("b", 30.0)], n=2000)
    assert lo == 10.0
    assert hi == 30.0
