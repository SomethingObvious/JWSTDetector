from __future__ import annotations

import numpy as np

from jwstdetector.vet import flag, significance


def _planes(*bright_in: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(3, 280, 280))
    # A source near the top left of the tile as scored, which is row 260 or so in
    # FITS order since the scored tile is flipped top to bottom.
    for i in bright_in:
        planes[i, 258:263, 18:23] += 40.0
    return planes


def test_a_source_in_every_filter_is_not_flagged() -> None:
    planes = _planes(0, 1, 2)
    sigs = [significance(p, (1, 1), 28) for p in planes]
    assert min(sigs) > 20
    assert flag(sigs) == ""


def test_a_source_in_one_filter_is_flagged() -> None:
    sigs = [significance(p, (1, 1), 28) for p in _planes(2)]
    assert sigs[2] > 20
    assert flag(sigs) == "one filter"


def test_the_patch_is_found_through_the_flip() -> None:
    # The same patch counted from the bottom, without the flip, is empty sky.
    planes = _planes(0, 1, 2)
    assert flag([significance(p, (26, 1), 28) for p in planes]) == "no filter"


def test_bright_sources_are_not_flagged_for_their_colour() -> None:
    # A red galaxy far above the noise in two filters and well seen in the third.
    assert flag([480.0, 160.0, 40.0]) == ""
    assert flag([50.0, 48.0, 4.7]) == ""
