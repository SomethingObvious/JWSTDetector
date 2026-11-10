from __future__ import annotations

import numpy as np

from jwst_peculiar.vet import flag, hot_cell, significance


def _planes(*bright_in: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(3, 280, 280))
    # Top left of the tile as scored, which is near row 260 in FITS order since the
    # scored tile is flipped top to bottom.
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


def test_a_filter_with_no_data_counts_against_the_source() -> None:
    assert flag([float("nan"), 2.0, 1.0]) == "no filter"
    assert flag([30.0, float("nan"), 1.0]) == "one filter"


def test_the_hot_patch_is_one_the_tile_scores() -> None:
    scores = np.zeros((28, 28))
    scores[0, 3] = 9.0  # an object cut off by the tile's edge
    scores[10, 12] = 5.0
    owned = np.zeros((28, 28), dtype=bool)
    owned[7:21, 7:21] = True
    assert hot_cell(scores, owned) == (10, 12)
