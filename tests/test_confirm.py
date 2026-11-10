from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

import pytest

from jwst_peculiar.confirm import main, sign_flip_p, total

if TYPE_CHECKING:
    from pathlib import Path


def test_a_gain_on_every_visit_gives_the_smallest_p() -> None:
    assert sign_flip_p([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]) == 1 / 1024


def test_sign_flip_matches_counting_the_patterns_by_hand() -> None:
    d = [12, -3, 7, 0, -5, 20]
    seen = sum(d)
    as_high = sum(
        sum(s * x for s, x in zip(signs, d, strict=True)) >= seen
        for signs in itertools.product((1, -1), repeat=len(d))
    )
    assert sign_flip_p(d) == as_high / 64
    assert sign_flip_p([3, -1]) == 0.5
    assert sign_flip_p([0, 0, 0]) == 1.0


def test_sign_flip_refuses_no_visits() -> None:
    with pytest.raises(ValueError, match="1 to 20 visits, not 0"):
        sign_flip_p([])


def _bench(folder: Path, found: list[int], beats_all: int = 0) -> Path:
    folder.mkdir()
    rows = ["detector,kind,mag,planted,found,beats_all"]
    rows += [f"knn,all,{22 + i},60,{n},{beats_all}" for i, n in enumerate(found)]
    rows += ["knn,arc,22,15,99,0", "pixels,all,22,60,99,0"]
    (folder / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def test_total_counts_one_detector_over_every_magnitude(tmp_path: Path) -> None:
    bench = _bench(tmp_path / "v", [30, 20, 5], beats_all=2)
    assert total(bench, "found") == 55
    assert total(bench, "beats_all") == 6
    with pytest.raises(ValueError, match="no rows for subspace"):
        total(bench, "found", "subspace")


def test_a_candidate_ahead_on_every_visit_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [str(_bench(tmp_path / f"d{i}", [100 + i])) for i in range(10)]
    tried = [str(_bench(tmp_path / f"c{i}", [110 + 2 * i])) for i in range(10)]
    assert main(["--default", *base, "--candidate", *tried, "--alpha", "0.0125"]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["default_total"] == 1045
    assert verdict["candidate_total"] == 1190
    assert verdict["p"] == 1 / 1024
    assert verdict["passed"] is True
    # The same benches the other way round can't pass.
    main(["--default", *tried, "--candidate", *base])
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_the_two_sides_have_to_pair_up(tmp_path: Path) -> None:
    one = str(_bench(tmp_path / "d", [1]))
    with pytest.raises(SystemExit, match="pair up visit by visit"):
        main(["--default", one, one, "--candidate", one])
