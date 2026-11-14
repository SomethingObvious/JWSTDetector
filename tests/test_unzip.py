from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import pytest

from jwst_peculiar.unzip import decompress_dir

if TYPE_CHECKING:
    from pathlib import Path


def test_each_archive_is_expanded_and_removed(tmp_path: Path) -> None:
    (tmp_path / "a.fits.gz").write_bytes(gzip.compress(b"SIMPLE  = T"))
    assert decompress_dir(tmp_path) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["a.fits"]
    assert (tmp_path / "a.fits").read_bytes() == b"SIMPLE  = T"


def test_a_broken_archive_leaves_only_itself(tmp_path: Path) -> None:
    # A download cut off part way, which gzip reads up to the cut and then fails.
    whole = gzip.compress(b"x" * 100_000)
    (tmp_path / "a.fits.gz").write_bytes(whole[: len(whole) // 2])
    with pytest.raises(EOFError):
        decompress_dir(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["a.fits.gz"]
