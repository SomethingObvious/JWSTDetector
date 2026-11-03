"""Checks for what the viewer reads: the run bundle, cross-matching and candidate tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.table import Table
from PIL import Image

from jwstdetector import bundle, candidates, catalogues

SCORES = {"a.png": 0.2, "b.png": float("nan"), "c.png": 0.9}
WCS_META = {
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 12.0,
    "CRPIX2": 4.0,
    "CRVAL1": 10.0,
    "CRVAL2": -5.0,
    "CD1_1": -1e-4,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 1e-4,
}


@pytest.fixture
def run(tmp_path: Path):
    """Three 8 px tiles side by side on one mosaic, each with its peak patch bottom left."""
    data, run = tmp_path / "data", tmp_path / "run"
    maps = run / "pass2" / "anomaly_maps" / "seed=0"
    for d in (data / "query", data / "mosaics", data / "cutouts", maps):
        d.mkdir(parents=True)
    with (data / "manifest.csv").open("w", newline="") as f:
        f.write("key,mosaic,x0,y0,size,ra,dec,cutout\n")
        for i, key in enumerate(SCORES):
            f.write(f"{key},m,{i * 8},0,8,10.0,-5.0,cutouts/{key}.fits\n")
            Image.new("RGB", (8, 8)).save(data / "query" / key)
            (data / "cutouts" / f"{key}.fits").write_bytes(b"fits")
            grid = np.zeros((2, 2), np.float32)
            grid[1, 0] = 1
            np.save(maps / f"{Path(key).stem}.npy", grid)
    meta = {"shape": [8, 24], "wcs": WCS_META, "pixel_scale_arcsec": 0.36}
    (data / "mosaics" / "m.json").write_text(json.dumps(meta))
    Image.new("RGB", (24, 8)).save(data / "mosaics" / "m.png")
    (run / "args.yaml").write_text(yaml.safe_dump({"resolution": 32, "seed": 0}))
    with (run / "measurements_final.csv").open("w", newline="") as f:
        f.write("Sample,AnomalyScore\n")
        f.writelines(f"{k},{v}\n" for k, v in SCORES.items())
    return run, data


def test_bundle_ranks_exports_and_saves_verdicts(run, tmp_path: Path) -> None:
    run_dir, data = run
    b = bundle.RunBundle(run_dir, data, writable=True)
    tiles = b.index["tiles"]
    assert [t["key"] for t in tiles] == ["c.png", "a.png", "b.png"]
    assert tiles[2]["score"] is None
    assert tiles[0]["peak"] == [1, 0]
    assert "NaN" not in b.read("index.json")[0].decode()
    assert len(b.read("maps/000000.f32")[0]) == 4 * 4

    n = bundle.export(b, tmp_path / "site", cutouts=1)
    assert sorted(p.name for p in (tmp_path / "site/run/cutouts").iterdir()) == ["000000.fits"]
    assert n == 2 + 3 * 2 + 1 + 1

    site = tmp_path / "site"
    (site / "index.html").write_text(
        '<html><head><title>x</title>\n<meta name="description" content="old" />\n'
        '<meta property="og:title" content="old" />\n  </head></html>',
        encoding="utf-8",
    )
    bundle.add_site_meta(b, site, "https://example.org/demo")
    page = (site / "index.html").read_text(encoding="utf-8")
    assert '<link rel="canonical" href="https://example.org/demo/" />' in page
    assert 'content="old"' not in page
    assert "application/ld+json" in page
    with Image.open(site / "social.png") as card:
        assert card.size == (1200, 630)
    assert "<loc>https://example.org/demo/</loc>" in (site / "sitemap.xml").read_text()

    bundle.save_verdict(run_dir, "a.png", "artifact", True, "A diffraction spike")
    bundle.save_verdict(run_dir, "c.png", "interesting", False, "")
    bundle.save_verdict(run_dir, "c.png", None, False, "")
    saved = bundle.read_verdicts(run_dir)
    assert saved == {
        "a.png": {
            "verdict": "artifact",
            "star": True,
            "note": "A diffraction spike",
            "updated": saved["a.png"]["updated"],
        }
    }


def test_peak_sky_position_follows_the_png_orientation(run) -> None:
    # The peak patch is bottom left in the PNG, which is FITS row y0 + 1.5 for a 2x2
    # grid of 4 px patches, not y0 + 5.5.
    b = bundle.RunBundle(*run, writable=False)
    c = b.index["tiles"][0]
    wcs = bundle.mosaic_wcs(b.index["mosaics"]["m"])
    ra, dec = wcs.all_pix2world([16 + 2 - 0.5], [8 - 6 - 0.5], 0)
    assert c["peak_sky"] == [round(float(ra[0]), 7), round(float(dec[0]), 7)]


def test_crossmatch_with_a_local_catalogue(run, tmp_path: Path) -> None:
    run_dir, data = run
    peak = bundle.RunBundle(run_dir, data, writable=False).index["tiles"][0]["peak_sky"]
    cat = tmp_path / "mine.csv"
    Table({"ID": ["near", "far"], "RA": [peak[0], 30.0], "DEC": [peak[1], 20.0]}).write(cat)

    result = catalogues.crossmatch(run_dir, data, radius=1.0, services=[], local=[cat])
    assert result["checked"] == ["mine"]
    assert result["tiles"]["c.png"]["uncatalogued"] is False
    assert result["tiles"]["c.png"]["nearest"]["mine"]["name"] == "mine near"
    assert result["tiles"]["c.png"]["nearest"]["mine"]["sep"] < 1e-3
    # a.png's peak is 8 px of 0.36 arcsec away from the source.
    assert result["tiles"]["a.png"]["uncatalogued"] is True
    cached = json.loads((run_dir / "catalogues" / "m.json").read_text())
    assert [r[0] for r in cached["catalogues"]["mine"]["rows"]] == ["mine near"]

    b = bundle.RunBundle(run_dir, data, writable=False)
    assert b.index["run"]["xmatch"]["checked"] == ["mine"]
    over_c = json.loads(b.read("cats/000000.json")[0])
    assert len(over_c) == 1
    name, u, v = over_c[0][1], over_c[0][4], over_c[0][5]
    assert name == "mine near"
    # The peak patch's centre, a quarter of the way in from the left and from the bottom.
    assert abs(u - 0.25) < 1e-4
    assert abs(v - 0.75) < 1e-4
    assert json.loads(b.read("cats/000001.json")[0]) == []

    table = candidates.candidate_table(b)
    assert list(table["rank"]) == [1, 2, 3]
    assert list(table["mine_name"][:2]) == ["mine near", "mine near"]
    assert list(table["uncatalogued"][:2]) == [False, True]
    unmatched = candidates.candidate_table(b, uncatalogued=True)
    assert list(unmatched["tile"]) == ["a.png", "b.png"]
    reg = tmp_path / "c.reg"
    candidates.write_regions(table, reg)
    assert reg.read_text().count("circle(") == 3


def test_descriptor_embedding_is_unit_length_and_padded(run) -> None:
    run_dir, data = run
    rng = np.random.default_rng(0)
    np.savez(
        run_dir / "pass2" / "descriptors.npz",
        keys=np.array(list(SCORES)),
        vectors=rng.normal(size=(3, 16)).astype(np.float16),
    )
    b = bundle.RunBundle(run_dir, data, writable=False)
    assert b.index["run"]["embedding_dims"] == bundle.EMBED_DIMS
    raw = np.frombuffer(b.read("embedding.f32")[0], dtype="<f4").reshape(3, bundle.EMBED_DIMS)
    assert np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-5)
    assert "embedding.f32" in list(b.paths())


def test_a_failed_refresh_keeps_the_last_good_copy(run, monkeypatch) -> None:
    run_dir, data = run
    good = {"ok": True, "server": "a", "rows": [["NED x", 10.0, -5.0, "G", ""]], "error": ""}
    monkeypatch.setattr(catalogues, "fetch", lambda *a: good)
    catalogues.crossmatch(run_dir, data, services=["NED"])
    down = {"ok": False, "server": "", "rows": [], "error": "reset"}
    monkeypatch.setattr(catalogues, "fetch", lambda *a: down)
    result = catalogues.crossmatch(run_dir, data, services=["NED"], refresh=True)
    assert result["checked"] == ["NED"]
    assert result["failed"] == {}
    cached = json.loads((run_dir / "catalogues" / "m.json").read_text())
    assert cached["catalogues"]["NED"]["rows"] == good["rows"]


def test_only_the_catalogues_asked_for_are_used(run, monkeypatch) -> None:
    run_dir, data = run
    good = {"ok": True, "server": "a", "rows": [["NED x", 10.0, -5.0, "G", ""]], "error": ""}
    monkeypatch.setattr(catalogues, "fetch", lambda *a: dict(good))
    catalogues.crossmatch(run_dir, data, services=["NED"])

    # NED stays in the cache but isn't asked this time, so nothing is checked and no
    # tile can be called catalogued.
    result = catalogues.crossmatch(run_dir, data, services=[])
    assert result["checked"] == []
    assert result["tiles"] == {}
    cached = json.loads((run_dir / "catalogues" / "m.json").read_text())
    assert cached["catalogues"]["NED"]["rows"] == good["rows"]


def test_social_card_keeps_off_a_blank_gap() -> None:
    lit = np.r_[np.full(10, 0.4), np.zeros(5), np.full(30, 0.4)]
    lo, hi = bundle._lit_span(lit, 20, 20)
    assert 12 < lo <= 15
    assert hi == 45
    # Too short a run to fit the card falls back to the whole axis.
    assert bundle._lit_span(lit, 20, 40) == (0, 45)
