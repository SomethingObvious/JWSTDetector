from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING, Any

from astropy.table import Table
import pytest

from jwst_peculiar import catalogues

if TYPE_CHECKING:
    from pathlib import Path

    from jwst_peculiar.utils import AstropyTable


def test_coordinates_come_back_in_degrees_whatever_their_unit(tmp_path: Path) -> None:
    t: AstropyTable = Table({"ra": [1.5], "dec": [-300.0]})
    t["ra"].unit = "hourangle"
    t["dec"].unit = "arcmin"
    t.write(tmp_path / "h.ecsv")
    rows = catalogues.read_local(tmp_path / "h.ecsv")["rows"]
    assert rows == [["h 1", pytest.approx(22.5), pytest.approx(-5.0), "", ""]]
    # Without a unit, degrees.
    Table({"ra": [1.5], "dec": [-5.0]}).write(tmp_path / "d.ecsv")
    assert catalogues.read_local(tmp_path / "d.ecsv")["rows"] == [
        ["d 1", 1.5, -5.0, "", ""]
    ]


def test_a_coordinate_that_isnt_an_angle_is_refused(tmp_path: Path) -> None:
    t: AstropyTable = Table({"ra": [1.5], "dec": [-5.0]})
    t["ra"].unit = "s"
    t.write(tmp_path / "t.ecsv")
    with pytest.raises(SystemExit, match=r"gives ra in s, which isn't an angle"):
        catalogues.read_local(tmp_path / "t.ecsv")


def test_a_service_that_goes_quiet_is_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Takes the connection and never answers, like a server that has hung.
    with socket.create_server(("127.0.0.1", 0)) as stalled:
        url = f"http://127.0.0.1:{stalled.getsockname()[1]}/tap"
        monkeypatch.setitem(catalogues.SERVICES["SIMBAD"], "urls", [url, url])
        monkeypatch.setattr(catalogues, "TIMEOUT_S", 0.5)
        got: dict[str, Any] = {}

        def ask() -> None:
            got.update(catalogues.fetch("SIMBAD", [(10.0, -5.0)], 1.0))

        worker = threading.Thread(target=ask, daemon=True)
        worker.start()
        worker.join(timeout=20)
    assert not worker.is_alive()
    assert got["ok"] is False
    assert got["error"].count("Timeout") == 2
