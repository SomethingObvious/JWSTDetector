"""Expands every .gz in a folder and deletes it, for the COSMOS-Web weight maps.

astropy can read a .fits.gz directly, but it can't memmap one, so a big mosaic would
have to fit in RAM.

Usage: jwst-peculiar unzip [folder]
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path
import shutil

from tqdm import tqdm

logger = logging.getLogger(__name__)


def decompress_dir(base_dir: Path) -> int:
    """Expand each x.gz in `base_dir` to x and delete x.gz.

    Returns:
        How many archives there were.
    """
    gz_paths = sorted(base_dir.glob("*.gz"))
    for gz_path in tqdm(gz_paths, desc="Decompressing", unit="file"):
        out_path = gz_path.with_suffix("")
        # Under another name until it's whole, so a broken archive or a Ctrl+C
        # doesn't leave half a FITS file that prep would open.
        part = out_path.with_name(out_path.name + ".part")
        try:
            with gzip.open(gz_path, "rb") as f_in, part.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(out_path)
        gz_path.unlink()
    return len(gz_paths)


def main(argv: list[str] | None = None) -> int:
    """Expand the archives in the folder named on the command line.

    Returns:
        The exit code, which is always 0.
    """
    p = argparse.ArgumentParser(
        description="Expand every .gz in a folder and delete the .gz afterwards."
    )
    p.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Folder to expand, the current one by default.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = decompress_dir(Path(args.directory))
    logger.info("Expanded %d files in %s", n, args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
