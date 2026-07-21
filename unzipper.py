"""Decompress gzipped FITS mosaics in a directory (COSMOS-Web WHT files ship as .gz).

Each ``<name>.gz`` is expanded to ``<name>`` and the ``.gz`` is removed.

Usage: python unzipper.py [directory]   (defaults to the current directory)
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("unzipper")


def decompress_dir(base_dir: Path) -> int:
    """Expand every ``*.gz`` in ``base_dir`` in place, removing the archive. Returns the count."""
    gz_paths = sorted(base_dir.glob("*.gz"))
    for gz_path in tqdm(gz_paths, desc="Decompressing", unit="file"):
        out_path = gz_path.with_suffix("")  # drop the .gz extension
        with gzip.open(gz_path, "rb") as f_in, out_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
    return len(gz_paths)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Decompress every *.gz in a directory (removes the .gz afterwards)."
    )
    p.add_argument(
        "directory", nargs="?", default=".", help="Directory to scan (default: current)."
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = decompress_dir(Path(args.directory))
    logger.info("Decompressed %d file(s) in %s", n, args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
