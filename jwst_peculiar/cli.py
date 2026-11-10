"""The `jwst-peculiar` command, with one subcommand per step.

Each subcommand's module is imported only when it runs, so `jwst-peculiar --help`
doesn't wait for torch to load.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prep": (
        "jwst_peculiar.prep",
        "run",
        "Cut FITS mosaics into tiles and FITS cutouts.",
    ),
    "levels": (
        "jwst_peculiar.prep",
        "levels_main",
        "Measure one stretch per filter over datasets already cut.",
    ),
    "run": (
        "jwst_peculiar.bootstrap",
        "main",
        "Rank every tile with no labels, in two passes.",
    ),
    "score": (
        "jwst_peculiar.reference",
        "main",
        "Score tiles against tiles known to be normal.",
    ),
    "match": (
        "jwst_peculiar.catalogues",
        "main",
        "Cross-match the candidates with catalogues.",
    ),
    "candidates": (
        "jwst_peculiar.candidates",
        "main",
        "Write the candidates as a table and regions.",
    ),
    "view": (
        "jwst_peculiar.server",
        "main",
        "Open a run in the viewer, or export it as a site.",
    ),
    "bench": (
        "jwst_peculiar.benchmark",
        "main",
        "Plant sources and measure how many are found.",
    ),
    "recall": (
        "jwst_peculiar.recall",
        "main",
        "Rank objects already known to be in the field.",
    ),
    "unzip": (
        "jwst_peculiar.unzip",
        "main",
        "Expand .gz mosaics so they can be memory-mapped.",
    ),
}


def usage() -> str:
    """Return the top-level help, with one line per command."""
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    return (
        "usage: jwst-peculiar COMMAND [options]\n\n"
        "Unsupervised anomaly detection for JWST images.\n\ncommands:\n"
        + "\n".join(lines)
        + "\n\nRun `jwst-peculiar COMMAND -h` for a command's options."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the subcommand named in `argv`.

    Returns:
        The subcommand's exit code, or 2 when there's no such command.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(usage() + "\n")
        return 0
    if argv[0] not in COMMANDS:
        sys.stderr.write(
            f"jwst-peculiar: there's no {argv[0]!r} command.\n\n{usage()}\n"
        )
        return 2
    module, func, _ = COMMANDS[argv[0]]
    # argparse names the program after argv[0], and this makes its usage line say
    # `jwst-peculiar prep` rather than just `jwst-peculiar`.
    sys.argv[0] = f"jwst-peculiar {argv[0]}"
    return getattr(importlib.import_module(module), func)(argv[1:]) or 0
