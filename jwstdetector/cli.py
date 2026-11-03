"""The `jwstdetector` command, with one subcommand per step.

Each subcommand's module is imported only when it runs, so `jwstdetector --help`
doesn't wait for torch to load.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prep": ("jwstdetector.prep", "run", "Cut FITS mosaics into tiles and FITS cutouts."),
    "run": ("jwstdetector.bootstrap", "main", "Rank every tile with no labels, in two passes."),
    "score": ("jwstdetector.reference", "main", "Score tiles against tiles known to be normal."),
    "match": ("jwstdetector.catalogues", "main", "Cross-match the candidates with catalogues."),
    "candidates": (
        "jwstdetector.candidates",
        "main",
        "Write the candidates as a table and regions.",
    ),
    "view": ("jwstdetector.server", "main", "Open a run in the viewer, or export it as a site."),
    "bench": ("jwstdetector.benchmark", "main", "Plant sources and measure how many are found."),
    "unzip": ("jwstdetector.unzip", "main", "Expand .gz mosaics so they can be memory-mapped."),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    return (
        "usage: jwstdetector COMMAND [options]\n\n"
        "Unsupervised anomaly detection for JWST images.\n\ncommands:\n"
        + "\n".join(lines)
        + "\n\nRun `jwstdetector COMMAND -h` for a command's options."
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] not in COMMANDS:
        print(f"jwstdetector: there's no {argv[0]!r} command.\n\n{usage()}", file=sys.stderr)
        return 2
    module, func, _ = COMMANDS[argv[0]]
    # argparse names the program after argv[0], and this makes its usage line say
    # `jwstdetector prep` rather than just `jwstdetector`.
    sys.argv[0] = f"jwstdetector {argv[0]}"
    return getattr(importlib.import_module(module), func)(argv[1:]) or 0
