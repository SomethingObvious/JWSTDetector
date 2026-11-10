"""The model confirmation in docs/selection.md, from benches that have finished.

    python -m jwst_peculiar.confirm --default DIR [DIR ...] --candidate DIR [DIR ...]

Each DIR is the --out_dir of one visit's `jwst-peculiar bench`, with the visits in
the same order on both sides. The candidate passes if it found more planted sources
in total and the exact one-sided sign-flip test over the visit differences gives a
p at or below --alpha. It counts visits and not seeds, as a visit's seeds plant into
the same tiles and move together.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Every sign pattern is tried, and 2^20 of them is about a million.
MAX_VISITS = 20


def sign_flip_p(differences: Sequence[float]) -> float:
    """Exact one-sided p-value for the sum of paired differences.

    If the candidate were no better, each visit's difference would be as likely to
    have come out with the other sign. So p is the share of all the sign patterns
    whose sum is at least the one seen, which can't go below 1 over 2^n.

    Returns:
        The p-value.

    Raises:
        ValueError: If there are no differences or more than MAX_VISITS.
    """
    d = np.asarray(differences, dtype=np.float64)
    if not 0 < d.size <= MAX_VISITS:
        msg = f"The test takes 1 to {MAX_VISITS} visits, not {d.size}."
        raise ValueError(msg)
    bits = (np.arange(1 << d.size)[:, None] >> np.arange(d.size)) & 1
    sums = (1 - 2 * bits) @ d
    return float((sums >= d.sum()).mean())


def total(bench_dir: Path, measure: str, detector: str = "knn") -> int:
    """Count what one bench found, over every kind, magnitude and seed.

    Returns:
        The sum of `measure` over the bench's "all" rows for `detector`.

    Raises:
        ValueError: If summary.csv has no such rows.
    """
    with (bench_dir / "summary.csv").open(encoding="utf-8", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["detector"] == detector and r["kind"] == "all"
        ]
    if not rows:
        msg = f"{bench_dir / 'summary.csv'} has no rows for {detector}."
        raise ValueError(msg)
    return sum(int(r[measure]) for r in rows)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jwst_peculiar.confirm",
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0],
    )
    p.add_argument("--default", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--measure", default="found", choices=("found", "beats_all"))
    p.add_argument("--detector", default="knn")
    return p


def main(argv: list[str] | None = None) -> int:
    """Print each visit's totals, the p-value and whether the candidate passes.

    Returns:
        0, also when it doesn't pass.

    Raises:
        SystemExit: If the two sides don't list the same number of benches.
    """
    args = _parser().parse_args(argv)
    if len(args.default) != len(args.candidate):
        msg = (
            f"--default lists {len(args.default)} benches and --candidate "
            f"{len(args.candidate)}, and they have to pair up visit by visit."
        )
        raise SystemExit(msg)
    base = [total(d, args.measure, args.detector) for d in args.default]
    tried = [total(d, args.measure, args.detector) for d in args.candidate]
    p = sign_flip_p([t - b for t, b in zip(tried, base, strict=True)])
    verdict = {
        "measure": args.measure,
        "default": base,
        "candidate": tried,
        "default_total": sum(base),
        "candidate_total": sum(tried),
        "p": p,
        "alpha": args.alpha,
        "passed": sum(tried) > sum(base) and p <= args.alpha,
    }
    sys.stdout.write(json.dumps(verdict, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
