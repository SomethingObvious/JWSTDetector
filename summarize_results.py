#!/usr/bin/env python3
"""Gathers every run under a results folder into summary CSVs.

A run is any folder holding an args.yaml, and its scores are the measurements*.csv
files beside it. samples_sorted_by_anomaly_score.csv is the one viewer.py reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("summarize_results")

COLUMNS = ["Object", "Sample", "AnomalyScore", "MemoryBankTime", "InferenceTime"]


def parse_seed_from_anywhere(text: str) -> int | None:
    """The seed in a name like seed=3, seed_3 or measurements_seed=3.csv, or None."""
    m = re.search(r"seed(?:=|_)?(\d+)", text)
    return int(m.group(1)) if m else None


def normalize_measurements_df(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    """A measurements CSV in the summary's column layout, with numbers parsed.

    It takes the runners' Sample, AnomalyScore, MemoryBankTimeSec, InferenceTimeSec
    layout as well as AnomalyDINO's, which has an Object column and no Sec suffixes.
    """
    cols = {c.lower(): c for c in df.columns}
    if "sample" not in cols or "anomalyscore" not in cols:
        raise SystemExit(
            f"{src_path} needs Sample and AnomalyScore columns, and it has {list(df.columns)}"
        )

    def number(*names: str) -> pd.Series:
        for name in names:
            if name in cols:
                return pd.to_numeric(df[cols[name]], errors="coerce")
        return pd.Series(float("nan"), index=df.index)

    out = pd.DataFrame(
        {
            "Object": df[cols["object"]] if "object" in cols else "query",
            "Sample": df[cols["sample"]],
            "AnomalyScore": number("anomalyscore"),
            "MemoryBankTime": number("memorybanktimesec", "memorybanktime"),
            "InferenceTime": number("inferencetimesec", "inferencetime"),
        }
    )
    return out[COLUMNS]


def _nan_to_none(x: float) -> float | None:
    return None if pd.isna(x) else float(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", default="results", help="Folder holding the runs.")
    ap.add_argument(
        "--outdir",
        default="results_summary",
        help="The CSVs go in a subfolder named after --results_root.",
    )
    ap.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"There is no results folder at {results_root}")
    run_dirs = sorted({p.parent for p in results_root.rglob("args.yaml")})
    if not run_dirs:
        raise SystemExit(f"There are no runs under {results_root}, as nothing has an args.yaml.")

    outdir = Path(args.outdir) / results_root.name
    outdir.mkdir(parents=True, exist_ok=True)

    run_rows, object_rows, sample_frames = [], [], []
    for run_dir in run_dirs:
        run_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
        for meas_path in sorted(run_dir.glob("measurements*.csv")):
            seed = parse_seed_from_anywhere(meas_path.name)
            if seed is None:
                seed = run_args.get("seed")
            df = normalize_measurements_df(pd.read_csv(meas_path), meas_path)

            row = {
                "run_dir": str(run_dir),
                "seed": seed,
                "measurements_file": str(meas_path),
                "mean_inference_time_s": _nan_to_none(df["InferenceTime"].mean()),
                "mean_memorybank_time_s": _nan_to_none(df["MemoryBankTime"].mean()),
            }
            for key, value in run_args.items():
                row[f"args.{key}"] = json.dumps(value) if isinstance(value, list) else value
            run_rows.append(row)

            per_object = df.groupby("Object", as_index=False).agg(
                n_samples=("Sample", "count"),
                mean_anomaly_score=("AnomalyScore", "mean"),
                std_anomaly_score=("AnomalyScore", "std"),
                mean_inference_time_s=("InferenceTime", "mean"),
                mean_memorybank_time_s=("MemoryBankTime", "mean"),
            )
            for r in per_object.itertuples(index=False):
                object_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "seed": seed,
                        "object": r.Object,
                        "n_samples": int(r.n_samples),
                        "mean_anomaly_score": _nan_to_none(r.mean_anomaly_score),
                        "std_anomaly_score": _nan_to_none(r.std_anomaly_score),
                        "mean_inference_time_s": _nan_to_none(r.mean_inference_time_s),
                        "mean_memorybank_time_s": _nan_to_none(r.mean_memorybank_time_s),
                    }
                )

            df.insert(0, "run_dir", str(run_dir))
            df.insert(1, "seed", seed)
            sample_frames.append(df)

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    pd.DataFrame(run_rows).to_csv(outdir / "runs_summary.csv", index=False)
    pd.DataFrame(object_rows).to_csv(outdir / "objects_summary.csv", index=False)
    samples.to_csv(outdir / "samples.csv", index=False)
    if len(samples):
        samples = samples.sort_values("AnomalyScore", ascending=False, na_position="last")
    samples.to_csv(outdir / "samples_sorted_by_anomaly_score.csv", index=False)
    logger.info("Wrote the summaries for %d runs to %s", len(run_dirs), outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
