#!/usr/bin/env python3

import argparse
import json
import logging
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("summarize_results.py needs pyyaml. Install it with `pip install pyyaml`.") from None

try:
    import pandas as pd
except ImportError:
    raise SystemExit("summarize_results.py needs pandas. Install it with `pip install pandas`.") from None

logger = logging.getLogger("summarize_results")


def flatten_json(obj, prefix=""):
    """Nested dicts as one flat dict with dotted keys, and lists as JSON text."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        key = prefix if prefix else "value"
        out[key] = json.dumps(obj)
    else:
        key = prefix if prefix else "value"
        out[key] = obj
    return out


def safe_read_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_read_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_run_dirs(results_root: Path):
    # Both runners write an args.yaml, so any folder holding one is a run.
    return sorted({p.parent for p in results_root.rglob("args.yaml")})


def parse_seed_from_anywhere(text: str):
    """The seed in a name like seed=3, seed_3 or measurements_seed=3.csv, or None."""
    m = re.search(r"seed(?:=|_)?(\d+)", text)
    return int(m.group(1)) if m else None


def pick_measurement_files(run_dir: Path):
    cands = list(run_dir.glob("measurements*.csv"))
    cands = [p for p in cands if "measure" in p.name.lower()]
    return sorted(cands)


def pick_metrics_files(run_dir: Path):
    cands = list(run_dir.glob("metrics*.json"))
    cands = [p for p in cands if "metric" in p.name.lower()]
    return sorted(cands)


def normalize_measurements_df(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    """A measurements CSV in the summary's column layout.

    It takes the older layout with an Object column and times in MemoryBankTime and
    InferenceTime, or the one both runners write, which has no Object column and
    puts the times in MemoryBankTimeSec and InferenceTimeSec.
    """
    cols = {c.lower(): c for c in df.columns}

    def has(col):
        return col.lower() in cols

    # The older layout, with an Object column.
    if has("object") and has("sample") and has("anomalyscore"):
        out = df.rename(
            columns={
                cols["object"]: "Object",
                cols["sample"]: "Sample",
                cols["anomalyscore"]: "AnomalyScore",
                cols.get("memorybanktime", "MemoryBankTime"): "MemoryBankTime",
                cols.get("inferencetime", "InferenceTime"): "InferenceTime",
            }
        )
        if "MemoryBankTime" not in out.columns:
            out["MemoryBankTime"] = None
        if "InferenceTime" not in out.columns:
            out["InferenceTime"] = None
        return out

    if has("sample") and has("anomalyscore"):
        out = df.rename(
            columns={
                cols["sample"]: "Sample",
                cols["anomalyscore"]: "AnomalyScore",
            }
        )
        if has("memorybanktimesec"):
            out = out.rename(columns={cols["memorybanktimesec"]: "MemoryBankTime"})
        elif has("memorybanktime"):
            out = out.rename(columns={cols["memorybanktime"]: "MemoryBankTime"})
        else:
            out["MemoryBankTime"] = None

        if has("inferencetimesec"):
            out = out.rename(columns={cols["inferencetimesec"]: "InferenceTime"})
        elif has("inferencetime"):
            out = out.rename(columns={cols["inferencetime"]: "InferenceTime"})
        else:
            out["InferenceTime"] = None

        # The grouping below needs an Object, and every row here is a query tile.
        out["Object"] = "query"
        return out[["Object", "Sample", "AnomalyScore", "MemoryBankTime", "InferenceTime"]]

    raise SystemExit(f"{src_path} needs Sample and AnomalyScore columns, and it has {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(
        description="Gather every run under a results folder into summary CSVs."
    )
    ap.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Folder holding the runs.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="results_summary",
        help="The CSVs go in a subfolder named after --results_root.",
    )
    ap.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"There is no results folder at {results_root}")

    # A subfolder per results folder, so summaries of different runs don't overwrite each other.
    outdir = Path(args.outdir) / results_root.name
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(results_root)
    if not run_dirs:
        raise SystemExit(f"There are no runs under {results_root}, as nothing has an args.yaml.")

    run_rows = []
    object_rows = []
    sample_rows = []

    for run_dir in run_dirs:
        args_yaml = safe_read_yaml(run_dir / "args.yaml")
        preprocess_yaml = safe_read_yaml(run_dir / "preprocess.yaml")

        meas_files = pick_measurement_files(run_dir)
        metrics_files = pick_metrics_files(run_dir)

        # Metrics with no seed in the name are filed under None.
        metrics_by_seed = {}
        for mf in metrics_files:
            seed = parse_seed_from_anywhere(mf.name)
            metrics_by_seed[seed] = safe_read_json(mf)

        if not meas_files:
            continue

        for meas_path in meas_files:
            seed = parse_seed_from_anywhere(meas_path.name)
            # measurements_final.csv has no seed in its name, so it takes the one in args.yaml.
            if seed is None and isinstance(args_yaml, dict):
                seed = args_yaml.get("seed", None)

            df_raw = read_csv(meas_path)
            df = normalize_measurements_df(df_raw, meas_path)

            run_mean_infer = (
                float(pd.to_numeric(df["InferenceTime"], errors="coerce").mean())
                if len(df)
                else None
            )
            run_mean_mb = (
                float(pd.to_numeric(df["MemoryBankTime"], errors="coerce").mean())
                if len(df)
                else None
            )

            metrics_flat = flatten_json(metrics_by_seed.get(seed, {}), prefix="metrics")

            row = {
                "run_dir": str(run_dir),
                "seed": seed,
                "measurements_file": str(meas_path),
                "mean_inference_time_s": run_mean_infer,
                "mean_memorybank_time_s": run_mean_mb,
            }
            row.update(flatten_json(args_yaml, prefix="args"))
            row.update(flatten_json(preprocess_yaml, prefix="preprocess"))
            row.update(metrics_flat)
            run_rows.append(row)

            df_num = df.copy()
            df_num["AnomalyScore"] = pd.to_numeric(df_num["AnomalyScore"], errors="coerce")
            df_num["InferenceTime"] = pd.to_numeric(df_num["InferenceTime"], errors="coerce")
            df_num["MemoryBankTime"] = pd.to_numeric(df_num["MemoryBankTime"], errors="coerce")

            g = df_num.groupby("Object", as_index=False).agg(
                n_samples=("Sample", "count"),
                mean_anomaly_score=("AnomalyScore", "mean"),
                std_anomaly_score=("AnomalyScore", "std"),
                mean_inference_time_s=("InferenceTime", "mean"),
                mean_memorybank_time_s=("MemoryBankTime", "mean"),
            )

            for _, r in g.iterrows():
                object_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "seed": seed,
                        "object": r["Object"],
                        "n_samples": int(r["n_samples"]),
                        "mean_anomaly_score": None
                        if pd.isna(r["mean_anomaly_score"])
                        else float(r["mean_anomaly_score"]),
                        "std_anomaly_score": None
                        if pd.isna(r["std_anomaly_score"])
                        else float(r["std_anomaly_score"]),
                        "mean_inference_time_s": None
                        if pd.isna(r["mean_inference_time_s"])
                        else float(r["mean_inference_time_s"]),
                        "mean_memorybank_time_s": None
                        if pd.isna(r["mean_memorybank_time_s"])
                        else float(r["mean_memorybank_time_s"]),
                    }
                )

            df2 = df_num.copy()
            df2.insert(0, "run_dir", str(run_dir))
            df2.insert(1, "seed", seed)
            sample_rows.append(df2)

    runs_df = pd.DataFrame(run_rows)
    objs_df = pd.DataFrame(object_rows)
    samples_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    runs_path = outdir / "runs_summary.csv"
    objs_path = outdir / "objects_summary.csv"
    samples_path = outdir / "samples.csv"
    samples_sorted_path = outdir / "samples_sorted_by_anomaly_score.csv"

    runs_df.to_csv(runs_path, index=False)
    objs_df.to_csv(objs_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    if len(samples_df) > 0 and "AnomalyScore" in samples_df.columns:
        tmp = samples_df.copy()
        tmp["AnomalyScore"] = pd.to_numeric(tmp["AnomalyScore"], errors="coerce")
        tmp = tmp.sort_values("AnomalyScore", ascending=False, na_position="last")
        tmp.to_csv(samples_sorted_path, index=False)
    else:
        pd.DataFrame().to_csv(samples_sorted_path, index=False)

    logger.info(
        "Wrote\n %s\n %s\n %s\n %s", runs_path, objs_path, samples_path, samples_sorted_path
    )


if __name__ == "__main__":
    main()
