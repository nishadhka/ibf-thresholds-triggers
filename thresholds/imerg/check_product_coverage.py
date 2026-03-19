#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=0.1.0",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Disentangle IMERG product quality (Final / Late / Early) per day.

Parses Icechunk commit history to determine which IMERG product was used
to fill each day in the store. Outputs a monthly summary and optional
per-day CSV.

Product hierarchy:
  Final  (GPM_3IMERGHH)  — research-grade, ~3.5 month latency, quality=3
  Late   (GPM_3IMERGHHL) — intermediate, ~14 hour latency,    quality=2
  Early  (GPM_3IMERGHHE) — near-real-time, ~4 hour latency,   quality=1

Usage:
    uv run --python 3.12 check_product_coverage.py
    uv run --python 3.12 check_product_coverage.py --csv product_coverage.csv
    uv run --python 3.12 check_product_coverage.py --gcs-prefix test_ea_imerg_ic_store
"""

import argparse
import re
from pathlib import Path

import pandas as pd

GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"

IMERG_PRODUCTS = {
    "GPM_3IMERGHH":  {"name": "Final", "quality": 3},
    "GPM_3IMERGHHL": {"name": "Late",  "quality": 2},
    "GPM_3IMERGHHE": {"name": "Early", "quality": 1},
}

QUALITY_TO_NAME = {3: "Final", 2: "Late", 1: "Early", 0: "Empty"}


def parse_commits(repo):
    """Parse Icechunk commit history and return {time_index: quality_level}."""
    filled = {}  # t_start_idx -> max quality level

    for commit in repo.ancestry(branch="main"):
        msg = commit.message
        if msg.startswith("initialize "):
            break  # stop at init boundary

        if not msg.startswith("fill batch "):
            continue

        # Determine product from tag: "fill batch X-Y [GPM_3IMERGHHE]: N/M OK"
        quality = 0
        for short_name, info in IMERG_PRODUCTS.items():
            if f"[{short_name}]" in msg:
                quality = info["quality"]
                break
        if quality == 0:
            # Old commits without product tag are assumed Final
            quality = 3

        # Parse index range
        try:
            range_part = msg.split("[")[0] if "[" in msg else msg.split(":")[0]
            range_str = range_part.replace("fill batch ", "").strip().rstrip("-")
            b_start, b_end = range_str.split("-")
            for idx in range(int(b_start), int(b_end) + 1):
                filled[idx] = max(filled.get(idx, 0), quality)
        except (ValueError, IndexError):
            pass

    return filled


def main():
    parser = argparse.ArgumentParser(
        description="Show IMERG product quality (Final/Late/Early) per day"
    )
    parser.add_argument("--gcs-prefix", type=str, default=None)
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--csv", type=str, default=None,
                        help="Write per-day product coverage to CSV file")
    args = parser.parse_args()

    import icechunk
    import xarray as xr

    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    sa_path = Path(__file__).parent / args.sa_file
    if not sa_path.exists():
        sa_path = Path(__file__).parent / ".." / "hf-gdo" / args.sa_file

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=gcs_prefix,
        service_account_file=str(sa_path),
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )

    # Get time axis
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    n_days = len(times) // 48
    ds.close()

    print(f"Store: gs://{GCS_BUCKET}/{gcs_prefix}")
    print(f"Days:  {n_days}  (time steps: {len(times)})")
    print(f"Range: {times[0].date()} → {times[-1].date()}")

    # Parse commits
    print("\nParsing commit history...")
    filled = parse_commits(repo)
    print(f"  Commits cover {len(filled)} time indices")

    # Map time indices to days and their product quality
    days = []
    for d in range(n_days):
        t_idx = d * 48
        date = times[t_idx].date()
        quality = filled.get(t_idx, 0)
        days.append({
            "date": date,
            "product": QUALITY_TO_NAME[quality],
            "quality": quality,
        })

    df = pd.DataFrame(days)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    # Overall summary
    print(f"\n{'='*50}")
    print("Overall product breakdown:")
    print(f"{'='*50}")
    for q in [3, 2, 1, 0]:
        name = QUALITY_TO_NAME[q]
        count = (df["quality"] == q).sum()
        pct = 100 * count / len(df)
        print(f"  {name:<8} {count:>6} days  ({pct:5.1f}%)")

    # Monthly summary
    print(f"\n{'Month':<10} {'Final':>7} {'Late':>7} {'Early':>7} {'Empty':>7} {'Total':>7}")
    print(f"{'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for month in sorted(df["month"].unique()):
        mdf = df[df["month"] == month]
        n_final = (mdf["quality"] == 3).sum()
        n_late = (mdf["quality"] == 2).sum()
        n_early = (mdf["quality"] == 1).sum()
        n_empty = (mdf["quality"] == 0).sum()
        total = len(mdf)

        # Only show flag for non-Final months
        flag = ""
        if n_late > 0 or n_early > 0:
            parts = []
            if n_late: parts.append(f"{n_late}L")
            if n_early: parts.append(f"{n_early}E")
            flag = f"  ← {'+'.join(parts)}"
        elif n_empty > 0:
            flag = f"  ← {n_empty} empty"

        print(f"{str(month):<10} {n_final:>7} {n_late:>7} {n_early:>7} {n_empty:>7} {n_total:>7}{flag}"
              if False else
              f"{str(month):<10} {n_final:>7} {n_late:>7} {n_early:>7} {n_empty:>7} {total:>7}{flag}")

    # CSV export
    if args.csv:
        out = df[["date", "product", "quality"]].copy()
        out.to_csv(args.csv, index=False)
        print(f"\nPer-day coverage written to {args.csv}")

    # Show transition points
    print(f"\n{'='*50}")
    print("Product transitions (where quality level changes):")
    print(f"{'='*50}")
    prev_q = None
    for _, row in df.iterrows():
        if row["quality"] != prev_q:
            print(f"  {row['date']}  →  {row['product']}")
            prev_q = row["quality"]


if __name__ == "__main__":
    main()
