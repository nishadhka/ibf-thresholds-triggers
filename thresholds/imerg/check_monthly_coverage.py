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
Monthly coverage report for the IMERG HH EA Icechunk store on GCS.

For each month in the store, reports:
  - Expected days / actual filled days
  - Number of missing days
  - Mean precipitation (sanity check — 0 means unfilled)

Faster than check_missing_days.py: samples one timestep per day instead
of loading full day slices.

Usage:
    uv run --python 3.12 check_monthly_coverage.py
    uv run --python 3.12 check_monthly_coverage.py --gcs-prefix test_ea_imerg_ic_store
    uv run --python 3.12 check_monthly_coverage.py --list-missing   # print every missing date
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"


def main():
    parser = argparse.ArgumentParser(description="Monthly coverage report for IMERG store")
    parser.add_argument("--gcs-prefix", type=str, default=None)
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--list-missing", action="store_true",
                        help="Print every missing date")
    args = parser.parse_args()

    import icechunk
    import xarray as xr

    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    sa_path = Path(__file__).parent / args.sa_file
    if not sa_path.exists():
        sa_path = Path(__file__).parent / ".." / "hf-gdo" / args.sa_file
    if not sa_path.exists():
        print(f"ERROR: SA file not found: {args.sa_file}")
        return

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=gcs_prefix,
        service_account_file=str(sa_path),
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    times = pd.DatetimeIndex(ds.time.values)
    n_time = len(times)
    n_days = n_time // 48

    print(f"Store: gs://{GCS_BUCKET}/{gcs_prefix}")
    print(f"Shape: {dict(ds.sizes)}")
    print(f"Time:  {times[0]} → {times[-1]}")
    print(f"Days:  {n_days}  ({n_time} half-hourly steps)")
    print()

    # Build day index: first HH timestep index for each day
    day_dates = [times[d * 48].date() for d in range(n_days)]
    day_months = pd.Series([pd.Timestamp(d).to_period("M") for d in day_dates])

    # Check each day: sample first HH timestep, check if any non-zero
    print("Scanning days (one sample per day)...")
    filled = np.zeros(n_days, dtype=bool)
    start = time.time()

    for d in range(n_days):
        t_idx = d * 48
        sample = ds["precipitation"].isel(time=t_idx).values
        filled[d] = np.any(sample != 0)

        if (d + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"  {d+1}/{n_days} checked ({elapsed:.0f}s)")

    elapsed = time.time() - start
    total_filled = filled.sum()
    total_missing = n_days - total_filled
    print(f"\nScan complete in {elapsed:.0f}s")
    print(f"  Filled:  {total_filled}/{n_days} ({100*total_filled/n_days:.1f}%)")
    print(f"  Missing: {total_missing}/{n_days} ({100*total_missing/n_days:.1f}%)")

    # Monthly summary
    print(f"\n{'Month':<10} {'Expected':>8} {'Filled':>8} {'Missing':>8} {'Coverage':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    all_missing = []
    months = sorted(day_months.unique())

    for month in months:
        mask = (day_months == month).values
        expected = mask.sum()
        month_filled = filled[mask].sum()
        month_missing = expected - month_filled
        pct = 100 * month_filled / expected if expected > 0 else 0

        flag = "" if month_missing == 0 else f"  ← {month_missing} missing"
        print(f"{str(month):<10} {expected:>8} {month_filled:>8} {month_missing:>8} {pct:>9.1f}%{flag}")

        if month_missing > 0:
            missing_idx = np.where(mask & ~filled)[0]
            for idx in missing_idx:
                all_missing.append(day_dates[idx])

    # Summary
    print(f"\n{'='*60}")
    complete_months = sum(1 for m in months
                         if filled[(day_months == m).values].sum() == (day_months == m).sum())
    print(f"Complete months: {complete_months}/{len(months)}")
    print(f"Months with gaps: {len(months) - complete_months}")

    if all_missing and args.list_missing:
        print(f"\nAll missing dates ({len(all_missing)}):")
        # Group into ranges
        ranges = []
        range_start = all_missing[0]
        prev = all_missing[0]
        for d in all_missing[1:]:
            if (d - prev).days == 1:
                prev = d
            else:
                ranges.append((range_start, prev))
                range_start = d
                prev = d
        ranges.append((range_start, prev))

        for r_start, r_end in ranges:
            n = (r_end - r_start).days + 1
            if r_start == r_end:
                print(f"  {r_start}")
            else:
                print(f"  {r_start} → {r_end} ({n} days)")

        # Write to file
        with open("missing_days.txt", "w") as f:
            for d in all_missing:
                f.write(f"{d}\n")
        print(f"\nMissing dates written to missing_days.txt")

    elif all_missing:
        print(f"\nUse --list-missing to see all {len(all_missing)} missing dates")


if __name__ == "__main__":
    main()
