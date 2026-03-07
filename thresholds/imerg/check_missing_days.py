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
Check for missing days in the IMERG HH EA Icechunk store on GCS.

Scans each day-chunk (48 HH timesteps) and reports which days have data
vs which are still empty (all zeros or fill values).

Usage:
    uv run --python 3.12 check_missing_days.py
    uv run --python 3.12 check_missing_days.py --gcs-prefix test_ea_imerg_ic_store
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
FILL_VALUE = np.float32(-9999.9)


def main():
    parser = argparse.ArgumentParser(description="Check missing days in IMERG store")
    parser.add_argument("--gcs-prefix", type=str, default=None)
    args = parser.parse_args()

    import icechunk
    import xarray as xr

    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    sa_path = Path(__file__).parent / SERVICE_ACCOUNT_FILE

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
    print(f"Time:  {times[0]} -> {times[-1]}")
    print(f"Days:  {n_days}")
    print()

    filled_days = []
    empty_days = []
    start = time.time()

    for d in range(n_days):
        t_start = d * 48
        t_end = t_start + 48
        day_date = times[t_start].date()

        # Load one full timestep to check — 400x345 float32 = 552 KB
        sample = ds["precipitation"].isel(time=t_start).values

        # Unfilled template has all zeros; filled data has some non-zero precip
        has_data = np.any(sample != 0)

        if has_data:
            filled_days.append(day_date)
        else:
            empty_days.append(day_date)

        if (d + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  Checked {d+1}/{n_days} days ({elapsed:.0f}s) — "
                  f"{len(filled_days)} filled, {len(empty_days)} empty")

    elapsed = time.time() - start
    print(f"\nScan complete in {elapsed:.0f}s")
    print(f"  Filled: {len(filled_days)} / {n_days} ({100*len(filled_days)/n_days:.1f}%)")
    print(f"  Empty:  {len(empty_days)} / {n_days} ({100*len(empty_days)/n_days:.1f}%)")

    if empty_days:
        print(f"\nMissing days ({len(empty_days)}):")
        # Group consecutive missing days into ranges
        ranges = []
        range_start = empty_days[0]
        prev = empty_days[0]
        for d in empty_days[1:]:
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
                print(f"  {r_start} to {r_end} ({n} days)")

        # Write missing dates to file for easy re-fill
        with open("missing_days.txt", "w") as f:
            for d in empty_days:
                f.write(f"{d}\n")
        print(f"\nMissing dates written to missing_days.txt")
    else:
        print("\nNo missing days — store is complete!")


if __name__ == "__main__":
    main()
