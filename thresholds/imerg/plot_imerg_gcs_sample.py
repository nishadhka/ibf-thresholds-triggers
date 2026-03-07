#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=0.1.0",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "matplotlib>=3.8.0",
#     "cartopy>=0.22.0",
#     "geopandas>=0.14.0",
#     "pandas>=2.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Plot 10 random days from the IMERG HH GCS Icechunk store.

Connects to gs://cpc_awc/ea_imerg_ic_store, picks 10 random days,
sums 48 HH timesteps to daily total, and plots with EA borders.

Usage:
    uv run --python 3.12 plot_imerg_gcs_sample.py
    uv run --python 3.12 plot_imerg_gcs_sample.py --n-days 5
    uv run --python 3.12 plot_imerg_gcs_sample.py --gcs-prefix test_ea_imerg_ic_store
"""

import argparse
import random
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"


def plot_day(daily_precip, gdf, date_str, outdir):
    """Plot a single day's precipitation map."""
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = daily_precip.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        vmin=0,
        vmax=max(min(float(daily_precip.max(skipna=True)), 100), 1),
        add_colorbar=False,
    )

    gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8, transform=ccrs.PlateCarree())

    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, centroid.y, row["GID_0"],
            transform=ccrs.PlateCarree(),
            fontsize=7, fontweight="bold", ha="center",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
        )

    ax.coastlines(linewidth=0.5)
    ax.set_extent([21, 53, -12, 23], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Precipitation (mm/day)")

    ax.set_title(f"IMERG HH Daily Sum — {date_str}", fontsize=13)

    out_path = outdir / f"imerg_gcs_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot random days from IMERG GCS store")
    parser.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    parser.add_argument("--n-days", type=int, default=10, help="Number of random days to plot")
    parser.add_argument("--geojson", type=str, default="ea_ghcf_simple.geojson")
    parser.add_argument("--outdir", type=str, default="./plots_gcs_sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sa_path = Path(__file__).parent / SERVICE_ACCOUNT_FILE
    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=args.gcs_prefix,
        service_account_file=str(sa_path),
    )
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    times = pd.DatetimeIndex(ds.time.values)
    n_time = len(times)
    n_days = n_time // 48

    print(f"Store: gs://{GCS_BUCKET}/{args.gcs_prefix}")
    print(f"Shape: {dict(ds.sizes)}")
    print(f"Time:  {times[0]} -> {times[-1]}")
    print(f"Days:  {n_days}")

    # Pick random days
    random.seed(args.seed)
    n_sample = min(args.n_days, n_days)
    day_indices = sorted(random.sample(range(n_days), n_sample))

    print(f"\nPlotting {n_sample} random days...")

    gdf = gpd.read_file(args.geojson)

    for d in day_indices:
        t_start = d * 48
        t_end = t_start + 48
        day_date = str(times[t_start].date())

        # Load 48 HH timesteps and sum to daily
        day_data = ds["precipitation"].isel(time=slice(t_start, t_end)).load()

        # Mask fill values (-9999.9) before summing
        day_data = day_data.where(day_data > -9999)
        has_data = bool(day_data.notnull().any())
        # Sum HH precipitation rates (mm/hr) * 0.5hr = mm per half-hour, sum to daily
        daily_total = day_data.sum(dim="time", min_count=1) * 0.5

        status = "HAS DATA" if has_data else "EMPTY (all zeros)"
        print(f"  Day {d}: {day_date} — {status}, max={float(daily_total.max()):.2f} mm")

        plot_day(daily_total, gdf, day_date, outdir)

    print(f"\nDone: {n_sample} plots saved to {outdir}/")


if __name__ == "__main__":
    main()
