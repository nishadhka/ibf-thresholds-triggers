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
#     "huggingface_hub>=0.20.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Plot daily IMERG precipitation from Icechunk store with EA country borders.

Supports local stores and HuggingFace-hosted stores. Use --start-date and
--end-date to select a date range (defaults to all available timesteps).

Usage:
    # Plot all days in a local store
    uv run --python 3.12 plot_imerg_ea.py --store ./imerg_ea_local

    # Plot a specific 7-day window
    uv run --python 3.12 plot_imerg_ea.py --store ./imerg_ea_local \
        --start-date 2024-12-05 --end-date 2024-12-11

    # Plot from HuggingFace store
    uv run --python 3.12 plot_imerg_ea.py \
        --hf-repo E4DRR/icechunk-stores --hf-prefix test3_imerg-update-test \
        --start-date 2024-12-01 --end-date 2024-12-07
"""

import argparse
import os
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def open_store(args):
    """Open Icechunk store from local path or HuggingFace."""
    if args.hf_repo:
        from dotenv import load_dotenv
        from huggingface_hub import snapshot_download

        load_dotenv()
        hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
        prefix = args.hf_prefix or "test3_imerg-update-test"

        print(f"Downloading store from HF: {args.hf_repo}/{prefix}...")
        local_dir = snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            allow_patterns=f"{prefix}/**",
            local_dir=tempfile.mkdtemp(prefix="imerg_hf_"),
            token=hf_token,
        )
        store_path = os.path.join(local_dir, prefix)
        print(f"  Cached to: {store_path}")
    else:
        store_path = args.store

    storage = icechunk.local_filesystem_storage(path=store_path)
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def plot_day(da, gdf, outdir):
    """Plot a single day's precipitation map."""
    date_str = str(da.time.values)[:10]
    da = da.load()

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        vmin=0,
        vmax=min(float(da.max()), 100),
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

    ax.set_title(f"IMERG Daily Early Precipitation — {date_str}", fontsize=13)

    out_path = outdir / f"imerg_ea_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot IMERG EA daily precipitation")
    parser.add_argument("--store", type=str, default="./imerg_ea_local",
                        help="Path to local Icechunk store")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace dataset repo (e.g. E4DRR/icechunk-stores)")
    parser.add_argument("--hf-prefix", type=str, default=None,
                        help="Store prefix in HF repo (e.g. test3_imerg-update-test)")
    parser.add_argument("--geojson", type=str, default="ea_ghcf_simple.geojson",
                        help="Path to EA countries GeoJSON")
    parser.add_argument("--outdir", type=str, default="./plots",
                        help="Output directory for PNGs")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date to plot (YYYY-MM-DD), defaults to first in store")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date to plot (YYYY-MM-DD), defaults to last in store")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = open_store(args)

    # Select date range
    if args.start_date or args.end_date:
        start = args.start_date or str(ds.time.values[0])[:10]
        end = args.end_date or str(ds.time.values[-1])[:10]
        ds = ds.sel(time=slice(start, end))
        print(f"Selected {ds.sizes['time']} days: {start} to {end}")
    else:
        print(f"Plotting all {ds.sizes['time']} days in store")

    if ds.sizes["time"] == 0:
        print("No timesteps in selected range!")
        return

    gdf = gpd.read_file(args.geojson)

    n_time = ds.sizes["time"]
    for t in range(n_time):
        plot_day(ds["precipitation"].isel(time=t), gdf, outdir)

    print(f"\nDone: {n_time} plots saved to {outdir}/")


if __name__ == "__main__":
    main()
