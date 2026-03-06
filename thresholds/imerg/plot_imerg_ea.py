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
# ]
# ///
"""
Plot daily IMERG precipitation from local Icechunk store with EA country borders.
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description="Plot IMERG EA daily precipitation")
    parser.add_argument("--store", type=str, default="./imerg_ea_local",
                        help="Path to local Icechunk store")
    parser.add_argument("--geojson", type=str, default="ea_ghcf_simple.geojson",
                        help="Path to EA countries GeoJSON")
    parser.add_argument("--outdir", type=str, default="./plots",
                        help="Output directory for PNGs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Open Icechunk store
    storage = icechunk.local_filesystem_storage(path=args.store)
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    # Load country boundaries
    gdf = gpd.read_file(args.geojson)

    # Plot each day
    n_time = ds.sizes["time"]
    for t in range(n_time):
        da = ds["precipitation"].isel(time=t).load()
        date_str = str(da.time.values)[:10]

        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Precipitation
        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="Blues",
            vmin=0,
            vmax=min(float(da.max()), 100),
            add_colorbar=False,
        )

        # Country borders
        gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8, transform=ccrs.PlateCarree())

        # Country labels
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y, row["GID_0"],
                transform=ccrs.PlateCarree(),
                fontsize=7, fontweight="bold", ha="center",
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
            )

        # Map features
        ax.coastlines(linewidth=0.5)
        ax.set_extent([21, 53, -12, 23], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
        cbar.set_label("Precipitation (mm/day)")

        ax.set_title(f"IMERG Daily Early Precipitation — {date_str}", fontsize=13)

        out_path = outdir / f"imerg_ea_{date_str}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nDone: {n_time} plots saved to {outdir}/")


if __name__ == "__main__":
    main()
