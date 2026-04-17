#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "earthengine-api>=1.0",
#     "xee>=0.0.20",
#     "icechunk>=0.1.0",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "matplotlib>=3.8.0",
#     "cartopy>=0.22.0",
#     "geopandas>=0.14.0",
#     "scipy>=1.11.0",
# ]
# ///
"""Validate the source.coop IMERG-HH icechunk store against GEE IMERG v7.

Computes the 24-hour precipitation total (mm/day) over East Africa for a given
date from both sources and renders a 3-panel cartopy figure
(GEE | icechunk | difference) with country borders overlaid.

Default date is 2026-03-06 — flagged as suspicious (icechunk store reports
zero rainfall while observations show heavy rain over EA on that date).

Usage:
    uv run --python 3.12 compare_imerg_gee_vs_icechunk.py --date 2026-03-06
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import ee
import geopandas as gpd
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/imerg_hh_icechunk"

# East Africa bbox matches the icechunk store grid
EA_LON_MIN, EA_LAT_MIN, EA_LON_MAX, EA_LAT_MAX = 19.5, -14.5, 54.0, 25.5
GEE_IMERG = "NASA/GPM_L3/IMERG_V07"


def init_ee(sa_path: str) -> None:
    info = json.loads(Path(sa_path).read_text())
    creds = ee.ServiceAccountCredentials(info["client_email"], sa_path)
    ee.Initialize(
        creds,
        project=info["project_id"],
        opt_url="https://earthengine-highvolume.googleapis.com",
    )


def fetch_gee_daily_mm(date_str: str) -> xr.DataArray:
    start = ee.Date(date_str)
    end = start.advance(1, "day")
    ic = (
        ee.ImageCollection(GEE_IMERG)
        .filterDate(start, end)
        .select("precipitation")
    )
    region = ee.Geometry.Rectangle(
        [EA_LON_MIN, EA_LAT_MIN, EA_LON_MAX, EA_LAT_MAX]
    )
    proj = ic.first().select(0).projection()
    ds = xr.open_dataset(
        ic, engine="ee", projection=proj, geometry=region,
    )
    # half-hourly mm/hr -> daily mm: sum * 0.5 hr
    da = (ds["precipitation"].sum(dim="time") * 0.5).load()
    # xee returns lon, lat with lat descending — normalize names/order
    if "lon" in da.dims:
        da = da.rename({"lon": "x", "lat": "y"})
    elif "longitude" in da.dims:
        da = da.rename({"longitude": "x", "latitude": "y"})
    da = da.sortby("y").sortby("x").transpose("y", "x")
    return da


def fetch_icechunk_daily_mm(date_str: str) -> tuple[xr.DataArray, dict]:
    storage = icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=SOURCE_COOP_PREFIX,
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    day = ds["precipitation"].sel(time=date_str)
    arr = day.load()
    diag = {
        "n_timesteps": int(arr.sizes["time"]),
        "first_time": str(arr.time.values[0]),
        "last_time": str(arr.time.values[-1]),
        "non_nan_frac": float(np.isfinite(arr.values).mean()),
        "max_half_hour_mm_hr": float(np.nanmax(arr.values)),
    }
    daily = (arr.sum(dim="time") * 0.5).rename({"lon": "x", "lat": "y"})
    daily = daily.sortby("y").sortby("x").transpose("y", "x")
    return daily, diag


def plot_three(da_gee, da_ic, gdf, date_str, outpath):
    da_gee_on_ic = da_gee.interp(
        x=da_ic["x"], y=da_ic["y"], method="linear"
    )
    diff = da_ic - da_gee_on_ic

    vmax = max(float(np.nanmax(da_gee.values)), float(np.nanmax(da_ic.values)))
    vmax = max(min(vmax, 150.0), 1.0)
    dmax = max(float(np.nanmax(np.abs(diff.values))), 1.0)
    dmax = min(dmax, 150.0)

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    panels = [
        (axes[0], da_gee, "Blues", 0, vmax,
         f"GEE IMERG v7 (24-h sum) — {date_str}"),
        (axes[1], da_ic, "Blues", 0, vmax,
         f"source.coop icechunk store (24-h sum) — {date_str}"),
        (axes[2], diff, "RdBu_r", -dmax, dmax,
         "Difference: icechunk − GEE"),
    ]
    for ax, da, cmap, vmn, vmx, title in panels:
        im = ax.pcolormesh(
            da["x"], da["y"], da.values,
            cmap=cmap, vmin=vmn, vmax=vmx,
            transform=ccrs.PlateCarree(),
        )
        gdf.boundary.plot(
            ax=ax, edgecolor="black", linewidth=0.6,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.4, color="0.3")
        ax.set_extent([21, 53, -12, 23], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(title, fontsize=11)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                            pad=0.06, shrink=0.85)
        cbar.set_label("mm/day")

    fig.suptitle(
        f"IMERG v7 24-hour rainfall — GEE vs source.coop icechunk — {date_str}",
        fontsize=13,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-03-06")
    parser.add_argument("--sa", default="ee-service-account.json")
    parser.add_argument(
        "--geojson",
        default="/scratch/notebook/bn-ibf/flood_ibf/ea_ghcf_simple.geojson",
    )
    parser.add_argument("--out", default="imerg_gee_vs_icechunk_{date}.png")
    args = parser.parse_args()
    out = args.out.format(date=args.date)

    print(f"Init GEE with {args.sa} ...")
    init_ee(args.sa)

    print(f"Fetching GEE IMERG v7 for {args.date} ...")
    da_gee = fetch_gee_daily_mm(args.date)
    print(f"  GEE  : shape={tuple(da_gee.shape)} "
          f"max={float(da_gee.max()):.2f} mm/day "
          f"mean={float(da_gee.mean()):.3f} mm/day")

    print(f"Reading source.coop icechunk for {args.date} ...")
    da_ic, diag = fetch_icechunk_daily_mm(args.date)
    print(f"  Store diag: {diag}")
    print(f"  Store: shape={tuple(da_ic.shape)} "
          f"max={float(da_ic.max()):.2f} mm/day "
          f"mean={float(da_ic.mean()):.3f} mm/day")

    gdf = gpd.read_file(args.geojson)
    plot_three(da_gee, da_ic, gdf, args.date, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
