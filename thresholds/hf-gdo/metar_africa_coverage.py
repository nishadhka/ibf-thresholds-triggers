#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.1.0",
#     "numpy>=1.26.0",
#     "pyarrow>=15.0.0",
#     "gcsfs>=2024.2.0",
#     "matplotlib>=3.8.0",
#     "cartopy>=0.22.0",
# ]
# ///
"""
METAR station coverage over East Africa for 2020-01.

Reads one monthly partition from the WeatherBench2 METAR mirror, filters
to the EA bbox, and produces:

  1. metar_ea_stations_2020-01.png  -- station map coloured by coverage %
  2. metar_ea_coverage_2020-01.png  -- coverage histogram + obs/day bar
  3. console summary: # stations, # obs, top/bottom by reporting frequency

Dataset temporal range: 2001-07 to 2023-12 (23 years).
"""

import sys

import gcsfs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# WeatherBench2 METAR partition (January 2020)
PARQUET_URL = (
    "gs://weatherbench2/datasets/metar/metar-timeNominal-by-month/"
    "year=2020/month=1/2020-01.parquet"
)

# East Africa bbox (matches GDO ingest scripts)
LAT_MIN, LAT_MAX = -14.5, 25.5
LON_MIN, LON_MAX = 19.5, 54.0

# January 2020: 31 days × 24 hourly slots = 744 expected per station
EXPECTED_SLOTS = 31 * 24


def load_partition(url):
    print(f"Reading {url} ...")
    # WB2 bucket is public — use anonymous gcsfs so we don't pick up
    # whatever default credentials the VM happens to have.
    fs = gcsfs.GCSFileSystem(token="anon")
    df = pd.read_parquet(url, filesystem=fs)
    print(f"  total rows: {len(df):,}  unique stations: {df.stationName.nunique():,}")
    return df


def filter_bbox(df):
    afr = df[
        (df.latitude >= LAT_MIN)
        & (df.latitude <= LAT_MAX)
        & (df.longitude >= LON_MIN)
        & (df.longitude <= LON_MAX)
    ].copy()
    print(f"  rows: {len(afr):,}  stations: {afr.stationName.nunique():,}")
    return afr


def per_station_summary(afr):
    # Per-station summary: location + obs count + coverage % over 744 slots
    grp = (
        afr.groupby("stationName")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            locationName=("locationName", "first"),
            obs=("stationName", "size"),
        )
        .reset_index()
    )
    grp["coverage_pct"] = (grp["obs"] / EXPECTED_SLOTS * 100.0).clip(upper=100.0)
    grp = grp.sort_values("obs", ascending=False).reset_index(drop=True)
    return grp


def plot_stations_map(grp, afr, out_path):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN - 1, LON_MAX + 1, LAT_MIN - 1, LAT_MAX + 1],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f5f0e8")
    ax.add_feature(cfeature.OCEAN, facecolor="#cde3f0")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    sc = ax.scatter(
        grp["longitude"], grp["latitude"],
        c=grp["coverage_pct"],
        cmap="viridis",
        s=30, edgecolor="black", linewidth=0.3,
        vmin=0, vmax=100,
        transform=ccrs.PlateCarree(),
    )
    cb = plt.colorbar(sc, ax=ax, orientation="vertical",
                      shrink=0.7, pad=0.04)
    cb.set_label("Coverage (% of 744 hourly slots reported)")

    ax.set_title(
        f"METAR stations over East Africa — January 2020\n"
        f"{len(grp):,} stations  ·  {len(afr):,} observations"
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_coverage(afr, grp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1) Histogram of per-station coverage %
    axes[0].hist(grp["coverage_pct"], bins=np.arange(0, 105, 5),
                 color="#4a7ab7", edgecolor="white")
    axes[0].set_xlabel("Coverage (% of 744 hourly slots)")
    axes[0].set_ylabel("# stations")
    axes[0].set_title(
        f"Reporting completeness — {len(grp):,} EA stations"
    )
    axes[0].axvline(grp["coverage_pct"].median(), color="#d9534f",
                    linewidth=1, linestyle="--",
                    label=f"median {grp['coverage_pct'].median():.0f}%")
    axes[0].legend()

    # 2) Obs per day with twin axis for distinct stations/day
    afr_d = afr.copy()
    afr_d["date"] = pd.to_datetime(afr_d["timeObs"]).dt.date
    daily = afr_d.groupby("date").agg(
        obs=("stationName", "size"),
        stations=("stationName", "nunique"),
    )
    ax2 = axes[1]
    ax2.bar(range(len(daily)), daily["obs"], color="#4a7ab7",
            label="# observations")
    ax2.set_xlabel("Day of January 2020")
    ax2.set_ylabel("# observations / day", color="#4a7ab7")
    ax2.tick_params(axis="y", labelcolor="#4a7ab7")
    ax2.set_xticks(range(0, len(daily), 5))
    ax2.set_xticklabels([str(d.day) for d in daily.index[::5]])
    ax3 = ax2.twinx()
    ax3.plot(range(len(daily)), daily["stations"], color="#d9534f",
             marker="o", linewidth=1.5, label="# stations reporting")
    ax3.set_ylabel("# distinct stations / day", color="#d9534f")
    ax3.tick_params(axis="y", labelcolor="#d9534f")
    ax2.set_title("Daily observation volume")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    df = load_partition(PARQUET_URL)
    afr = filter_bbox(df)
    if len(afr) == 0:
        print("No observations in EA bbox.")
        sys.exit(1)

    grp = per_station_summary(afr)

    pcts = grp["coverage_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print("\nCoverage percentiles (% of 744 slots):")
    for k in ("10%", "25%", "50%", "75%", "90%"):
        print(f"  p{k}: {pcts[k]:.1f}%")

    print("\nTop 10 by obs:")
    print(grp.head(10)[["stationName", "locationName", "obs", "coverage_pct"]]
          .to_string(index=False))
    print("\nBottom 10 by obs:")
    print(grp.tail(10)[["stationName", "locationName", "obs", "coverage_pct"]]
          .to_string(index=False))

    out1 = "metar_ea_stations_2020-01.png"
    out2 = "metar_ea_coverage_2020-01.png"
    plot_stations_map(grp, afr, out1)
    plot_coverage(afr, grp, out2)


if __name__ == "__main__":
    main()
