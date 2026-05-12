#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.1.0",
#     "numpy>=1.26.0",
#     "pyarrow>=15.0.0",
#     "matplotlib>=3.8.0",
#     "cartopy>=0.22.0",
# ]
# ///
"""
MAM 2026 (Mar 1 – May 11) METAR coverage over East Africa from the local
Iowa Mesonet backfill parquets.

Outputs:
  metar_ea_mam2026_stations.png  -- station map coloured by coverage %
  metar_ea_mam2026_coverage.png  -- coverage histogram + daily volume
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# East Africa bbox (matches GDO ingest scripts)
LAT_MIN, LAT_MAX = -14.5, 25.5
LON_MIN, LON_MAX = 19.5, 54.0

DATA_ROOT = Path("data/metar_ea_iowa")
PARQUETS = [
    DATA_ROOT / "year=2026" / "month=3" / "2026-03.parquet",
    DATA_ROOT / "year=2026" / "month=4" / "2026-04.parquet",
    DATA_ROOT / "year=2026" / "month=5" / "2026-05.parquet",
]

# MAM window actually present in the parquets: 2026-03-01 → 2026-05-11
WINDOW_START = pd.Timestamp("2026-03-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-05-12", tz="UTC")  # exclusive
# 71 days × 24 hourly slots; SPECIs add half-hour rows but we still benchmark
# coverage against the canonical hourly grid for comparability with Jan 2020.
EXPECTED_SLOTS = int((WINDOW_END - WINDOW_START).total_seconds() // 3600)


def load():
    frames = []
    for p in PARQUETS:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit("No parquets to read.")
    df = pd.concat(frames, ignore_index=True)
    df["timeObs"] = pd.to_datetime(df["timeObs"], utc=True)
    df = df[(df.timeObs >= WINDOW_START) & (df.timeObs < WINDOW_END)]
    print(f"  rows: {len(df):,}  stations: {df.stationName.nunique():,}")
    return df


def per_station_summary(df):
    # Coverage % = # distinct hourly nominal slots reported / expected slots
    df = df.copy()
    df["hour_slot"] = pd.to_datetime(df["timeObs"], utc=True).dt.floor("h")
    grp = (
        df.groupby("stationName")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            locationName=("locationName", "first"),
            obs=("stationName", "size"),
            hours_reported=("hour_slot", "nunique"),
        )
        .reset_index()
    )
    grp["coverage_pct"] = (grp["hours_reported"] / EXPECTED_SLOTS * 100.0).clip(upper=100.0)
    grp = grp.sort_values("obs", ascending=False).reset_index(drop=True)
    return grp


def plot_stations_map(grp, df, out_path):
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
    cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.04)
    cb.set_label(f"Coverage (% of {EXPECTED_SLOTS} hourly slots reported)")

    ax.set_title(
        f"METAR stations over East Africa — MAM 2026 "
        f"({WINDOW_START.date()} → {(WINDOW_END - pd.Timedelta('1D')).date()})\n"
        f"{len(grp):,} stations  ·  {len(df):,} observations"
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_coverage(df, grp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(grp["coverage_pct"], bins=np.arange(0, 105, 5),
                 color="#4a7ab7", edgecolor="white")
    axes[0].set_xlabel(f"Coverage (% of {EXPECTED_SLOTS} hourly slots)")
    axes[0].set_ylabel("# stations")
    axes[0].set_title(f"Reporting completeness — {len(grp):,} EA stations")
    axes[0].axvline(grp["coverage_pct"].median(), color="#d9534f",
                    linewidth=1, linestyle="--",
                    label=f"median {grp['coverage_pct'].median():.0f}%")
    axes[0].legend()

    df_d = df.copy()
    df_d["date"] = pd.to_datetime(df_d["timeObs"], utc=True).dt.date
    daily = df_d.groupby("date").agg(
        obs=("stationName", "size"),
        stations=("stationName", "nunique"),
    )
    ax2 = axes[1]
    ax2.bar(range(len(daily)), daily["obs"], color="#4a7ab7")
    ax2.set_xlabel(f"Day of MAM 2026 ({len(daily)} days)")
    ax2.set_ylabel("# observations / day", color="#4a7ab7")
    ax2.tick_params(axis="y", labelcolor="#4a7ab7")
    step = max(1, len(daily) // 10)
    ax2.set_xticks(range(0, len(daily), step))
    ax2.set_xticklabels([str(d) for d in daily.index[::step]], rotation=45, ha="right")
    ax3 = ax2.twinx()
    ax3.plot(range(len(daily)), daily["stations"], color="#d9534f",
             marker="o", markersize=3, linewidth=1.2)
    ax3.set_ylabel("# distinct stations / day", color="#d9534f")
    ax3.tick_params(axis="y", labelcolor="#d9534f")
    ax2.set_title("Daily observation volume")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print(f"Loading MAM 2026 parquets from {DATA_ROOT}/")
    df = load()
    grp = per_station_summary(df)

    pcts = grp["coverage_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"\nCoverage percentiles (% of {EXPECTED_SLOTS} hourly slots):")
    for k in ("10%", "25%", "50%", "75%", "90%"):
        print(f"  p{k}: {pcts[k]:.1f}%")

    print("\nTop 10 by obs:")
    print(grp.head(10)[["stationName", "locationName", "obs", "hours_reported", "coverage_pct"]]
          .to_string(index=False))
    print("\nBottom 10 by obs:")
    print(grp.tail(10)[["stationName", "locationName", "obs", "hours_reported", "coverage_pct"]]
          .to_string(index=False))

    plot_stations_map(grp, df, "metar_ea_mam2026_stations.png")
    plot_coverage(df, grp, "metar_ea_mam2026_coverage.png")


if __name__ == "__main__":
    main()
