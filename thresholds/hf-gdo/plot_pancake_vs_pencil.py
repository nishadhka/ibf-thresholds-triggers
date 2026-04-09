#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "icechunk>=0.1.0",
#     "matplotlib>=3.8.0",
#     "s3fs>=2024.1.0",
# ]
# ///
"""
Compare pancake vs pencil stores — each plot plays to its chunk layout's strength:
  Top row:    5 spatial maps from PANCAKE store (fast: 1 chunk per map)
  Bottom row: time-series at 5 locations from PENCIL store (fast: 1 chunk per series)

This demonstrates why each layout exists:
  Pancake (12, 800, 690) → fast spatial maps at one timestep
  Pencil  (all, 5, 5)    → fast time-series at one location

Usage:
    uv run plot_pancake_vs_pencil.py
    uv run plot_pancake_vs_pencil.py --dataset rfe2
    uv run plot_pancake_vs_pencil.py --n-steps 3
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_BASE_PREFIX = "e4drr-project/observations"

DATASETS = {
    "chirps_spi": {
        "pancake_prefix": "chirps_spi_icechunk",
        "pencil_path": f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/chirps_spi_pencil_zarr",
        "var": "spi_gamma_3_month",
        "label": "CHIRPS SPI-3",
    },
    "gdo_fpar": {
        "pancake_prefix": "gdo_fpar_icechunk",
        "pencil_path": f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/gdo_fpar_pencil_zarr",
        "var": "fapan",
        "label": "fAPAR Anomaly",
    },
    "gdo_sma": {
        "pancake_prefix": "gdo_sma_icechunk",
        "pencil_path": f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/gdo_sma_pencil_zarr",
        "var": "smian",
        "label": "Soil Moisture Anomaly",
    },
    "rfe2": {
        "pancake_prefix": "rfe2_icechunk",
        "pencil_path": f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/rfe2_pencil_zarr",
        "var": "rfe2",
        "label": "RFE2 Rainfall",
    },
}


def open_icechunk_store(prefix):
    """Open an Icechunk store from source.coop S3 (anonymous)."""
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=f"{SOURCE_COOP_BASE_PREFIX}/{prefix}",
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def open_pencil_store(info):
    """Open pencil store (plain Zarr on S3)."""
    import xarray as xr

    return xr.open_zarr(info["pencil_path"], storage_options={"anon": True},
                        consolidated=True)


def plot_dataset(ds_name, info, n_steps=5):
    """Plot spatial maps from pancake + time-series from pencil."""

    print(f"\n  Opening {ds_name} pancake...")
    ds_pan = open_icechunk_store(info["pancake_prefix"])

    print(f"  Opening {ds_name} pencil...")
    ds_pen = open_pencil_store(info)

    var = info["var"]
    n_time = ds_pan.sizes["time"]
    n_lat = ds_pan.sizes["lat"]
    n_lon = ds_pan.sizes["lon"]

    pan_chunks = ds_pan[var].encoding.get("chunks", "?")
    pen_chunks = ds_pen[var].encoding.get("chunks", "?")
    print(f"  Pancake: {dict(ds_pan.sizes)}, chunks={pan_chunks}")
    print(f"  Pencil:  {dict(ds_pen.sizes)}, chunks={pen_chunks}")

    # Pick 5 evenly spaced time indices for spatial maps (pancake)
    time_indices = np.linspace(0, n_time - 1, n_steps, dtype=int)

    # Pick 5 scattered lat/lon points for time-series (pencil)
    lat_indices = np.linspace(n_lat // 4, 3 * n_lat // 4, n_steps, dtype=int)
    lon_indices = np.linspace(n_lon // 4, 3 * n_lon // 4, n_steps, dtype=int)

    print(f"  Spatial maps at t={time_indices.tolist()}")
    print(f"  Time-series at lat={lat_indices.tolist()}, lon={lon_indices.tolist()}")

    fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8),
                             constrained_layout=True)
    fig.suptitle(f"{info['label']}\n"
                 f"Pancake {pan_chunks} — spatial maps  |  "
                 f"Pencil {pen_chunks} — time-series",
                 fontsize=13, fontweight="bold")

    # Top row: spatial maps from PANCAKE (fast — reads 1 chunk per map)
    print(f"  Loading {n_steps} spatial maps from pancake...")
    for col, tidx in enumerate(time_indices):
        data = ds_pan[var].isel(time=tidx).values
        im = axes[0, col].imshow(data, aspect="auto", origin="lower")
        t_str = str(ds_pan.time.values[tidx])[:10]
        axes[0, col].set_title(f"t={tidx}  {t_str}", fontsize=9)
        if col == 0:
            axes[0, col].set_ylabel("Pancake\n(spatial map)", fontsize=11)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        plt.colorbar(im, ax=axes[0, col], shrink=0.7)

    # Bottom row: time-series from PENCIL (fast — reads 1 chunk per series)
    print(f"  Loading {n_steps} time-series from pencil...")
    for col in range(n_steps):
        li, lo = int(lat_indices[col]), int(lon_indices[col])
        ts = ds_pen[var].isel(lat=li, lon=lo).values
        time_vals = ds_pen.time.values

        axes[1, col].plot(time_vals, ts, linewidth=0.5, color="steelblue")
        valid = np.count_nonzero(~np.isnan(ts))
        lat_val = float(ds_pen.lat.values[li])
        lon_val = float(ds_pen.lon.values[lo])
        axes[1, col].set_title(f"({lat_val:.1f}, {lon_val:.1f})\n"
                               f"{valid}/{len(ts)} valid", fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel("Pencil\n(time-series)", fontsize=11)
        axes[1, col].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1, col].tick_params(axis="y", labelsize=7)

    out = f"/scratch/notebook/ibf-thresholds-triggers/thresholds/hf-gdo/{ds_name}_pancake_vs_pencil.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    ds_pan.close()
    ds_pen.close()


def main():
    parser = argparse.ArgumentParser(description="Plot pancake vs pencil comparison")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"],
                        default="all", help="Dataset to plot (default: all)")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Number of time steps / points to plot (default: 5)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for ds_name, info in datasets.items():
        print(f"\nPlotting {ds_name}...")
        plot_dataset(ds_name, info, args.n_steps)

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
