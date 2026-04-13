#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netcdf4",
# ]
# ///
"""
Check estimated sizes of ECMWF SPI stores using xarray .nbytes.

Builds the dataset from local NetCDF files (same shape as the Icechunk
store would have) and reports nbytes for both fill and pencil layouts.

Usage:
    uv run check_store_size.py --data-dir data/
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr

SPI_PERIODS = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]


def main():
    parser = argparse.ArgumentParser(description="Check SPI store sizes via xarray nbytes")
    parser.add_argument("--data-dir", type=str, default="data/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load each SPI period and concatenate along time
    datasets = {}
    for spi in SPI_PERIODS:
        files = sorted(data_dir.glob(f"{spi}_gamma_global_era5_moda_ref1991to2020_*.nc"))
        if not files:
            print(f"  {spi}: no files found, skipping")
            continue

        ds_list = [xr.open_dataset(f) for f in files]
        ds_concat = xr.concat(ds_list, dim="time").sortby("time")
        for d in ds_list:
            d.close()

        var_name = [v for v in ds_concat.data_vars if v.startswith("SPI")][0]
        datasets[spi] = ds_concat[var_name].astype(np.float32)
        print(f"  {spi}: {len(files)} files → shape {datasets[spi].shape}")

    # Build combined dataset (as it would appear in the Icechunk store)
    ds = xr.Dataset({spi: da for spi, da in datasets.items()})

    n_time = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]

    print(f"\n{'='*60}")
    print(f"Combined Dataset")
    print(f"{'='*60}")
    print(f"  Dimensions: time={n_time}, lat={n_lat}, lon={n_lon}")
    print(f"  Variables:  {list(ds.data_vars)}")
    print(f"  ds.nbytes:  {ds.nbytes:,} bytes")
    print(f"              {ds.nbytes / 1024**2:.1f} MB")
    print(f"              {ds.nbytes / 1024**3:.3f} GB")

    # Per-variable breakdown
    print(f"\n  Per-variable nbytes:")
    for var in ds.data_vars:
        nb = ds[var].nbytes
        print(f"    {var}: {nb:,} bytes ({nb / 1024**2:.1f} MB)")

    # Chunk layout estimates
    print(f"\n{'='*60}")
    print(f"Fill Store Layout — chunks (120, {n_lat}, {n_lon})")
    print(f"{'='*60}")
    fill_chunk_bytes = 120 * n_lat * n_lon * 4
    n_time_chunks = -(-n_time // 120)
    n_fill_chunks = n_time_chunks * len(datasets)
    print(f"  Chunk size:   {fill_chunk_bytes / 1024**2:.1f} MB")
    print(f"  Chunks/var:   {n_time_chunks}")
    print(f"  Total chunks: {n_fill_chunks}")
    print(f"  Read 1 pixel: {n_time_chunks} chunks × {fill_chunk_bytes/1024**2:.1f} MB = {n_time_chunks * fill_chunk_bytes / 1024**2:.0f} MB I/O")

    print(f"\n{'='*60}")
    print(f"Pencil Store Layout — chunks ({n_time}, 5, 5)")
    print(f"{'='*60}")
    pencil_chunk_bytes = n_time * 5 * 5 * 4
    n_lat_chunks = -(-n_lat // 5)
    n_lon_chunks = -(-n_lon // 5)
    n_pencil_chunks = n_lat_chunks * n_lon_chunks * len(datasets)
    print(f"  Chunk size:   {pencil_chunk_bytes / 1024:.1f} KB ({pencil_chunk_bytes / 1024**2:.2f} MB)")
    print(f"  Chunks/var:   {n_lat_chunks * n_lon_chunks} ({n_lat_chunks} lat × {n_lon_chunks} lon)")
    print(f"  Total chunks: {n_pencil_chunks}")
    print(f"  Read 1 pixel: 1 chunk × {pencil_chunk_bytes/1024:.0f} KB = {pencil_chunk_bytes/1024:.0f} KB I/O")

    speedup = (n_time_chunks * fill_chunk_bytes) / pencil_chunk_bytes
    print(f"\n  I/O reduction (pencil vs fill): {speedup:.0f}×")

    ds.close()


if __name__ == "__main__":
    main()
