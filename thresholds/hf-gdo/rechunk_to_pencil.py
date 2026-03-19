#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "icechunk>=0.1.0",
#     "dask>=2024.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Rechunk Icechunk stores from pancake (spatial) to pencil (time-series) layout.

Reads from existing GCS Icechunk store with pancake chunks (small time,
full spatial) and writes to a new GCS Icechunk store with pencil chunks
(full time, small spatial) for fast time-series point queries.

  Pancake: (12, 800, 690)  — good for spatial maps at one timestep
  Pencil:  (all, 5, 5)     — good for time-series at one location

Each dataset gets a clearly named new store:
  chirps_spi_ic_store       → chirps_spi_pencil_ic_store
  gdo_fpar_ic_store         → gdo_fpar_pencil_ic_store
  gdo_sma_ic_store          → gdo_sma_pencil_ic_store
  rfe2_ic_store             → rfe2_pencil_ic_store

Usage:
    # Dry run — show source/target shapes and chunk sizes
    uv run rechunk_to_pencil.py gdo_fpar --dry-run

    # Rechunk a single dataset
    uv run rechunk_to_pencil.py gdo_fpar

    # Rechunk all datasets
    uv run rechunk_to_pencil.py all

    # Custom pencil chunk size
    uv run rechunk_to_pencil.py rfe2 --chunk-lat 10 --chunk-lon 10

    # Verify pencil store
    uv run rechunk_to_pencil.py gdo_fpar --verify
"""

import argparse
import time
from pathlib import Path

import numpy as np

GCS_BUCKET = "cpc_awc"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"

DATASETS = {
    "chirps_spi": {
        "gcs_prefix": "chirps_spi_ic_store",
        "pencil_prefix": "chirps_spi_pencil_ic_store",
        "s3_prefix": "chirps_spi_pencil_icechunk",
    },
    "gdo_fpar": {
        "gcs_prefix": "gdo_fpar_ic_store",
        "pencil_prefix": "gdo_fpar_pencil_ic_store",
        "s3_prefix": "gdo_fpar_pencil_icechunk",
    },
    "gdo_sma": {
        "gcs_prefix": "gdo_sma_ic_store",
        "pencil_prefix": "gdo_sma_pencil_ic_store",
        "s3_prefix": "gdo_sma_pencil_icechunk",
    },
    "rfe2": {
        "gcs_prefix": "rfe2_ic_store",
        "pencil_prefix": "rfe2_pencil_ic_store",
        "s3_prefix": "rfe2_pencil_icechunk",
    },
}


def open_source_store(info, sa_path):
    """Open source Icechunk store (pancake chunks) from GCS."""
    import icechunk
    import xarray as xr

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=info["gcs_prefix"],
        service_account_file=str(sa_path),
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds


def rechunk_dataset(name, info, sa_path, chunk_lat, chunk_lon, dry_run=False):
    """Rechunk one dataset from pancake to pencil chunks."""
    import dask
    import dask.array as da
    import icechunk
    import xarray as xr

    print(f"\n{'='*60}")
    print(f"Rechunk: {name}")
    print(f"{'='*60}")

    # Open source
    print(f"\n  Source: gs://{GCS_BUCKET}/{info['gcs_prefix']}")
    ds_src = open_source_store(info, sa_path)
    print(f"  Shape:  {dict(ds_src.sizes)}")
    for var in ds_src.data_vars:
        src_chunks = ds_src[var].encoding.get("chunks", "unknown")
        print(f"  {var}: pancake chunks = {src_chunks}")

    # Compute pencil chunk sizes
    n_time = ds_src.sizes["time"]
    n_lat = ds_src.sizes["lat"]
    n_lon = ds_src.sizes["lon"]

    pencil_chunks = {"time": n_time, "lat": chunk_lat, "lon": chunk_lon}
    n_lat_chunks = -(-n_lat // chunk_lat)
    n_lon_chunks = -(-n_lon // chunk_lon)
    n_vars = len(ds_src.data_vars)
    chunk_bytes = n_time * chunk_lat * chunk_lon * 4
    total_chunks = n_lat_chunks * n_lon_chunks * n_vars

    print(f"\n  Target: gs://{GCS_BUCKET}/{info['pencil_prefix']}")
    print(f"  Pencil chunks: (time={n_time}, lat={chunk_lat}, lon={chunk_lon})")
    print(f"  Chunk size: {chunk_bytes / (1024**2):.1f} MB")
    print(f"  Total chunks: {total_chunks} ({n_lat_chunks} lat × {n_lon_chunks} lon × {n_vars} vars)")

    total_gb = n_time * n_lat * n_lon * 4 * n_vars / (1024**3)
    print(f"  Total data: {total_gb:.2f} GB")

    if dry_run:
        print("\n  (dry run — no rechunking)")
        ds_src.close()
        return

    # Create target Icechunk store
    print(f"\n  Creating target store...")
    target_storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=info["pencil_prefix"],
        service_account_file=str(sa_path),
    )
    try:
        target_repo = icechunk.Repository.create(
            target_storage, config=icechunk.RepositoryConfig.default()
        )
        print("  Created new repository")
    except Exception:
        target_repo = icechunk.Repository.open(
            target_storage, config=icechunk.RepositoryConfig.default()
        )
        print("  Opened existing repository")

    # Write pencil-chunked template (metadata only)
    target_session = target_repo.writable_session("main")

    template = xr.Dataset(
        {
            var: (
                ds_src[var].dims,
                da.zeros(
                    ds_src[var].shape,
                    chunks=tuple(pencil_chunks.get(d, s) for d, s in zip(ds_src[var].dims, ds_src[var].shape)),
                    dtype=ds_src[var].dtype,
                ),
                ds_src[var].attrs,
            )
            for var in ds_src.data_vars
        },
        coords=ds_src.coords,
        attrs=ds_src.attrs,
    )

    encoding = {
        var: {"chunks": tuple(pencil_chunks.get(d, ds_src.sizes[d]) for d in ds_src[var].dims)}
        for var in ds_src.data_vars
    }

    template.to_zarr(
        target_session.store,
        compute=False,
        mode="w",
        encoding=encoding,
        consolidated=False,
    )
    target_session.commit(f"initialize {name} pencil template ({n_time}, {chunk_lat}, {chunk_lon})")
    print(f"  Template written")
    ds_src.close()

    # Fill pencil store by iterating over spatial tiles
    # Read one pancake slab (all lat/lon for a time chunk), write pencil columns
    print(f"\n  Filling pencil store (spatial tile iteration)...")
    start = time.time()

    # Re-open source with explicit chunks matching stored layout
    ds_src = open_source_store(info, sa_path)

    tiles_done = 0
    for lat_start in range(0, n_lat, chunk_lat):
        lat_end = min(lat_start + chunk_lat, n_lat)

        for lon_start in range(0, n_lon, chunk_lon):
            lon_end = min(lon_start + chunk_lon, n_lon)

            # Read full time-series for this spatial tile from source
            slices = {
                "lat": slice(lat_start, lat_end),
                "lon": slice(lon_start, lon_end),
            }
            tile = ds_src.isel(**slices).load()

            # Write to target store
            target_storage2 = icechunk.gcs_storage(
                bucket=GCS_BUCKET,
                prefix=info["pencil_prefix"],
                service_account_file=str(sa_path),
            )
            target_repo2 = icechunk.Repository.open(
                target_storage2, config=icechunk.RepositoryConfig.default()
            )
            session2 = target_repo2.writable_session("main")

            tile.to_zarr(
                session2.store,
                region={
                    "time": slice(0, n_time),
                    "lat": slice(lat_start, lat_end),
                    "lon": slice(lon_start, lon_end),
                },
                consolidated=False,
            )
            session2.commit(
                f"fill tile lat={lat_start}-{lat_end} lon={lon_start}-{lon_end}"
            )

            tiles_done += 1
            if tiles_done % 50 == 0:
                elapsed = time.time() - start
                pct = 100 * tiles_done / total_chunks * n_vars
                print(f"    [{pct:5.1f}%] {tiles_done} tiles ({elapsed:.0f}s)")

            del tile

    ds_src.close()
    elapsed = time.time() - start
    print(f"\n  Rechunk complete: {tiles_done} tiles in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def verify_pencil(name, info, sa_path):
    """Verify pencil-chunked store."""
    import icechunk
    import xarray as xr

    print(f"\n  Verifying: gs://{GCS_BUCKET}/{info['pencil_prefix']}")

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=info["pencil_prefix"],
        service_account_file=str(sa_path),
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    print(f"  Dimensions: {dict(ds.sizes)}")
    for var in ds.data_vars:
        chunks = ds[var].encoding.get("chunks", "unknown")
        print(f"  {var}: pencil chunks = {chunks}")

    if "time" in ds.coords:
        print(f"  Time: {ds.time.values[0]} → {ds.time.values[-1]} ({len(ds.time)} steps)")

    # Spot check: load one pencil (full time at one point)
    for var in list(ds.data_vars)[:1]:
        ts = ds[var].isel(lat=0, lon=0).values
        valid = np.count_nonzero(~np.isnan(ts))
        print(f"  {var} time-series (0,0): {valid}/{len(ts)} valid ({100*valid/len(ts):.1f}%)")

    ds.close()
    print("  Verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk Icechunk stores from pancake to pencil layout"
    )
    parser.add_argument(
        "dataset",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to rechunk (or 'all')",
    )
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--chunk-lat", type=int, default=5,
                        help="Pencil chunk lat size (default: 5)")
    parser.add_argument("--chunk-lon", type=int, default=5,
                        help="Pencil chunk lon size (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sizes only, don't rechunk")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pencil store")
    args = parser.parse_args()

    sa_path = Path(__file__).parent / args.sa_file
    if not sa_path.exists():
        print(f"ERROR: SA file not found: {sa_path}")
        return

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, info in datasets.items():
        if args.verify:
            verify_pencil(name, info, sa_path)
        else:
            rechunk_dataset(name, info, sa_path, args.chunk_lat, args.chunk_lon, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
