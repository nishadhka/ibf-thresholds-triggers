#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "netcdf4",
#     "zarr>=3",
#     "icechunk>=0.1",
#     "dask",
#     "coiled",
#     "distributed",
#     "gcsfs",
# ]
# ///
"""
ECMWF SPI East Africa — Icechunk Store + Pencil Rechunking
============================================================

Ingests monthly SPI NetCDF files (SPI1, SPI3, SPI6, SPI12, SPI24, SPI36, SPI48)
from the ECMWF ERA5-Drought dataset into an Icechunk store, then rechunks to
pencil format (full time × 5 lat × 5 lon) for fast per-pixel time-series access.

Source: https://cds.climate.copernicus.eu/datasets/derived-drought-historical-monthly

The data covers East Africa (lat: -14.25 to 25.25, lon: 19.75 to 53.75) at
0.25° resolution, monthly from 1940 onwards.  Each SPI accumulation period
is stored as a separate variable (SPI1, SPI3, etc.) in the same Zarr store.

Subcommands:

  init     — Create empty template store with correct dimensions
  fill     — Read local NetCDF files and write into the Icechunk store
  rechunk  — Rechunk to pencil chunks → new Zarr store
  verify   — Inspect store contents

Usage:
    # Step 1: Create empty template
    uv run ecmwf_spi_icechunk.py init --data-dir data/

    # Step 2: Fill with data from local NetCDF files
    uv run ecmwf_spi_icechunk.py fill --data-dir data/

    # Step 3: Rechunk to pencil chunks (local)
    uv run ecmwf_spi_icechunk.py rechunk \
        --source-path ecmwf_spi_ea_store \
        --target-path ecmwf_spi_ea_pencil

    # Step 4: Verify
    uv run ecmwf_spi_icechunk.py verify --store-path ecmwf_spi_ea_pencil

    # For GCS storage, add --gcs-bucket and --gcs-prefix flags.

Author: AI Assistant
Date: 2026-02-17
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ecmwf_spi_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GCS_BUCKET = "cpc_awc"

# SPI accumulation periods available in the dataset
SPI_PERIODS = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]

# Chunk sizes
ZARR_CHUNK_TIME = 120     # 10 years of monthly data per chunk
PENCIL_CHUNK_SIZE = (-1, 5, 5)  # full time × 5 lat × 5 lon

# Fill value for missing data (SPI uses NaN for ocean/missing)
FILL_VALUE = np.nan


# ─── Helpers ────────────────────────────────────────────────────────────────


def discover_files(data_dir: str):
    """Discover and catalog all SPI NetCDF files by period.

    Returns dict: {spi_period: [(datetime, filepath), ...]} sorted by time.
    """
    data_path = Path(data_dir)
    catalog = {}

    for spi in SPI_PERIODS:
        pattern = f"{spi}_gamma_global_era5_moda_ref1991to2020_*.nc"
        files = sorted(data_path.glob(pattern))
        if not files:
            continue

        entries = []
        for f in files:
            # Extract YYYYMM from filename
            # e.g. SPI3_gamma_global_era5_moda_ref1991to2020_194003.area-subset...nc
            stem = f.stem  # before .nc
            parts = stem.split("_")
            # The date part is after "ref1991to2020_"
            date_part = parts[5].split(".")[0]  # "194003" from "194003.area-subset..."
            year = int(date_part[:4])
            month = int(date_part[4:6])
            dt = pd.Timestamp(year=year, month=month, day=1)
            entries.append((dt, str(f)))

        entries.sort(key=lambda x: x[0])
        catalog[spi] = entries
        logger.info(f"  {spi}: {len(entries)} files [{entries[0][0]} .. {entries[-1][0]}]")

    return catalog


def get_spatial_coords(sample_file: str):
    """Read lat/lon coordinates from a sample NetCDF file."""
    import xarray as xr

    ds = xr.open_dataset(sample_file)
    lat = ds["lat"].values.copy()
    lon = ds["lon"].values.copy()
    ds.close()
    return lat, lon


# ─── Phase 1: init ─────────────────────────────────────────────────────────


def init_store(args):
    """Create empty template Icechunk store for all SPI variables."""
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating ECMWF SPI template store")
    logger.info("=" * 60)
    start = time.time()

    # Discover files
    logger.info(f"Scanning data directory: {args.data_dir}")
    catalog = discover_files(args.data_dir)

    if not catalog:
        logger.error("No SPI files found!")
        return

    # Get spatial coords from first available file
    first_spi = list(catalog.keys())[0]
    sample_file = catalog[first_spi][0][1]
    lat, lon = get_spatial_coords(sample_file)
    n_lat = len(lat)
    n_lon = len(lon)
    logger.info(f"  Lat: {n_lat} points [{lat[0]:.2f} .. {lat[-1]:.2f}]")
    logger.info(f"  Lon: {n_lon} points [{lon[0]:.2f} .. {lon[-1]:.2f}]")

    # Build unified time coordinate from the longest series (SPI1)
    # All SPI periods share the same grid; shorter ones just start later
    all_times = set()
    for spi, entries in catalog.items():
        for dt, _ in entries:
            all_times.add(dt)
    time_coords = np.array(sorted(all_times), dtype="datetime64[ns]")
    n_time = len(time_coords)
    logger.info(f"  Time: {n_time} months [{time_coords[0]} .. {time_coords[-1]}]")

    # Create template dataset with all SPI variables
    data_vars = {}
    for spi in SPI_PERIODS:
        if spi not in catalog:
            continue
        data_vars[spi] = (
            ("time", "lat", "lon"),
            da.full(
                (n_time, n_lat, n_lon),
                np.nan,
                chunks=(n_time, n_lat, n_lon),
                dtype=np.float32,
            ),
            {
                "long_name": f"Standardized Precipitation Index ({spi})",
                "units": "dimensionless",
            },
        )

    template = xr.Dataset(
        data_vars,
        coords={
            "time": time_coords,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF ERA5-Drought SPI — East Africa",
            "source": "ECMWF CDS derived-drought-historical-monthly",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "resolution": "0.25 x 0.25 degrees",
            "reference_period": "1991-2020",
            "Conventions": "CF-1.8",
        },
    )
    logger.info(f"  Template:\n{template}")

    # Set up storage
    if args.local:
        store_path = args.local
    else:
        store_path = args.store_path

    if args.gcs_bucket:
        import icechunk
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )
    else:
        import icechunk
        storage = icechunk.local_filesystem_storage(path=store_path)

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    # Write metadata only (compute=False)
    session = repo.writable_session("main")
    chunk_time = min(ZARR_CHUNK_TIME, n_time)
    encoding = {
        spi: {"chunks": (chunk_time, n_lat, n_lon)}
        for spi in data_vars
    }
    template.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding=encoding,
        consolidated=False,
    )
    session.commit("initialize ECMWF SPI template")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("INIT COMPLETE")
    logger.info(f"  Shape: ({n_time}, {n_lat}, {n_lon})")
    logger.info(f"  Variables: {list(data_vars.keys())}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


# ─── Phase 2: fill ─────────────────────────────────────────────────────────


def fill_store(args):
    """Fill the store with SPI data from local NetCDF files.

    Reads each SPI period's monthly files, concatenates in batches,
    and writes into the Icechunk store using region writes.
    """
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: Populating SPI store from local NetCDF files")
    logger.info("=" * 60)
    overall_start = time.time()

    # Discover files
    catalog = discover_files(args.data_dir)

    # Open target store
    if args.gcs_bucket:
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )
    else:
        store_path = args.local or args.store_path
        storage = icechunk.local_filesystem_storage(path=store_path)

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default(),
    )

    # Read the template to get the time index
    session = repo.readonly_session("main")
    ds_template = xr.open_zarr(session.store, consolidated=False)
    store_times = pd.DatetimeIndex(ds_template.time.values)
    ds_template.close()

    BATCH_SIZE = args.batch_size  # months per commit

    for spi, entries in catalog.items():
        logger.info(f"\n--- Filling {spi} ({len(entries)} files) ---")

        for batch_start in range(0, len(entries), BATCH_SIZE):
            batch = entries[batch_start : batch_start + BATCH_SIZE]
            batch_times = [e[0] for e in batch]
            batch_files = [e[1] for e in batch]

            # Find time indices in the store
            t_indices = []
            for bt in batch_times:
                idx = store_times.get_loc(bt)
                t_indices.append(idx)

            if not t_indices:
                continue

            t_start = min(t_indices)
            t_end = max(t_indices) + 1

            # Read and concatenate this batch
            datasets = []
            for f in batch_files:
                ds = xr.open_dataset(f)
                # Rename variable to match store (file might have SPI3, store has SPI3)
                var_name = [v for v in ds.data_vars if v.startswith("SPI")][0]
                datasets.append(ds[[var_name]])

            ds_batch = xr.concat(datasets, dim="time").sortby("time")
            for d in datasets:
                d.close()

            # Convert to float32
            var_name = [v for v in ds_batch.data_vars][0]
            ds_batch[var_name] = ds_batch[var_name].astype(np.float32)

            # Write to store
            session = repo.writable_session("main")
            ds_write = xr.Dataset({
                var_name: (("time", "lat", "lon"), ds_batch[var_name].values),
            })
            ds_write.to_zarr(
                session.store,
                region={"time": slice(t_start, t_end)},
                consolidated=False,
            )
            session.commit(
                f"fill {spi} months {batch_start}-{batch_start + len(batch) - 1}"
            )

            logger.info(
                f"  {spi} batch {batch_start//BATCH_SIZE + 1}: "
                f"wrote {len(batch)} months (t[{t_start}:{t_end}])"
            )
            del ds_batch

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


# ─── Phase 3: rechunk ──────────────────────────────────────────────────────


def rechunk_store(args):
    """Rechunk to pencil chunks using Dask.

    Reads from the Icechunk store and writes to a new Zarr store
    with pencil chunks (full time × 5 lat × 5 lon).
    """
    import dask
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("RECHUNK: Converting to pencil chunks")
    logger.info("=" * 60)
    overall_start = time.time()

    dask.config.set({
        "array.rechunk.method": "p2p",
        "optimization.fuse.active": False,
    })

    # Open source store
    if args.gcs_bucket and not args.source_path.startswith("/"):
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.source_gcs_prefix,
            service_account_file=args.service_account,
        )
        repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default(),
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)
    else:
        storage = icechunk.local_filesystem_storage(path=args.source_path)
        repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default(),
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)

    n_time = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    logger.info(f"  Source: time={n_time}, lat={n_lat}, lon={n_lon}")

    # Target pencil chunk sizes
    chunk_time = n_time if args.chunk_time == -1 else args.chunk_time
    chunk_lat = args.chunk_lat
    chunk_lon = args.chunk_lon
    pencil_chunks = {"time": chunk_time, "lat": chunk_lat, "lon": chunk_lon}

    n_lat_chunks = -(-n_lat // chunk_lat)
    n_lon_chunks = -(-n_lon // chunk_lon)
    n_target_chunks = n_lat_chunks * n_lon_chunks
    logger.info(f"  Target chunks: ({chunk_time}, {chunk_lat}, {chunk_lon})")
    logger.info(f"  Total target chunks: {n_target_chunks}")

    # Rechunk
    ds_rechunked = ds.chunk(pencil_chunks)

    # Write target
    target_path = args.target_path
    storage_options = None
    if target_path.startswith("gs://"):
        import json
        with open(args.service_account) as f:
            sa_info = json.load(f)
        storage_options = {"token": sa_info}

    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"chunks": (chunk_time, chunk_lat, chunk_lon)}

    logger.info(f"  Writing to: {target_path}")
    ds_rechunked.to_zarr(
        target_path,
        storage_options=storage_options,
        encoding=encoding,
        mode="w",
        consolidated=True,
    )

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("RECHUNK COMPLETE")
    logger.info(f"  Target: {target_path}")
    logger.info(f"  Chunks: ({chunk_time}, {chunk_lat}, {chunk_lon})")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


# ─── Phase 4: verify ───────────────────────────────────────────────────────


def verify_store(args):
    """Inspect and verify the SPI store."""
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting SPI store")
    logger.info("=" * 60)

    store_path = args.store_path
    storage_options = None

    if store_path.startswith("gs://"):
        import json
        with open(args.service_account) as f:
            sa_info = json.load(f)
        storage_options = {"token": sa_info}

    if args.store_type == "zarr":
        ds = xr.open_zarr(store_path, consolidated=True,
                          storage_options=storage_options)
    else:
        import icechunk
        if args.gcs_bucket:
            storage = icechunk.gcs_storage(
                bucket=args.gcs_bucket,
                prefix=args.gcs_prefix or store_path,
                service_account_file=args.service_account,
            )
        else:
            storage = icechunk.local_filesystem_storage(path=store_path)
        repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default(),
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    if "time" in ds.dims:
        logger.info(f"  Time: {ds.time.values[0]} → {ds.time.values[-1]}")
    if "lat" in ds.dims:
        logger.info(f"  Lat: {float(ds.lat.values[0]):.2f} → {float(ds.lat.values[-1]):.2f}")
    if "lon" in ds.dims:
        logger.info(f"  Lon: {float(ds.lon.values[0]):.2f} → {float(ds.lon.values[-1]):.2f}")

    # Spot-check each SPI variable
    for var in ds.data_vars:
        logger.info(f"\nVariable '{var}':")
        da = ds[var]
        logger.info(f"  dtype: {da.dtype}, shape: {da.shape}")

        if args.spot_check:
            logger.info("  Loading sample (first 12 months)...")
            try:
                sample = da.isel(time=slice(0, 12)).load()
                vals = sample.values
                valid = ~np.isnan(vals)
                logger.info(f"  Valid: {valid.sum()}/{vals.size} ({100*valid.mean():.1f}%)")
                if valid.any():
                    logger.info(f"  Min: {float(np.nanmin(vals)):.4f}")
                    logger.info(f"  Max: {float(np.nanmax(vals)):.4f}")
                    logger.info(f"  Mean: {float(np.nanmean(vals)):.4f}")
                    logger.info(f"  Std: {float(np.nanstd(vals)):.4f}")
            except Exception as e:
                logger.error(f"  Spot-check failed: {e}")

    # Addis Ababa spot check (lat ~9.0, lon ~38.7)
    logger.info("\n--- Addis Ababa Spot Check (lat≈9.0, lon≈38.7) ---")
    try:
        for var in ["SPI3", "SPI12"]:
            if var in ds.data_vars:
                point = ds[var].sel(lat=9.0, lon=38.75, method="nearest").load()
                vals = point.values
                valid = ~np.isnan(vals)
                logger.info(
                    f"  {var}: {valid.sum()} valid months, "
                    f"mean={float(np.nanmean(vals)):.3f}, "
                    f"std={float(np.nanstd(vals)):.3f}, "
                    f"min={float(np.nanmin(vals)):.3f}"
                )
    except Exception as e:
        logger.warning(f"  Spot check failed: {e}")

    logger.info("\nVerification complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ECMWF SPI East Africa — Icechunk Store + Pencil Rechunking",
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ──
    p_init = sub.add_parser("init", help="Create empty template store")
    p_init.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing SPI NetCDF files")
    p_init.add_argument("--store-path", type=str, default="ecmwf_spi_ea_store",
                        help="Local path for the Icechunk store")
    p_init.add_argument("--local", type=str, default=None)
    p_init.add_argument("--gcs-bucket", type=str, default=None)
    p_init.add_argument("--gcs-prefix", type=str, default="ecmwf_spi_ea")
    p_init.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)

    # ── fill ──
    p_fill = sub.add_parser("fill", help="Fill store from local NetCDF files")
    p_fill.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing SPI NetCDF files")
    p_fill.add_argument("--store-path", type=str, default="ecmwf_spi_ea_store",
                        help="Local path for the Icechunk store")
    p_fill.add_argument("--local", type=str, default=None)
    p_fill.add_argument("--gcs-bucket", type=str, default=None)
    p_fill.add_argument("--gcs-prefix", type=str, default="ecmwf_spi_ea")
    p_fill.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_fill.add_argument("--batch-size", type=int, default=120,
                        help="Months per commit batch (default: 120 = 10 years)")

    # ── rechunk ──
    p_rechunk = sub.add_parser("rechunk", help="Rechunk to pencil chunks")
    p_rechunk.add_argument("--source-path", type=str, default="ecmwf_spi_ea_store",
                           help="Source Icechunk store path")
    p_rechunk.add_argument("--source-gcs-prefix", type=str, default=None)
    p_rechunk.add_argument("--target-path", type=str, required=True,
                           help="Target Zarr store path (local or gs://...)")
    p_rechunk.add_argument("--gcs-bucket", type=str, default=None)
    p_rechunk.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_rechunk.add_argument("--chunk-time", type=int, default=-1)
    p_rechunk.add_argument("--chunk-lat", type=int, default=5)
    p_rechunk.add_argument("--chunk-lon", type=int, default=5)

    # ── verify ──
    p_verify = sub.add_parser("verify", help="Inspect store contents")
    p_verify.add_argument("--store-path", type=str, default="ecmwf_spi_ea_store")
    p_verify.add_argument("--store-type", type=str, default="icechunk",
                          choices=["icechunk", "zarr"])
    p_verify.add_argument("--gcs-bucket", type=str, default=None)
    p_verify.add_argument("--gcs-prefix", type=str, default=None)
    p_verify.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_verify.add_argument("--spot-check", action="store_true", default=True)
    p_verify.add_argument("--no-spot-check", action="store_false", dest="spot_check")

    args = parser.parse_args()

    if args.command == "init":
        init_store(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "rechunk":
        rechunk_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
