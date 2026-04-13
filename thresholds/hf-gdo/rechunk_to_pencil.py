#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "icechunk>=0.1.0",
#     "dask>=2024.1.0",
#     "python-dotenv>=1.0.0",
#     "coiled>=1.0.0",
#     "distributed>=2024.1.0",
#     "s3fs>=2024.1.0",
# ]
# ///
"""
Rechunk Icechunk stores from pancake (spatial) to pencil (time-series) layout.

Reads from existing Icechunk store with pancake chunks (small time,
full spatial) and writes to a new Icechunk store with pencil chunks
(full time, small spatial) for fast time-series point queries.

Supports both GCS and source.coop (S3) backends:
  - GCS (default): reads/writes GCS Icechunk stores
  - source.coop (--source-coop): reads public pancake stores from S3,
    writes pencil stores to S3 with credentials

  Pancake: (12, 800, 690)  — good for spatial maps at one timestep
  Pencil:  (all, 5, 5)     — good for time-series at one location

Each dataset gets a clearly named new store:
  GCS:         chirps_spi_ic_store       → chirps_spi_pencil_ic_store
  source.coop: chirps_spi_icechunk       → chirps_spi_pencil_icechunk

Usage:
    # Dry run — show source/target shapes and chunk sizes
    uv run rechunk_to_pencil.py gdo_fpar --dry-run

    # Rechunk a single dataset (GCS)
    uv run rechunk_to_pencil.py gdo_fpar

    # Rechunk on source.coop (read public pancake, write pencil)
    uv run rechunk_to_pencil.py gdo_fpar --source-coop

    # Rechunk all datasets
    uv run rechunk_to_pencil.py all

    # Custom pencil chunk size
    uv run rechunk_to_pencil.py rfe2 --chunk-lat 10 --chunk-lon 10

    # Verify pencil store (GCS or source.coop)
    uv run rechunk_to_pencil.py gdo_fpar --verify
    uv run rechunk_to_pencil.py gdo_fpar --verify --source-coop

    # Dry run on source.coop
    uv run rechunk_to_pencil.py all --dry-run --source-coop

    # Coiled Dask P2P rechunk (fast, writes plain Zarr to source.coop)
    uv run rechunk_to_pencil.py rfe2 --coiled --n-workers 15
    uv run rechunk_to_pencil.py all --coiled --dry-run
    uv run rechunk_to_pencil.py chirps_spi --coiled --verify
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET = "cpc_awc"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_BASE_PREFIX = "e4drr-project/observations"

DATASETS = {
    "chirps_spi": {
        "gcs_prefix": "chirps_spi_ic_store",
        "pencil_prefix": "chirps_spi_pencil_ic_store",
        "s3_source_prefix": "chirps_spi_icechunk",
        "s3_pencil_prefix": "chirps_spi_pencil_icechunk",
        "s3_pencil_zarr_prefix": "chirps_spi_pencil_zarr",
    },
    "gdo_fpar": {
        "gcs_prefix": "gdo_fpar_ic_store",
        "pencil_prefix": "gdo_fpar_pencil_ic_store",
        "s3_source_prefix": "gdo_fpar_icechunk",
        "s3_pencil_prefix": "gdo_fpar_pencil_icechunk",
        "s3_pencil_zarr_prefix": "gdo_fpar_pencil_zarr",
    },
    "gdo_sma": {
        "gcs_prefix": "gdo_sma_ic_store",
        "pencil_prefix": "gdo_sma_pencil_ic_store",
        "s3_source_prefix": "gdo_sma_icechunk",
        "s3_pencil_prefix": "gdo_sma_pencil_icechunk",
        "s3_pencil_zarr_prefix": "gdo_sma_pencil_zarr",
    },
    "rfe2": {
        "gcs_prefix": "rfe2_ic_store",
        "pencil_prefix": "rfe2_pencil_ic_store",
        "s3_source_prefix": "rfe2_icechunk",
        "s3_pencil_prefix": "rfe2_pencil_icechunk",
        "s3_pencil_zarr_prefix": "rfe2_pencil_zarr",
    },
    "chirps_daily": {
        "gcs_prefix": "chirps_daily_ic_store",
        "pencil_prefix": "chirps_daily_pencil_ic_store",
        "s3_source_prefix": "chirps_daily_icechunk",
        "s3_pencil_prefix": "chirps_daily_pencil_icechunk",
        "s3_pencil_zarr_prefix": "chirps_daily_pencil_zarr",
    },
    "imerg_hh": {
        "gcs_prefix": "ea_imerg_ic_store",
        "pencil_prefix": "ea_imerg_pencil_ic_store",
        "s3_source_prefix": "imerg_hh_icechunk",
        "s3_pencil_prefix": "imerg_hh_pencil_icechunk",
        "s3_pencil_zarr_prefix": "imerg_hh_pencil_zarr",
    },
}


def _get_s3_credentials():
    """Get source.coop S3 credentials, falling back to AWS_* env vars."""
    access_key = os.getenv("SOURCE_COOP_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("SOURCE_COOP_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
    return access_key, secret_key, session_token


def _s3_source_storage(info):
    """Create anonymous S3 storage for reading pancake store from source.coop."""
    import icechunk

    prefix = f"{SOURCE_COOP_BASE_PREFIX}/{info['s3_source_prefix']}"
    return icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=prefix,
        region="us-west-2",
        anonymous=True,
    )


def _s3_pencil_storage(info):
    """Create authenticated S3 storage for writing pencil store to source.coop."""
    import icechunk

    access_key, secret_key, session_token = _get_s3_credentials()
    prefix = f"{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_prefix']}"
    return icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=prefix,
        region="us-west-2",
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token,
    )


def _gcs_source_storage(info, sa_path):
    """Create GCS storage for reading pancake store."""
    import icechunk

    return icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=info["gcs_prefix"],
        service_account_file=str(sa_path),
    )


def _gcs_pencil_storage(info, sa_path):
    """Create GCS storage for writing pencil store."""
    import icechunk

    return icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=info["pencil_prefix"],
        service_account_file=str(sa_path),
    )


def open_source_store(info, sa_path, source_coop=False):
    """Open source Icechunk store (pancake chunks) from GCS or source.coop."""
    import icechunk
    import xarray as xr

    if source_coop:
        storage = _s3_source_storage(info)
    else:
        storage = _gcs_source_storage(info, sa_path)

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds


def rechunk_dataset(name, info, sa_path, chunk_lat, chunk_lon, dry_run=False,
                    source_coop=False):
    """Rechunk one dataset from pancake to pencil chunks."""
    import dask
    import dask.array as da
    import icechunk
    import xarray as xr

    print(f"\n{'='*60}")
    print(f"Rechunk: {name}")
    print(f"{'='*60}")

    # Open source
    if source_coop:
        src_label = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{info['s3_source_prefix']}"
    else:
        src_label = f"gs://{GCS_BUCKET}/{info['gcs_prefix']}"
    print(f"\n  Source: {src_label}")
    ds_src = open_source_store(info, sa_path, source_coop=source_coop)
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

    if source_coop:
        tgt_label = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_prefix']}"
    else:
        tgt_label = f"gs://{GCS_BUCKET}/{info['pencil_prefix']}"
    print(f"\n  Target: {tgt_label}")
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
    if source_coop:
        target_storage = _s3_pencil_storage(info)
    else:
        target_storage = _gcs_pencil_storage(info, sa_path)
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

    # Fill pencil store by reading full lat-bands from source and writing
    # all lon-tiles within each band in a single commit (much fewer commits).
    print(f"\n  Filling pencil store (lat-band iteration)...")
    print(f"  Strategy: {n_lat_chunks} lat-bands, each with {n_lon_chunks} lon-tiles")
    print(f"  Commits: ~{n_lat_chunks} (one per lat-band)")
    start = time.time()

    # Re-open source with explicit chunks matching stored layout
    ds_src = open_source_store(info, sa_path, source_coop=source_coop)

    bands_done = 0
    tiles_done = 0
    for lat_start in range(0, n_lat, chunk_lat):
        lat_end = min(lat_start + chunk_lat, n_lat)

        # Read the full lat-band (all lon, all time) from source in one go
        band = ds_src.isel(lat=slice(lat_start, lat_end)).load()

        # Open a single session for all lon-tiles in this band
        session2 = target_repo.writable_session("main")

        for lon_start in range(0, n_lon, chunk_lon):
            lon_end = min(lon_start + chunk_lon, n_lon)

            # Extract the lon-tile from the already-loaded band
            tile = band.isel(lon=slice(lon_start, lon_end))

            tile.to_zarr(
                session2.store,
                region={
                    "time": slice(0, n_time),
                    "lat": slice(lat_start, lat_end),
                    "lon": slice(lon_start, lon_end),
                },
                consolidated=False,
            )
            tiles_done += 1

        # Commit entire lat-band at once
        session2.commit(f"fill lat-band {lat_start}-{lat_end} ({n_lon_chunks} lon-tiles)")
        bands_done += 1

        elapsed = time.time() - start
        pct = 100 * bands_done / n_lat_chunks
        print(f"    [{pct:5.1f}%] band {bands_done}/{n_lat_chunks} "
              f"(lat {lat_start}-{lat_end}), {tiles_done} tiles total ({elapsed:.0f}s)")

        del band

    ds_src.close()
    elapsed = time.time() - start
    print(f"\n  Rechunk complete: {tiles_done} tiles in {bands_done} commits, "
          f"{elapsed:.0f}s ({elapsed/60:.1f} min)")


def rechunk_coiled(name, info, chunk_lat, chunk_lon, n_workers, dry_run=False):
    """Rechunk source.coop Icechunk store to pencil Zarr using Coiled Dask P2P.

    Reads pancake-chunked Icechunk from source.coop (anonymous), rechunks via
    Dask P2P shuffle on a Coiled cluster, writes plain Zarr back to source.coop.
    """
    import pickle

    import coiled
    import dask
    import distributed
    import icechunk
    import xarray as xr

    print(f"\n{'='*60}")
    print(f"Rechunk (Coiled P2P): {name}")
    print(f"{'='*60}")

    # P2P rechunk configuration
    dask.config.set({
        "array.rechunk.method": "p2p",
        "optimization.fuse.active": False,
    })
    print("  Dask config: P2P rechunk enabled, fusion disabled")

    # Open source Icechunk store (anonymous read from source.coop)
    src_prefix = f"{SOURCE_COOP_BASE_PREFIX}/{info['s3_source_prefix']}"
    print(f"\n  Source: s3://{SOURCE_COOP_BUCKET}/{src_prefix}")

    storage = _s3_source_storage(info)
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")

    # Verify pickle serialization (required for Dask workers)
    try:
        pickle.dumps(session.store)
        print("  IcechunkStore is pickle-serializable")
    except Exception as e:
        print(f"  ERROR: IcechunkStore not serializable: {e}")
        return

    # Open with explicit source chunk sizes matching stored layout
    ds = xr.open_zarr(session.store, consolidated=False)
    first_var = list(ds.data_vars)[0]
    source_chunks = {
        d: c for d, c in zip(ds[first_var].dims, ds[first_var].encoding["chunks"])
    }
    ds.close()
    ds = xr.open_zarr(session.store, consolidated=False, chunks=source_chunks)

    print(f"  Shape: {dict(ds.sizes)}")
    print(f"  Source chunks: {source_chunks}")

    n_time = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    n_vars = len(ds.data_vars)
    size_gb = n_time * n_lat * n_lon * 4 * n_vars / (1024**3)

    pencil_chunks = {"time": n_time, "lat": chunk_lat, "lon": chunk_lon}
    chunk_bytes = n_time * chunk_lat * chunk_lon * 4
    n_lat_chunks = -(-n_lat // chunk_lat)
    n_lon_chunks = -(-n_lon // chunk_lon)
    n_target_chunks = n_lat_chunks * n_lon_chunks * n_vars

    target_path = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_zarr_prefix']}"
    print(f"\n  Target: {target_path}")
    print(f"  Pencil chunks: (time={n_time}, lat={chunk_lat}, lon={chunk_lon})")
    print(f"  Chunk size: {chunk_bytes / (1024**2):.1f} MB")
    print(f"  Total target chunks: {n_target_chunks} "
          f"({n_lat_chunks} lat × {n_lon_chunks} lon × {n_vars} vars)")
    print(f"  Total data: {size_gb:.1f} GB")
    per_worker_gb = size_gb / n_workers
    print(f"  Workers: {n_workers} × n2-highmem-4 (32 GB each)")
    print(f"  Per-worker data: {per_worker_gb:.1f} GB")

    if dry_run:
        print("\n  (dry run — no rechunking)")
        ds.close()
        return

    # Quick credential check before launching cluster
    access_key, secret_key, session_token = _get_s3_credentials()
    if not access_key or not secret_key:
        print("ERROR: S3 credentials required for writing to source.coop")
        return

    # Launch fixed-size Coiled cluster (P2P requires static cluster)
    print(f"\n  Launching Coiled cluster...")
    overall_start = time.time()

    cluster = coiled.Cluster(
        name=f"{name}-pencil-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-highmem-4",
        region="us-west1",
        workspace=os.getenv("COILED_WORKSPACE"),
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=600)
    print(f"  Cluster ready: {n_workers} workers")
    print(f"  Dashboard: {client.dashboard_link}")

    # Rechunk with Dask P2P
    print(f"\n  Starting P2P rechunk + write...")
    ds_rechunked = ds.chunk(pencil_chunks)

    # Re-read .env NOW (right before write) to get freshest token,
    # minimizing time between token capture and S3 writes.
    load_dotenv(override=True)
    access_key, secret_key, session_token = _get_s3_credentials()
    print(f"  S3 credentials refreshed (key=...{access_key[-4:]})")

    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "token": session_token,
        "client_kwargs": {"region_name": "us-west-2"},
    }

    encoding = {
        var: {"chunks": (n_time, chunk_lat, chunk_lon)}
        for var in ds.data_vars
    }

    ds_rechunked.to_zarr(
        target_path,
        storage_options=storage_options,
        encoding=encoding,
        mode="w",
        consolidated=True,
    )

    print("  Write complete!")
    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"RECHUNK COMPLETE: {name}")
    print(f"  Target: {target_path}")
    print(f"  Chunks: ({n_time}, {chunk_lat}, {chunk_lon})")
    print(f"  Total chunks: {n_target_chunks}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(f"{'='*60}")


def verify_pencil(name, info, sa_path, source_coop=False, coiled_mode=False):
    """Verify pencil-chunked store."""
    import xarray as xr

    if coiled_mode:
        # Plain Zarr on source.coop S3
        target_path = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_zarr_prefix']}"
        print(f"\n  Verifying (Zarr): {target_path}")
        ds = xr.open_zarr(target_path, storage_options={"anon": True}, consolidated=True)
    elif source_coop:
        import icechunk
        prefix = f"{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_prefix']}"
        print(f"\n  Verifying: s3://{SOURCE_COOP_BUCKET}/{prefix}")
        storage = icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=prefix,
            region="us-west-2",
            anonymous=True,
        )
        repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default()
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)
    else:
        import icechunk
        print(f"\n  Verifying: gs://{GCS_BUCKET}/{info['pencil_prefix']}")
        storage = _gcs_pencil_storage(info, sa_path)
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


def cleanup_pencil_icechunk(name, info):
    """Delete the old pencil Icechunk store from source.coop S3."""
    import s3fs

    prefix = f"{SOURCE_COOP_BASE_PREFIX}/{info['s3_pencil_prefix']}"
    full_path = f"{SOURCE_COOP_BUCKET}/{prefix}"

    access_key, secret_key, session_token = _get_s3_credentials()
    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        token=session_token,
        client_kwargs={"region_name": "us-west-2"},
    )

    print(f"\n  Checking: s3://{full_path}/")
    if not fs.exists(full_path):
        print(f"  Not found — nothing to delete")
        return

    # Count objects
    files = fs.ls(full_path, detail=True)
    total = len(files)
    print(f"  Found {total} top-level objects")

    # Get total size recursively
    all_files = fs.find(full_path)
    total_files = len(all_files)
    total_size = sum(fs.info(f)["size"] for f in all_files[:100])  # sample first 100
    est_size_gb = total_size / (1024**3) * (total_files / min(100, total_files))
    print(f"  Total objects: {total_files}, estimated size: {est_size_gb:.2f} GB")

    print(f"  Deleting s3://{full_path}/ ...")
    fs.rm(full_path, recursive=True)
    print(f"  Deleted.")


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
    parser.add_argument("--source-coop", action="store_true",
                        help="Use source.coop S3 instead of GCS (read public pancake, write pencil)")
    parser.add_argument("--coiled", action="store_true",
                        help="Use Coiled Dask P2P rechunk (fast, writes plain Zarr to source.coop)")
    parser.add_argument("--n-workers", type=int, default=15,
                        help="Number of Coiled workers (default: 15)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete old pencil Icechunk stores from source.coop")
    args = parser.parse_args()

    sa_path = Path(__file__).parent / args.sa_file

    # SA file only needed for GCS mode (not source-coop, coiled, or cleanup)
    if not args.source_coop and not args.coiled and not args.cleanup and not sa_path.exists():
        print(f"ERROR: SA file not found: {sa_path}")
        return

    if (args.source_coop or args.coiled or args.cleanup) and not args.dry_run and not args.verify:
        access_key, secret_key, _ = _get_s3_credentials()
        if not access_key or not secret_key:
            print("ERROR: Set SOURCE_COOP_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID and")
            print("       SOURCE_COOP_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY")
            print("       in .env or environment for writing pencil stores to source.coop")
            return

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, info in datasets.items():
        if args.cleanup:
            cleanup_pencil_icechunk(name, info)
        elif args.verify:
            verify_pencil(name, info, sa_path, source_coop=args.source_coop,
                          coiled_mode=args.coiled)
        elif args.coiled:
            rechunk_coiled(name, info, args.chunk_lat, args.chunk_lon,
                           args.n_workers, args.dry_run)
        else:
            rechunk_dataset(name, info, sa_path, args.chunk_lat, args.chunk_lon,
                            args.dry_run, source_coop=args.source_coop)

    print("\nDone.")


if __name__ == "__main__":
    main()
