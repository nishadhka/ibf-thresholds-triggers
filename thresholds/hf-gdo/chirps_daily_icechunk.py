#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "icechunk>=0.1.0",
#     "rioxarray>=0.15.0",
#     "rasterio>=1.3.0",
#     "s3fs>=2024.1.0",
#     "python-dotenv>=1.0.0",
#     "coiled>=1.0.0",
#     "distributed>=2024.1.0",
# ]
# ///
"""
CHIRPS v2.0 Daily Rainfall — Download from DE Africa S3, subset EA, write to Icechunk
======================================================================================

Daily rainfall from CHIRPS v2.0 (1981-present), 0.05° resolution.
Source: s3://deafrica-input-datasets/rainfall_chirps_daily/<year>/<month>/

Batched by month: each commit contains ~28-31 daily GeoTIFF files.

  For each month-batch:
    1. Read daily GeoTIFF from DE Africa public S3
    2. Subset to East Africa bounding box
    3. Concatenate month of daily arrays
    4. Append to Icechunk store (source.coop or GCS)
    5. Commit with month label

East Africa bbox matches other stores: lat [-14.5, 25.5], lon [19.5, 54.0]
Resolution: 0.05° (~5 km), matching chirps_spi and gdo_sma grids.

Usage:
    # Ingest all available months to source.coop (sequential, slow ~12h)
    uv run --python 3.12 chirps_daily_icechunk.py ingest --source-coop

    # Parallel with Coiled cluster (recommended, ~20-40 min for full backfill)
    uv run --python 3.12 chirps_daily_icechunk.py ingest --source-coop \
        --coiled --n-workers 4

    # Parallel with local threads (no cluster, ~1.5h from non-AWS host)
    uv run --python 3.12 chirps_daily_icechunk.py ingest --source-coop --threads 16

    # Verify store contents
    uv run --python 3.12 chirps_daily_icechunk.py verify --source-coop

    # Dry run: list available months without ingesting
    uv run --python 3.12 chirps_daily_icechunk.py ingest --source-coop --dry-run
"""

import logging
import os
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chirps_daily_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

S3_BUCKET = "deafrica-input-datasets"
S3_PREFIX = "rainfall_chirps_daily"

VAR_NAME = "precip"

# East Africa bounding box (same as rfe2, chirps_spi, gdo_sma, etc.)
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "chirps_daily_ic_store"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/chirps_daily_icechunk"

# Chunking: ~1 month of daily data per time chunk
ZARR_CHUNK_TIME = 31


def get_storage(args):
    """Create GCS, local, or source.coop S3 Icechunk storage."""
    import icechunk

    if args.local:
        logger.info(f"  Storage: local ({args.local})")
        return icechunk.local_filesystem_storage(path=args.local)

    if args.source_coop:
        logger.info(f"  Storage: s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_PREFIX}")
        return icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=SOURCE_COOP_PREFIX,
            region="us-west-2",
            access_key_id=os.getenv("SOURCE_COOP_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY"),
            session_token=os.getenv("SOURCE_COOP_SESSION_TOKEN"),
        )

    bucket = args.gcs_bucket
    prefix = args.gcs_prefix
    logger.info(f"  Storage: gs://{bucket}/{prefix}")
    return icechunk.gcs_storage(
        bucket=bucket,
        prefix=prefix,
        service_account_file=args.service_account,
    )


def list_remote_files():
    """List all CHIRPS daily GeoTIFF files on DE Africa S3."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    logger.info("  Scanning DE Africa S3 for CHIRPS daily files...")

    base = f"{S3_BUCKET}/{S3_PREFIX}"
    years = sorted(fs.ls(base))
    year_names = [y.split("/")[-1] for y in years if y.split("/")[-1].isdigit()]
    logger.info(f"  Years available: {year_names[0]} -> {year_names[-1]} ({len(year_names)} years)")

    all_files = []
    for year in year_names:
        year_path = f"{base}/{year}"
        months = sorted(fs.ls(year_path))
        for month_path in months:
            files = sorted(fs.ls(month_path))
            tifs = [f for f in files if f.endswith(".tif")]
            all_files.extend(tifs)

    logger.info(f"  Found {len(all_files)} daily TIF files")
    if all_files:
        first = all_files[0].split("/")[-1]
        last = all_files[-1].split("/")[-1]
        logger.info(f"  Range: {first} -> {last}")
    return all_files


def group_by_month(filepaths):
    """Group S3 file paths by YYYY-MM, return ordered list of (month_key, [paths])."""
    import re

    groups = defaultdict(list)
    pattern = re.compile(r"chirps-v2\.0_(\d{4})\.(\d{2})\.(\d{2})\.tif")
    for fp in filepaths:
        fname = fp.split("/")[-1]
        m = pattern.match(fname)
        if m:
            month_key = f"{m.group(1)}{m.group(2)}"  # YYYYMM
            groups[month_key].append(fp)
    return sorted(groups.items())


def fetch_and_subset(s3_path):
    """Worker-safe function: read TIF from S3, subset to EA, return (time, lat, lon, arr).

    Returns plain numpy + datetime so it serializes cheaply across Dask workers.
    Each worker creates its own s3fs filesystem.
    """
    import re

    import numpy as np
    import pandas as pd
    import rasterio
    import s3fs
    from rasterio.io import MemoryFile
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform_fn

    fs = s3fs.S3FileSystem(anon=True)

    fname = s3_path.split("/")[-1]
    m = re.match(r"chirps-v2\.0_(\d{4})\.(\d{2})\.(\d{2})\.tif", fname)
    date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    time_val = pd.Timestamp(date_str)

    data = fs.cat(s3_path)
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            transform = src.transform
            res_x = transform.a
            res_y = transform.e  # negative (descending)

            col_min = max(0, int((EA_LON_MIN - transform.c) / res_x))
            col_max = min(src.width, int((EA_LON_MAX - transform.c) / res_x))
            row_min = max(0, int((EA_LAT_MAX - transform.f) / res_y))
            row_max = min(src.height, int((EA_LAT_MIN - transform.f) / res_y))

            window = Window.from_slices((row_min, row_max), (col_min, col_max))
            arr = src.read(1, window=window).astype(np.float32)

            wt = win_transform_fn(window, transform)
            n_rows, n_cols = arr.shape
            lons = np.array(
                [wt.c + (j + 0.5) * wt.a for j in range(n_cols)], dtype=np.float64
            )
            lats = np.array(
                [wt.f + (i + 0.5) * wt.e for i in range(n_rows)], dtype=np.float64
            )

    arr[arr == -9999.0] = np.nan
    arr[arr < 0] = np.nan

    return time_val, lats, lons, arr


def results_to_arrays(results):
    """Convert worker results [(time, lat, lon, arr), ...] into list of DataArrays."""
    import xarray as xr

    arrays = []
    for time_val, lats, lons, arr in results:
        da = xr.DataArray(
            arr[np.newaxis, :, :],
            dims=("time", "lat", "lon"),
            coords={"time": [time_val], "lat": lats, "lon": lons},
        )
        arrays.append(da)
    return arrays


def get_committed_months(repo):
    """Return set of month keys already committed to icechunk store."""
    committed = set()
    try:
        for commit in repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("ingest "):
                committed.add(msg.replace("ingest ", ""))
    except Exception:
        pass
    return committed


# ──────────────────────────────────────────────────────────────
# ingest: monthly batches -> icechunk
# ──────────────────────────────────────────────────────────────


def cmd_ingest(args):
    """Download daily CHIRPS files in monthly batches, subset to EA, write to Icechunk."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INGEST: CHIRPS daily from DE Africa S3 -> subset EA -> Icechunk")
    logger.info("=" * 60)
    overall_start = time.time()

    # Get full file listing and group by month
    all_files = list_remote_files()
    monthly_batches = group_by_month(all_files)
    logger.info(f"  {len(monthly_batches)} monthly batches, {len(all_files)} total files")

    storage = get_storage(args)
    config = icechunk.RepositoryConfig.default()

    is_new_store = True
    repo = None
    committed = set()
    try:
        repo = icechunk.Repository.open(storage, config=config)
        is_new_store = False
        committed = get_committed_months(repo)
        logger.info(f"  Existing store: {len(committed)} months already ingested")
    except Exception:
        logger.info("  Will create new store with first batch")

    if args.dry_run:
        new_months = [mk for mk, _ in monthly_batches if mk not in committed]
        logger.info(f"\n  DRY RUN: {len(new_months)} months to ingest, {len(committed)} already done")
        if new_months:
            logger.info(f"  First new: {new_months[0]}, Last new: {new_months[-1]}")
        return

    # ── Set up parallelism: Coiled cluster, local threads, or sequential ──
    client = None
    cluster = None
    executor = None

    if args.coiled:
        import coiled
        import distributed

        logger.info(f"\n  Launching Coiled cluster: {args.n_workers} workers in us-west1")
        cluster = coiled.Cluster(
            name=f"chirps-daily-{int(time.time()) % 10000}",
            n_workers=args.n_workers,
            worker_vm_types="n2-standard-2",
            region="us-west1",
            workspace=os.getenv("COILED_WORKSPACE"),
            idle_timeout="20 minutes",
        )
        client = distributed.Client(cluster)
        client.wait_for_workers(n_workers=args.n_workers, timeout=600)
        logger.info(f"  Cluster ready: {args.n_workers} workers")
        logger.info(f"  Dashboard: {client.dashboard_link}")
    elif args.threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.threads)
        logger.info(f"\n  Local thread pool: {args.threads} threads")

    total_ingested = 0
    total_skipped = 0
    total_failed = 0

    for i, (month_key, month_files) in enumerate(monthly_batches):
        label = f"[{i+1}/{len(monthly_batches)}]"

        if month_key in committed:
            logger.info(f"  {label} {month_key} ({len(month_files)} files) -- already ingested, skipping")
            total_skipped += 1
            continue

        logger.info(f"\n  {label} {month_key} ({len(month_files)} files)")
        batch_start = time.time()
        arrays = []
        failed_files = 0

        # Fetch all files in this month batch in parallel
        try:
            if client is not None:
                # Coiled: dispatch to cluster, gather results
                futures = client.map(fetch_and_subset, month_files)
                results = client.gather(futures)
                arrays = results_to_arrays(results)
            elif executor is not None:
                # Local threads
                results = list(executor.map(fetch_and_subset, month_files))
                arrays = results_to_arrays(results)
            else:
                # Sequential
                for s3_path in month_files:
                    try:
                        results = [fetch_and_subset(s3_path)]
                        arrays.extend(results_to_arrays(results))
                    except Exception as e:
                        fname = s3_path.split("/")[-1]
                        logger.warning(f"    SKIP {fname}: {e}")
                        failed_files += 1
        except Exception as e:
            logger.error(f"    Batch fetch failed for {month_key}: {e}")
            total_failed += 1
            continue

        if not arrays:
            logger.error(f"    No data for {month_key}, skipping commit")
            total_failed += 1
            continue

        # Concatenate month
        combined = xr.concat(arrays, dim="time", join="override").sortby("time")

        try:
            if is_new_store:
                n_lat = combined.sizes["lat"]
                n_lon = combined.sizes["lon"]
                chunk_time = min(ZARR_CHUNK_TIME, combined.sizes["time"])

                ds_out = xr.Dataset(
                    {
                        VAR_NAME: (
                            ("time", "lat", "lon"),
                            combined.values,
                            {
                                "long_name": "CHIRPS v2.0 Daily Precipitation",
                                "units": "mm/day",
                                "source": "CHIRPS v2.0 (Climate Hazards Group)",
                            },
                        ),
                    },
                    coords={
                        "time": combined.time.values,
                        "lat": ("lat", combined.lat.values.astype(np.float64), {"units": "degrees_north"}),
                        "lon": ("lon", combined.lon.values.astype(np.float64), {"units": "degrees_east"}),
                    },
                    attrs={
                        "title": "CHIRPS v2.0 Daily Precipitation - East Africa Subset",
                        "source": "Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS)",
                        "source_url": f"s3://{S3_BUCKET}/{S3_PREFIX}/",
                        "region": "East Africa",
                        "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
                        "temporal_resolution": "daily",
                        "spatial_resolution": "0.05 degrees (~5 km)",
                        "variable": "precip - daily precipitation (mm/day)",
                    },
                )

                storage = get_storage(args)
                repo = icechunk.Repository.create(storage, config=config)
                session = repo.writable_session("main")
                ds_out.to_zarr(
                    session.store,
                    mode="w",
                    encoding={
                        VAR_NAME: {"chunks": (chunk_time, n_lat, n_lon)},
                    },
                    consolidated=False,
                )
                session.commit(f"ingest {month_key}")
                is_new_store = False
                elapsed_batch = time.time() - batch_start
                logger.info(f"    Store created, {len(arrays)} days committed ({elapsed_batch:.0f}s)")

            else:
                session = repo.writable_session("main")
                ds_existing = xr.open_zarr(session.store, consolidated=False)

                ds_append = xr.Dataset({
                    VAR_NAME: (("time", "lat", "lon"), combined.values),
                }, coords={
                    "time": combined.time.values,
                    "lat": ds_existing.lat.values,
                    "lon": ds_existing.lon.values,
                })

                ds_append.to_zarr(
                    session.store,
                    append_dim="time",
                    consolidated=False,
                )
                session.commit(f"ingest {month_key}")
                elapsed_batch = time.time() - batch_start
                logger.info(f"    Appended {len(arrays)} days ({elapsed_batch:.0f}s)")

            total_ingested += 1
            if failed_files:
                logger.warning(f"    {failed_files} files failed in this batch")

        except Exception as e:
            logger.error(f"    COMMIT FAILED for {month_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            total_failed += 1

    elapsed = time.time() - overall_start
    logger.info("\n" + "=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info(f"  Months ingested: {total_ingested}, Skipped: {total_skipped}, Failed: {total_failed}")
    logger.info(f"  Total months: {len(monthly_batches)}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)

    # Tear down parallelism
    if executor is not None:
        executor.shutdown(wait=False)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    if cluster is not None:
        try:
            cluster.close()
            logger.info("  Coiled cluster shut down")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# verify
# ──────────────────────────────────────────────────────────────


def cmd_verify(args):
    """Inspect the Icechunk store."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY")
    logger.info("=" * 60)

    storage = get_storage(args)
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")
    logger.info(f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]} ({ds.sizes['time']} steps)")
    logger.info(f"  Lat: {float(ds.lat.min()):.2f} -> {float(ds.lat.max()):.2f}")
    logger.info(f"  Lon: {float(ds.lon.min()):.2f} -> {float(ds.lon.max()):.2f}")

    if VAR_NAME in ds.data_vars:
        logger.info(f"\nSpot-check: loading first 10 timesteps...")
        sample = ds[VAR_NAME].isel(time=slice(0, 10)).load()
        valid = sample.values[~np.isnan(sample.values)]
        logger.info(f"  Valid: {len(valid)}/{sample.values.size}")
        if len(valid) > 0:
            logger.info(f"  Min={valid.min():.4f}, Max={valid.max():.4f}, Mean={valid.mean():.4f}")

    commits = list(repo.ancestry(branch="main"))
    logger.info(f"\nCommits ({len(commits)}):")
    for c in commits[:15]:
        logger.info(f"  {c.message}")
    if len(commits) > 15:
        logger.info(f"  ... and {len(commits) - 15} more")


# ──────────────────────────────────────────────────────────────
# run: full pipeline
# ──────────────────────────────────────────────────────────────


def cmd_run(args):
    """Full pipeline: ingest all files then verify."""
    cmd_ingest(args)
    logger.info("")
    cmd_verify(args)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def add_storage_args(parser):
    """Add common GCS/local storage arguments."""
    parser.add_argument("--service-account", type=str, default="coiled-data-e4drr_202505.json",
                        help="GCS service account JSON file")
    parser.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    parser.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    parser.add_argument("--local", type=str, default=None,
                        help="Use local storage instead of GCS")
    parser.add_argument("--source-coop", action="store_true",
                        help="Write to source.coop S3 instead of GCS")
    parser.add_argument("--dry-run", action="store_true",
                        help="List available months without ingesting")
    parser.add_argument("--coiled", action="store_true",
                        help="Use Coiled cluster for parallel file reads (us-west1)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of Coiled workers (default: 4)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Local thread pool size for parallel reads (default: 1, sequential). "
                             "Ignored if --coiled is used.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CHIRPS v2.0 Daily Precipitation EA -> Icechunk store",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Full pipeline: ingest + verify")
    add_storage_args(p_run)

    p_ing = sub.add_parser("ingest", help="Monthly batches: S3 read -> subset -> icechunk")
    add_storage_args(p_ing)

    p_vf = sub.add_parser("verify", help="Inspect Icechunk store")
    add_storage_args(p_vf)

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
