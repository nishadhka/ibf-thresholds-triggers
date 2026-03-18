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
#     "requests>=2.31.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
CPC RFE2 Africa Rainfall Estimates — Download, subset EA, write to GCS Icechunk
================================================================================

Daily rainfall estimates from CPC/FEWS RFE2 (2001-present).
Batched by month: each commit contains ~28-31 daily files.

  For each month-batch:
    1. Download daily ZIP files from CPC FTP
    2. Extract GeoTIFF, subset to East Africa
    3. Concatenate month of daily arrays
    4. Append to GCS-backed Icechunk store
    5. Delete temp files

Source: https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/geotiff/

Data: daily rainfall estimate (mm), Africa 0.1° resolution, 2001-present.
Variable: band_data -> rfe2 (rainfall estimate)

Usage:
    uv run --python 3.12 rfe2_icechunk.py run \
        --service-account coiled-data-e4drr_202505.json

    uv run --python 3.12 rfe2_icechunk.py ingest \
        --service-account coiled-data-e4drr_202505.json
    uv run --python 3.12 rfe2_icechunk.py verify \
        --service-account coiled-data-e4drr_202505.json
"""

import logging
import os
import tempfile
import time
import zipfile
from collections import defaultdict

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rfe2_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

BASE_URL = "https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/geotiff/"

VAR_NAME = "rfe2"

# East Africa bounding box
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "rfe2_ic_store"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/rfe2_icechunk"

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
    """Scrape the FTP directory listing for all africa_rfe.*.tif.zip files."""
    import re

    logger.info("  Fetching file listing from CPC FTP...")
    r = requests.get(BASE_URL, timeout=120)
    r.raise_for_status()
    pattern = re.compile(r"africa_rfe\.(\d{8})\.tif\.zip")
    matches = sorted(set(pattern.findall(r.text)))
    filenames = [f"africa_rfe.{d}.tif.zip" for d in matches]
    logger.info(f"  Found {len(filenames)} files ({matches[0]} -> {matches[-1]})")
    return filenames


def group_by_month(filenames):
    """Group filenames by YYYY-MM, return ordered list of (month_key, [filenames])."""
    groups = defaultdict(list)
    for f in filenames:
        # africa_rfe.YYYYMMDD.tif.zip
        date_str = f.split(".")[1]
        month_key = date_str[:6]  # YYYYMM
        groups[month_key].append(f)
    return sorted(groups.items())


def download_and_extract_tif(filename):
    """Download a single ZIP, extract TIF, return path to extracted TIF."""
    url = BASE_URL + filename
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            for chunk in r.iter_content(chunk_size=8192 * 16):
                tmp.write(chunk)
            tmp.close()

            # Extract TIF
            tmpdir = tempfile.mkdtemp(prefix="rfe2_")
            with zipfile.ZipFile(tmp.name) as zf:
                tifs = [n for n in zf.namelist() if n.endswith(".tif")]
                zf.extractall(tmpdir, members=tifs)
            os.unlink(tmp.name)

            tif_path = os.path.join(tmpdir, tifs[0])
            return tif_path, tmpdir
        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                logger.warning(f"    Retry {attempt+1}/3 for {filename}: {e}")
                time.sleep(wait)
            else:
                raise


def subset_tif_ea(tif_path, date_str):
    """Read GeoTIFF, subset to EA, return DataArray with time dim."""
    import pandas as pd
    import rioxarray  # noqa: F401
    import xarray as xr

    ds = xr.open_dataset(tif_path, engine="rasterio", decode_times=False)
    time_val = pd.Timestamp(date_str)

    # Handle near-duplicate y values (same issue as fAPAR/SMA files)
    y_vals = ds.y.values
    n_unique_rounded = len(np.unique(np.round(y_vals, 4)))
    if n_unique_rounded < len(y_vals):
        logger.info(f"    Deduplicating y: {len(y_vals)} -> {n_unique_rounded} (taking every 2nd)")
        ds = ds.isel(y=slice(None, None, 2))

    # y is descending (40 -> -40), x ascending (-20 -> 55)
    # Use slightly exclusive upper bounds to avoid boundary rounding issues
    # (some files include 54.0 while others stop at 53.9)
    da_ea = ds["band_data"].sel(
        y=slice(EA_LAT_MAX, EA_LAT_MIN),
        x=slice(EA_LON_MIN, EA_LON_MAX - 0.01),
    ).squeeze("band", drop=True)

    da_ea = da_ea.expand_dims(time=[time_val]).load()
    ds.close()
    return da_ea


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
    """Download daily RFE2 files in monthly batches, subset to EA, write to GCS Icechunk."""
    import icechunk
    import shutil
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INGEST: monthly batches -> download -> subset EA -> GCS icechunk")
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
        temp_dirs = []
        failed_files = 0

        for filename in month_files:
            try:
                date_str = filename.split(".")[1]  # YYYYMMDD
                tif_path, tmpdir = download_and_extract_tif(filename)
                temp_dirs.append(tmpdir)
                da = subset_tif_ea(tif_path, date_str)
                arrays.append(da)
            except Exception as e:
                logger.warning(f"    SKIP {filename}: {e}")
                failed_files += 1

        # Cleanup temp dirs
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)

        if not arrays:
            logger.error(f"    No data for {month_key}, skipping commit")
            total_failed += 1
            continue

        # Concatenate month
        # join='override' prevents dimension doubling from floating-point
        # coordinate differences between files
        combined = xr.concat(arrays, dim="time", join="override").sortby("time")
        combined = combined.rename({"y": "lat", "x": "lon"})

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
                                "long_name": "Rainfall Estimate (RFE2)",
                                "units": "mm",
                                "source": "CPC/FEWS RFE2",
                            },
                        ),
                    },
                    coords={
                        "time": combined.time.values,
                        "lat": ("lat", combined.lat.values.astype(np.float64), {"units": "degrees_north"}),
                        "lon": ("lon", combined.lon.values.astype(np.float64), {"units": "degrees_east"}),
                    },
                    attrs={
                        "title": "CPC RFE2 Daily Rainfall Estimates - East Africa Subset",
                        "source": "NOAA CPC / FEWS NET RFE2",
                        "source_url": BASE_URL,
                        "region": "East Africa",
                        "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
                        "temporal_resolution": "daily",
                        "spatial_resolution": "0.1 degrees (~10 km)",
                        "variable": "rfe2 - Rainfall Estimate version 2 (mm/day)",
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
            total_failed += 1

    elapsed = time.time() - overall_start
    logger.info("\n" + "=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info(f"  Months ingested: {total_ingested}, Skipped: {total_skipped}, Failed: {total_failed}")
    logger.info(f"  Total months: {len(monthly_batches)}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# verify
# ──────────────────────────────────────────────────────────────


def cmd_verify(args):
    """Inspect the GCS Icechunk store."""
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CPC RFE2 Daily Rainfall EA -> GCS Icechunk store",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Full pipeline: ingest + verify")
    add_storage_args(p_run)

    p_ing = sub.add_parser("ingest", help="Monthly batches: download -> subset -> icechunk -> cleanup")
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
