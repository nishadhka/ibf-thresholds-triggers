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
GDO Soil Moisture Index Anomaly — Download, subset EA, write to GCS Icechunk
=============================================================================

Processes each ZIP file through all stages before moving to the next:
  For each ZIP (annual):
    1. Download ZIP from Copernicus GDO
    2. Extract GeoTIFF files (one per dekad)
    3. Subset each TIF to East Africa
    4. Append to GCS-backed Icechunk store
    5. Delete temp files

Source: https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/
        GDO_Soil_Moisture_Index_Anomaly/ver3-0-1/

Data: dekadal (~10-day) Soil Moisture Anomaly, global 0.05° resolution, 1995-present.
Variable: smang (Soil Moisture Index Anomaly)

Usage:
    uv run --python 3.12 gdo_sma_icechunk.py run \
        --service-account coiled-data-e4drr_202505.json

    uv run --python 3.12 gdo_sma_icechunk.py ingest \
        --service-account coiled-data-e4drr_202505.json
    uv run --python 3.12 gdo_sma_icechunk.py verify \
        --service-account coiled-data-e4drr_202505.json
"""

import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gdo_sma_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

BASE_URL = (
    "https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/"
    "GDO_Soil_Moisture_Index_Anomaly/ver3-0-1/"
)

VAR_NAME = "smang"

# East Africa bounding box
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "gdo_sma_ic_store"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/gdo_sma_icechunk"

# Chunking: ~1 year of dekadal data per chunk
ZARR_CHUNK_TIME = 36

ZIP_FILES = [
    "smang_m_gdo_19950101_19951221_t.zip",
    "smang_m_gdo_19960101_19961221_t.zip",
    "smang_m_gdo_19970101_19971221_t.zip",
    "smang_m_gdo_19980101_19981221_t.zip",
    "smang_m_gdo_19990101_19991221_t.zip",
    "smang_m_gdo_20000101_20001221_t.zip",
    "smang_m_gdo_20010101_20011221_t.zip",
    "smang_m_gdo_20020101_20021221_t.zip",
    "smang_m_gdo_20030101_20031221_t.zip",
    "smang_m_gdo_20040101_20041221_t.zip",
    "smang_m_gdo_20050101_20051221_t.zip",
    "smang_m_gdo_20060101_20061221_t.zip",
    "smang_m_gdo_20070101_20071221_t.zip",
    "smang_m_gdo_20080101_20081221_t.zip",
    "smang_m_gdo_20090101_20091221_t.zip",
    "smang_m_gdo_20100101_20101221_t.zip",
    "smang_m_gdo_20110101_20111221_t.zip",
    "smang_m_gdo_20120101_20121221_t.zip",
    "smang_m_gdo_20130101_20131221_t.zip",
    "smang_m_gdo_20140101_20141221_t.zip",
    "smang_m_gdo_20150101_20151221_t.zip",
    "smang_m_gdo_20160101_20161221_t.zip",
    "smang_m_gdo_20170101_20171221_t.zip",
    "smang_m_gdo_20180101_20181221_t.zip",
    "smang_m_gdo_20190101_20191221_t.zip",
    "smang_m_gdo_20200101_20201221_t.zip",
    "smang_m_gdo_20210101_20211221_t.zip",
    "smang_m_gdo_20220101_20221221_t.zip",
    "smang_m_gdo_20230101_20231221_t.zip",
    "smang_m_gdo_20240101_20241221_t.zip",
    "smang_m_gdo_20250101_20251221_t.zip",
    "smang_m_gdo_20260101_20260321_t.zip",
]


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


def download_zip(filename):
    """Download a single ZIP file, return path to temp file."""
    url = BASE_URL + filename
    logger.info(f"  Downloading {filename}...")

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=600, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            for chunk in r.iter_content(chunk_size=8192 * 16):
                tmp.write(chunk)
            tmp.close()
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            logger.info(f"    Downloaded {size_mb:.1f} MB")
            return tmp.name
        except Exception as e:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                logger.warning(f"    Retry {attempt+1}/3 after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def extract_and_subset_zip(zip_path):
    """Extract TIFs from ZIP, subset each to EA, return concatenated DataArray."""
    import pandas as pd
    import rioxarray  # noqa: F401 — registers rasterio engine
    import xarray as xr

    tmpdir = tempfile.mkdtemp(prefix="sma_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            tifs = sorted([n for n in zf.namelist() if n.endswith(".tif")])
            logger.info(f"    {len(tifs)} TIF files in ZIP")
            zf.extractall(tmpdir, members=tifs)

        arrays = []
        for tif_name in tifs:
            tif_path = os.path.join(tmpdir, tif_name)
            # decode_times=False to avoid cftime issues with older dates
            ds = xr.open_dataset(tif_path, engine="rasterio", decode_times=False)

            # Parse time from filename: smang_m_gdo_YYYYMMDD_t_300_z01.tif
            date_str = tif_name.split("_")[3]  # YYYYMMDD
            time_val = pd.Timestamp(date_str)

            # TIF coords are (y, x) — y is descending (lat), x is ascending (lon)
            y_vals = ds.y.values
            y_descending = y_vals[0] > y_vals[-1]

            if y_descending:
                da_ea = ds[VAR_NAME].sel(
                    y=slice(EA_LAT_MAX, EA_LAT_MIN),
                    x=slice(EA_LON_MIN, EA_LON_MAX),
                )
            else:
                da_ea = ds[VAR_NAME].sel(
                    y=slice(EA_LAT_MIN, EA_LAT_MAX),
                    x=slice(EA_LON_MIN, EA_LON_MAX),
                )

            # Ensure consistent descending lat order
            if not y_descending:
                da_ea = da_ea.isel(y=slice(None, None, -1))

            # Handle near-duplicate y values (same issue as fAPAR)
            y_unique_count = len(np.unique(np.round(da_ea.y.values, 4)))
            if y_unique_count < len(da_ea.y):
                da_ea = da_ea.isel(y=slice(None, None, 2))

            # Replace decoded time with parsed timestamp
            da_ea = da_ea.squeeze("time", drop=True).expand_dims(
                time=[time_val]
            ).load()
            ds.close()
            arrays.append(da_ea)

        # Concatenate all timesteps
        combined = xr.concat(arrays, dim="time")
        combined = combined.sortby("time")

        # Rename y/x to lat/lon for consistency
        combined = combined.rename({"y": "lat", "x": "lon"})

        logger.info(
            f"    EA subset: {combined.sizes['time']} time x "
            f"{combined.sizes['lat']} lat x {combined.sizes['lon']} lon"
        )
        return combined

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_committed_files(repo):
    """Return set of filenames already committed to icechunk store."""
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
# ingest: per-file download -> extract -> subset -> icechunk -> cleanup
# ──────────────────────────────────────────────────────────────


def cmd_ingest(args):
    """For each ZIP: download, extract TIFs, subset to EA, write to GCS Icechunk, cleanup."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INGEST: per-ZIP download -> extract -> subset -> GCS icechunk -> cleanup")
    logger.info("=" * 60)
    overall_start = time.time()

    storage = get_storage(args)
    config = icechunk.RepositoryConfig.default()

    is_new_store = True
    repo = None
    committed = set()
    try:
        repo = icechunk.Repository.open(storage, config=config)
        is_new_store = False
        committed = get_committed_files(repo)
        logger.info(f"  Existing store: {len(committed)} files already ingested")
    except Exception:
        logger.info("  Will create new store with first file")

    total_ingested = 0
    total_skipped = 0

    for i, filename in enumerate(ZIP_FILES):
        label = f"[{i+1}/{len(ZIP_FILES)}]"

        if filename in committed:
            logger.info(f"  {label} {filename} -- already ingested, skipping")
            total_skipped += 1
            continue

        logger.info(f"\n  {label} {filename}")
        zip_path = None
        try:
            # 1. Download ZIP
            zip_path = download_zip(filename)

            # 2. Extract TIFs + subset to EA
            da_ea = extract_and_subset_zip(zip_path)

            # 3. Write to Icechunk
            if is_new_store:
                n_lat = da_ea.sizes["lat"]
                n_lon = da_ea.sizes["lon"]
                chunk_time = min(ZARR_CHUNK_TIME, da_ea.sizes["time"])

                ds_out = xr.Dataset(
                    {
                        VAR_NAME: (
                            ("time", "lat", "lon"),
                            da_ea.values,
                            {
                                "long_name": "Soil Moisture Index Anomaly",
                                "source": "GDO Copernicus Drought Observatory",
                            },
                        ),
                    },
                    coords={
                        "time": da_ea.time.values,
                        "lat": ("lat", da_ea.lat.values.astype(np.float64), {"units": "degrees_north"}),
                        "lon": ("lon", da_ea.lon.values.astype(np.float64), {"units": "degrees_east"}),
                    },
                    attrs={
                        "title": "GDO Soil Moisture Index Anomaly - East Africa Subset",
                        "source": "Copernicus European Drought Observatory (GDO)",
                        "source_url": BASE_URL,
                        "region": "East Africa",
                        "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
                        "temporal_resolution": "dekadal (~10 days)",
                        "spatial_resolution": "0.05 degrees (~5 km)",
                        "variable": "smang - Soil Moisture Index Anomaly",
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
                session.commit(f"ingest {filename}")
                is_new_store = False
                logger.info(f"    Icechunk store created, first file committed")

            else:
                session = repo.writable_session("main")
                ds_existing = xr.open_zarr(session.store, consolidated=False)

                ds_append = xr.Dataset({
                    VAR_NAME: (("time", "lat", "lon"), da_ea.values),
                }, coords={
                    "time": da_ea.time.values,
                    "lat": ds_existing.lat.values,
                    "lon": ds_existing.lon.values,
                })

                ds_append.to_zarr(
                    session.store,
                    append_dim="time",
                    consolidated=False,
                )
                session.commit(f"ingest {filename}")
                logger.info(f"    Appended to Icechunk store")

            total_ingested += 1

        except Exception as e:
            logger.error(f"    FAILED: {e}")
        finally:
            if zip_path and os.path.exists(zip_path):
                os.unlink(zip_path)
                logger.info(f"    Cleaned up temp ZIP file")

    elapsed = time.time() - overall_start
    logger.info("\n" + "=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info(f"  Ingested: {total_ingested}, Skipped: {total_skipped}, Total: {len(ZIP_FILES)}")
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
        description="GDO Soil Moisture Index Anomaly EA -> GCS Icechunk store",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Full pipeline: ingest + verify")
    add_storage_args(p_run)

    p_ing = sub.add_parser("ingest", help="Per-ZIP: download -> extract -> subset -> icechunk -> cleanup")
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
