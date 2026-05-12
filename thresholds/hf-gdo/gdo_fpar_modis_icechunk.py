#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "icechunk>=0.1.0",
#     "netCDF4>=1.6.2",
#     "requests>=2.31.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
GDO fAPAR Anomalies (MODIS) — Download, subset EA, write to GCS Icechunk store
================================================================================

Processes each NC file through all stages before moving to the next:
  For each file:
    1. Download NC from Copernicus GDO to temp file
    2. Subset to East Africa
    3. Write/append to GCS-backed Icechunk store
    4. Delete temp NC file

Source: https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/
        GDO_Fraction_of_Absorbed_Photosynthetically_Active_Radiation_Anomalies_fAPAR_MODIS/ver1-3-1/

Usage:
    # Full pipeline: ingest all files + verify
    uv run --python 3.12 gdo_fpar_modis_icechunk.py run \
        --service-account coiled-data-e4drr_202505.json

    # Or to source.coop:
    uv run --python 3.12 gdo_fpar_modis_icechunk.py run --source-coop
"""

import logging
import os
import tempfile
import time

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gdo_fpar_modis_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

BASE_URL = (
    "https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/"
    "GDO_Fraction_of_Absorbed_Photosynthetically_Active_Radiation_Anomalies_fAPAR_MODIS/"
    "ver1-3-1/"
)

VAR_NAME = "fapan"

# East Africa bounding box
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "gdo_fpar_modis_ic_store"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/gdo_fpar_modis_icechunk"

# Chunking: ~1 year of dekadal data per chunk
ZARR_CHUNK_TIME = 36

# MODIS fAPAR anomalies: 2001 through 2022 (22 files).
# Last file is truncated at 20221111 (final dekad of the MODIS series).
NC_FILES = [
    "fapan_m_wld_20010101_20011221_t.nc",
    "fapan_m_wld_20020101_20021221_t.nc",
    "fapan_m_wld_20030101_20031221_t.nc",
    "fapan_m_wld_20040101_20041221_t.nc",
    "fapan_m_wld_20050101_20051221_t.nc",
    "fapan_m_wld_20060101_20061221_t.nc",
    "fapan_m_wld_20070101_20071221_t.nc",
    "fapan_m_wld_20080101_20081221_t.nc",
    "fapan_m_wld_20090101_20091221_t.nc",
    "fapan_m_wld_20100101_20101221_t.nc",
    "fapan_m_wld_20110101_20111221_t.nc",
    "fapan_m_wld_20120101_20121221_t.nc",
    "fapan_m_wld_20130101_20131221_t.nc",
    "fapan_m_wld_20140101_20141221_t.nc",
    "fapan_m_wld_20150101_20151221_t.nc",
    "fapan_m_wld_20160101_20161221_t.nc",
    "fapan_m_wld_20170101_20171221_t.nc",
    "fapan_m_wld_20180101_20181221_t.nc",
    "fapan_m_wld_20190101_20191221_t.nc",
    "fapan_m_wld_20200101_20201221_t.nc",
    "fapan_m_wld_20210101_20211221_t.nc",
    "fapan_m_wld_20220101_20221111_t.nc",
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


def download_nc(filename):
    """Download a single NC file, return path to temp file."""
    url = BASE_URL + filename
    logger.info(f"  Downloading {filename}...")

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=600, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
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


def subset_ea(nc_path):
    """Open NC file, subset to East Africa, return loaded DataArray."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Some files have near-duplicate lat values (consecutive pairs differ by
    # ~1e-14). Detect via rounded uniqueness and fix by taking every other lat.
    lat_vals = ds.lat.values
    n_unique_rounded = len(np.unique(np.round(lat_vals, 4)))
    if n_unique_rounded < len(lat_vals):
        logger.info(f"    Deduplicating lat: {len(lat_vals)} -> {n_unique_rounded} (taking every 2nd)")
        ds = ds.isel(lat=slice(None, None, 2))

    # Some files have descending lat (90 -> -56), others ascending (-56 -> 90)
    lat_ascending = ds.lat.values[0] < ds.lat.values[-1]
    if lat_ascending:
        da_ea = ds[VAR_NAME].sel(
            lat=slice(EA_LAT_MIN, EA_LAT_MAX),
            lon=slice(EA_LON_MIN, EA_LON_MAX),
        )
    else:
        da_ea = ds[VAR_NAME].sel(
            lat=slice(EA_LAT_MAX, EA_LAT_MIN),
            lon=slice(EA_LON_MIN, EA_LON_MAX),
        )
    # Ensure consistent descending lat order across all files
    if lat_ascending:
        da_ea = da_ea.isel(lat=slice(None, None, -1))
    if "band" in da_ea.dims:
        da_ea = da_ea.squeeze("band", drop=True)
    da_ea = da_ea.load()
    ds.close()

    logger.info(
        f"    EA subset: {da_ea.sizes['time']} time x "
        f"{da_ea.sizes['lat']} lat x {da_ea.sizes['lon']} lon"
    )
    return da_ea


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
# ingest: per-file download -> subset -> icechunk -> cleanup
# ──────────────────────────────────────────────────────────────


def cmd_ingest(args):
    """For each NC file: download, subset to EA, write to GCS Icechunk, delete temp NC."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INGEST: per-file download -> subset -> GCS icechunk -> cleanup")
    logger.info("=" * 60)
    overall_start = time.time()

    storage = get_storage(args)
    config = icechunk.RepositoryConfig.default()

    # Try to open existing store or mark as new
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

    for i, filename in enumerate(NC_FILES):
        label = f"[{i+1}/{len(NC_FILES)}]"

        if filename in committed:
            logger.info(f"  {label} {filename} -- already ingested, skipping")
            total_skipped += 1
            continue

        logger.info(f"\n  {label} {filename}")
        nc_path = None
        try:
            # 1. Download
            nc_path = download_nc(filename)

            # 2. Subset to EA
            da_ea = subset_ea(nc_path)

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
                                "long_name": "fAPAR anomaly (MODIS)",
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
                        "title": "GDO fAPAR Anomalies (MODIS) - East Africa Subset",
                        "source": "Copernicus European Drought Observatory (GDO)",
                        "source_url": BASE_URL,
                        "region": "East Africa",
                        "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
                        "temporal_resolution": "dekadal (~10 days)",
                        "spatial_resolution": "0.083 degrees (~8.3 km)",
                        "variable": "fapan - fAPAR anomaly from MODIS",
                    },
                )

                # Need fresh storage handle for create (GCS storage is consumed)
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
                # Subsequent files: append along time
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
            # 4. Cleanup temp NC file
            if nc_path and os.path.exists(nc_path):
                os.unlink(nc_path)
                logger.info(f"    Cleaned up temp NC file")

    elapsed = time.time() - overall_start
    logger.info("\n" + "=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info(f"  Ingested: {total_ingested}, Skipped: {total_skipped}, Total: {len(NC_FILES)}")
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
    for c in commits:
        logger.info(f"  {c.message}")


# ──────────────────────────────────────────────────────────────
# run: full pipeline (ingest + verify)
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
        description="GDO fAPAR Anomalies (MODIS) EA -> GCS Icechunk store",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Full pipeline: ingest + verify")
    add_storage_args(p_run)

    p_ing = sub.add_parser("ingest", help="Per-file: download -> subset -> icechunk -> cleanup")
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
