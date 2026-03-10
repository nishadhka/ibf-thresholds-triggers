#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "icechunk>=0.1.0",
#     "netCDF4>=1.6.0",
#     "requests>=2.31.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
GDO CHIRPS SPI (1,3,6,9,12,24,48) — Multi-variable Icechunk store on GCS
=========================================================================

Downloads all 7 SPI indices from Copernicus GDO, subsets to East Africa,
and writes them as separate variables in a single Icechunk store.

SPI1/SPI3 are dekadal; SPI6-48 are monthly. We resample SPI1/SPI3 to
monthly (day==1 timesteps) so all 7 variables share one time axis.

For each year (1991-2026):
  1. Download 7 NC files (one per SPI index)
  2. Subset each to EA, resample SPI1/SPI3 to monthly
  3. Merge into single Dataset with 7 variables
  4. Append to GCS Icechunk store
  5. Cleanup downloaded files

Store path: gs://cpc_awc/chirps_spi_ic_store

Usage:
    uv run --python 3.12 chirps_spi_icechunk.py run \\
        --service-account coiled-data-e4drr_202505.json
"""

import logging
import os
import re
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
        logging.FileHandler("chirps_spi_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

# SPI indices: (code, url_index, is_dekadal)
SPI_INDICES = [
    ("spc01", "SPI1", True),
    ("spc03", "SPI3", True),
    ("spc06", "SPI6", False),
    ("spc09", "SPI9", False),
    ("spc12", "SPI12", False),
    ("spc24", "SPI24", False),
    ("spc48", "SPI48", False),
]

BASE_URL_TEMPLATE = (
    "https://drought.emergency.copernicus.eu/data/"
    "Drought_Observatories_datasets/"
    "GDO_CHIRPS_Standardized_Precipitation_Index_{spi_name}/ver3-0-0/"
)

# East Africa bounding box
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "chirps_spi_ic_store"

ZARR_CHUNK_TIME = 12  # 1 year of monthly data


def get_storage(args):
    """Create GCS or local Icechunk storage."""
    import icechunk

    if args.local:
        logger.info(f"  Storage: local ({args.local})")
        return icechunk.local_filesystem_storage(path=args.local)

    bucket = args.gcs_bucket
    prefix = args.gcs_prefix
    logger.info(f"  Storage: gs://{bucket}/{prefix}")
    return icechunk.gcs_storage(
        bucket=bucket,
        prefix=prefix,
        service_account_file=args.service_account,
    )


def list_nc_files_for_index(var_code, spi_name):
    """Scrape directory listing for NC files of a given SPI index."""
    base_url = BASE_URL_TEMPLATE.format(spi_name=spi_name)
    r = requests.get(base_url, timeout=120)
    r.raise_for_status()
    pattern = re.compile(rf"({var_code}_m_gdo_\d{{8}}_\d{{8}}_[tm]\.nc)")
    matches = sorted(set(pattern.findall(r.text)))
    return base_url, matches


def get_year_file_map():
    """Build mapping: year -> {var_code: (url, filename)} for all SPI indices.

    Returns dict like:
        {"1991": {"spc01": ("https://...", "spc01_m_gdo_19910101_19911221_t.nc"), ...}, ...}
    """
    year_map = {}

    for var_code, spi_name, _is_dekadal in SPI_INDICES:
        base_url, nc_files = list_nc_files_for_index(var_code, spi_name)
        logger.info(f"  {spi_name} ({var_code}): {len(nc_files)} files")
        for f in nc_files:
            # Extract start year from filename: spcXX_m_gdo_YYYYMMDD_...
            year = f.split("_")[3][:4]
            if year not in year_map:
                year_map[year] = {}
            year_map[year][var_code] = (base_url + f, f)

    return dict(sorted(year_map.items()))


def download_nc(url, filename):
    """Download a single NC file, return temp path."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix=filename.replace(".nc", "_"), delete=False
            )
            size = 0
            for chunk in r.iter_content(chunk_size=8192 * 16):
                tmp.write(chunk)
                size += len(chunk)
            tmp.close()
            return tmp.name, size
        except Exception as e:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                logger.warning(f"    Retry {attempt + 1}/3 for {filename}: {e}")
                time.sleep(wait)
            else:
                raise


def subset_nc_ea(nc_path, var_code, is_dekadal):
    """Open NC file, subset to EA. If dekadal, filter to monthly (day==1)."""
    import pandas as pd
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Handle near-duplicate lat values
    lat_vals = ds.lat.values
    n_unique_rounded = len(np.unique(np.round(lat_vals, 4)))
    if n_unique_rounded < len(lat_vals):
        logger.info(
            f"    Deduplicating lat for {var_code}: "
            f"{len(lat_vals)} -> {n_unique_rounded}"
        )
        ds = ds.isel(lat=slice(None, None, 2))

    # Detect lat ordering
    lat_ascending = ds.lat.values[0] < ds.lat.values[-1]
    if lat_ascending:
        lat_slice = slice(EA_LAT_MIN, EA_LAT_MAX)
    else:
        lat_slice = slice(EA_LAT_MAX, EA_LAT_MIN)

    da = ds[var_code].sel(
        lat=lat_slice, lon=slice(EA_LON_MIN, EA_LON_MAX)
    )

    # Drop band dim if present
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)

    # Ensure lat is descending (north -> south)
    if lat_ascending:
        da = da.isel(lat=slice(None, None, -1))

    # If dekadal, select only monthly timesteps (day == 1)
    if is_dekadal:
        monthly_mask = pd.DatetimeIndex(da.time.values).day == 1
        da = da.isel(time=monthly_mask)

    da = da.load()
    ds.close()
    return da


def get_committed_years(repo):
    """Return set of year keys already committed."""
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
# ingest
# ──────────────────────────────────────────────────────────────


def cmd_ingest(args):
    """Download all SPI indices per year, merge, write to Icechunk."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INGEST: per-year download -> subset EA -> merge 7 SPI -> GCS icechunk")
    logger.info("=" * 60)
    overall_start = time.time()

    logger.info("  Fetching file listings...")
    year_map = get_year_file_map()
    years = sorted(year_map.keys())
    logger.info(f"  {len(years)} years: {years[0]} -> {years[-1]}")

    storage = get_storage(args)
    config = icechunk.RepositoryConfig.default()

    is_new_store = True
    repo = None
    committed = set()
    try:
        repo = icechunk.Repository.open(storage, config=config)
        is_new_store = False
        committed = get_committed_years(repo)
        logger.info(f"  Existing store: {len(committed)} years already ingested")
    except Exception:
        logger.info("  Will create new store with first year")

    total_ingested = 0
    total_skipped = 0

    for i, year in enumerate(years):
        label = f"[{i + 1}/{len(years)}]"

        if year in committed:
            logger.info(f"  {label} {year} -- already ingested, skipping")
            total_skipped += 1
            continue

        logger.info(f"\n  {label} {year}")
        year_start = time.time()
        var_data = {}
        tmp_files = []
        failed_vars = []

        for var_code, spi_name, is_dekadal in SPI_INDICES:
            if var_code not in year_map[year]:
                logger.warning(f"    {var_code}: no file for {year}")
                failed_vars.append(var_code)
                continue

            url, filename = year_map[year][var_code]
            try:
                logger.info(f"    Downloading {filename}...")
                nc_path, size_bytes = download_nc(url, filename)
                tmp_files.append(nc_path)
                logger.info(f"      {size_bytes / 1024**2:.1f} MB")

                da = subset_nc_ea(nc_path, var_code, is_dekadal)
                logger.info(f"      EA subset: {dict(da.sizes)}")
                var_data[var_code] = da

            except Exception as e:
                logger.error(f"    FAILED {var_code}: {e}")
                failed_vars.append(var_code)

        # Cleanup temp files
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

        if not var_data:
            logger.error(f"    No data for {year}, skipping")
            continue

        # Build merged dataset
        # Use time from the first available variable
        ref_var = list(var_data.keys())[0]
        ref_time = var_data[ref_var].time.values
        ref_lat = var_data[ref_var].lat.values
        ref_lon = var_data[ref_var].lon.values

        data_vars = {}
        for var_code, da in var_data.items():
            spi_label = var_code.replace("spc", "spi")  # spc01 -> spi1, spc48 -> spi48
            # Remove leading zeros: spi01 -> spi1
            spi_label = "spi" + str(int(spi_label[3:]))
            data_vars[spi_label] = (("time", "lat", "lon"), da.values)

        ds_year = xr.Dataset(
            data_vars,
            coords={
                "time": ref_time,
                "lat": ("lat", ref_lat, {"units": "degrees_north"}),
                "lon": ("lon", ref_lon, {"units": "degrees_east"}),
            },
        )

        try:
            if is_new_store:
                n_time = ds_year.sizes["time"]
                n_lat = ds_year.sizes["lat"]
                n_lon = ds_year.sizes["lon"]
                chunk_time = min(ZARR_CHUNK_TIME, n_time)

                ds_year.attrs = {
                    "title": "CHIRPS SPI Multi-Index - East Africa Subset",
                    "source": "Copernicus GDO / CHIRPS",
                    "variables": "spi1, spi3, spi6, spi9, spi12, spi24, spi48",
                    "region": "East Africa",
                    "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
                    "temporal_resolution": "monthly",
                    "spatial_resolution": "0.05 degrees (~5 km)",
                }

                encoding = {
                    v: {"chunks": (chunk_time, n_lat, n_lon)}
                    for v in ds_year.data_vars
                }

                storage = get_storage(args)
                repo = icechunk.Repository.create(storage, config=config)
                session = repo.writable_session("main")
                ds_year.to_zarr(
                    session.store,
                    mode="w",
                    encoding=encoding,
                    consolidated=False,
                )
                session.commit(f"ingest {year}")
                is_new_store = False
                elapsed = time.time() - year_start
                logger.info(
                    f"    Store created: {len(var_data)} vars, "
                    f"{ds_year.sizes['time']} months ({elapsed:.0f}s)"
                )

            else:
                session = repo.writable_session("main")
                ds_existing = xr.open_zarr(session.store, consolidated=False)

                # Build append dataset using existing coords for lat/lon
                append_vars = {}
                for vname in ds_year.data_vars:
                    append_vars[vname] = (("time", "lat", "lon"), ds_year[vname].values)

                ds_append = xr.Dataset(
                    append_vars,
                    coords={
                        "time": ds_year.time.values,
                        "lat": ds_existing.lat.values,
                        "lon": ds_existing.lon.values,
                    },
                )

                ds_append.to_zarr(
                    session.store,
                    append_dim="time",
                    consolidated=False,
                )
                session.commit(f"ingest {year}")
                elapsed = time.time() - year_start
                logger.info(
                    f"    Appended: {len(var_data)} vars, "
                    f"{ds_year.sizes['time']} months ({elapsed:.0f}s)"
                )

            total_ingested += 1

        except Exception as e:
            logger.error(f"    COMMIT FAILED for {year}: {e}")

        if failed_vars:
            logger.warning(f"    Missing vars for {year}: {failed_vars}")

    elapsed = time.time() - overall_start
    logger.info("\n" + "=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info(
        f"  Years ingested: {total_ingested}, "
        f"Skipped: {total_skipped}, Total: {len(years)}"
    )
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


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
    logger.info(
        f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]} "
        f"({ds.sizes['time']} steps)"
    )
    logger.info(f"  Lat: {float(ds.lat.min()):.2f} -> {float(ds.lat.max()):.2f}")
    logger.info(f"  Lon: {float(ds.lon.min()):.2f} -> {float(ds.lon.max()):.2f}")

    logger.info(f"\nVariables: {list(ds.data_vars)}")
    for vname in sorted(ds.data_vars):
        logger.info(f"\n  Spot-check {vname}: loading first 5 timesteps...")
        sample = ds[vname].isel(time=slice(0, 5)).load()
        valid = sample.values[~np.isnan(sample.values)]
        total = sample.values.size
        logger.info(f"    Valid: {len(valid)}/{total}")
        if len(valid) > 0:
            logger.info(
                f"    Min={valid.min():.4f}, Max={valid.max():.4f}, "
                f"Mean={valid.mean():.4f}"
            )

    commits = list(repo.ancestry(branch="main"))
    logger.info(f"\nCommits ({len(commits)}):")
    for c in commits[:10]:
        logger.info(f"  {c.message}")
    if len(commits) > 10:
        logger.info(f"  ... and {len(commits) - 10} more")


# ──────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────


def cmd_run(args):
    """Full pipeline: ingest then verify."""
    cmd_ingest(args)
    logger.info("")
    cmd_verify(args)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def add_storage_args(parser):
    parser.add_argument(
        "--service-account",
        type=str,
        default="coiled-data-e4drr_202505.json",
    )
    parser.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    parser.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    parser.add_argument("--local", type=str, default=None)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CHIRPS SPI Multi-Index EA -> GCS Icechunk store"
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Full pipeline: ingest + verify")
    add_storage_args(p_run)

    p_ing = sub.add_parser("ingest", help="Download + subset + icechunk")
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
