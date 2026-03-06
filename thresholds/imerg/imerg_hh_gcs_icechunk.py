#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "earthaccess>=0.12.0",
#     "netCDF4>=1.6.2",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "icechunk>=0.1.0",
#     "dask[distributed]>=2024.1.0",
#     "coiled>=1.0.0",
#     "python-dotenv>=1.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
IMERG Half-Hourly Final East Africa — GCS Icechunk via Coiled
=============================================================

Downloads GPM IMERG Final Half-Hourly (GPM_3IMERGHH v07) precipitation data,
subsets to East Africa, writes directly to a GCS-backed Icechunk store.

Workers on Coiled download granules from NASA GES DISC using authenticated
HTTPS, read with netcdf4-python, subset to EA, and return numpy arrays.
Coordinator writes to GCS Icechunk and commits in batches.

GCS bucket: cpc_awc
Store prefix: ea_imerg_ic_store (production), test_ea_imerg_ic_store (test)

Subcommands:

  init     — Create empty template Icechunk store on GCS
  fill     — Populate with data using 20 Coiled workers
  verify   — Inspect store contents

Usage:
    # Test: 1 month
    uv run --python 3.12 imerg_hh_gcs_icechunk.py init \\
        --start-date 2024-12-01 --end-date 2024-12-31 \\
        --gcs-prefix test_ea_imerg_ic_store

    uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \\
        --start-date 2024-12-01 --end-date 2024-12-31 \\
        --gcs-prefix test_ea_imerg_ic_store --n-workers 20

    uv run --python 3.12 imerg_hh_gcs_icechunk.py verify \\
        --gcs-prefix test_ea_imerg_ic_store

    # Production: full archive
    uv run --python 3.12 imerg_hh_gcs_icechunk.py init \\
        --start-date 2000-06-01 --end-date 2025-03-01

    uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \\
        --start-date 2000-06-01 --end-date 2025-03-01 --n-workers 20
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("imerg_hh_gcs_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

IMERG_SHORT_NAME = "GPM_3IMERGHH"
IMERG_VERSION = "07"
IMERG_VAR = "precipitation"

# East Africa bounding box (expanded)
EA_LAT_MIN = -14.5
EA_LAT_MAX = 25.5
EA_LON_MIN = 19.5
EA_LON_MAX = 54.0
EA_BBOX = (EA_LON_MIN, EA_LAT_MIN, EA_LON_MAX, EA_LAT_MAX)

# GCS Icechunk store
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"

# Chunk sizes: 48 timesteps (1 day) x full lat x full lon
ZARR_CHUNK_TIME = 48
FILL_VALUE = np.float32(-9999.9)

# Coiled
COILED_WORKSPACE = "e4drr"
COILED_REGION = "us-east1"


# --- Helpers ---


def authenticate_earthdata():
    """Authenticate with NASA Earthdata."""
    import earthaccess

    auth = earthaccess.login(strategy="environment")
    logger.info("Earthdata authentication successful")

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        username = os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("EARTHDATA_PASSWORD")
        netrc_path.write_text(
            f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
        )
        netrc_path.chmod(0o600)

    return auth


def get_earthdata_credentials():
    """Return (username, password) for passing to Coiled workers."""
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise ValueError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in .env")
    return username, password


def get_gcs_storage(gcs_prefix):
    """Create GCS-backed Icechunk storage."""
    import icechunk

    sa_path = Path(__file__).parent / SERVICE_ACCOUNT_FILE
    if not sa_path.exists():
        raise FileNotFoundError(
            f"GCS service account file not found: {sa_path}\n"
            f"Expected: {SERVICE_ACCOUNT_FILE} in script directory"
        )

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=gcs_prefix,
        service_account_file=str(sa_path),
    )
    logger.info(f"GCS storage: gs://{GCS_BUCKET}/{gcs_prefix}")
    return storage


def search_imerg_hh(start_date: str, end_date: str):
    """Search for IMERG Half-Hourly granules and return results + download URLs."""
    import earthaccess

    logger.info(f"Searching {IMERG_SHORT_NAME} v{IMERG_VERSION}: {start_date} to {end_date}")
    results = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(start_date, end_date),
        bounding_box=EA_BBOX,
    )
    logger.info(f"Found {len(results)} granules")

    # Extract HTTPS download URLs
    download_urls = []
    for item in results:
        for url_info in item["umm"]["RelatedUrls"]:
            url = url_info.get("URL", "")
            if url.endswith(".HDF5") or url.endswith(".nc4"):
                if "data.gesdisc" in url or "gpm1.gesdisc" in url:
                    download_urls.append(url)
                    break

    logger.info(f"Extracted {len(download_urls)} download URLs")
    return results, download_urls


# --- Worker function (runs on Coiled) ---


def worker_download_and_subset(granule_info):
    """Worker: download one IMERG HH granule, subset to EA, return numpy.

    Runs on Coiled workers. Downloads via authenticated HTTPS with retry,
    validates the response before opening with netcdf4, subsets to EA.

    Retries up to 3 times with backoff to handle GES DISC rate limiting
    which returns HTML error pages or truncated files.
    """
    import os
    import tempfile
    import time as _time
    from pathlib import Path

    import netCDF4 as nc4
    import numpy as np
    import requests

    idx = granule_info["idx"]
    url = granule_info["url"]
    username = granule_info["username"]
    password = granule_info["password"]

    lat_min = granule_info["lat_min"]
    lat_max = granule_info["lat_max"]
    lon_min = granule_info["lon_min"]
    lon_max = granule_info["lon_max"]

    # Set up .netrc on worker for redirected auth
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        netrc_path.write_text(
            f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
        )
        netrc_path.chmod(0o600)

    MAX_RETRIES = 3
    MIN_FILE_SIZE = 500_000  # IMERG HH files are ~10 MB; <500 KB = bad

    for attempt in range(MAX_RETRIES):
        # Download via authenticated HTTPS
        session = requests.Session()
        session.auth = (username, password)
        response = session.get(url, allow_redirects=True, timeout=180)
        response.raise_for_status()

        # Validate: reject HTML error pages and truncated files
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            if attempt < MAX_RETRIES - 1:
                _time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Granule {idx}: server returned HTML (rate limited?) after {MAX_RETRIES} attempts"
            )

        if len(response.content) < MIN_FILE_SIZE:
            if attempt < MAX_RETRIES - 1:
                _time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Granule {idx}: file too small ({len(response.content)} bytes), "
                f"likely truncated or error page"
            )

        break  # download looks valid

    # Write to temp file and read with netcdf4
    with tempfile.NamedTemporaryFile(suffix=".HDF5", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        nc = nc4.Dataset(tmp_path)
        grp = nc.groups["Grid"] if "Grid" in nc.groups else nc

        lat = grp.variables["lat"][:]
        lon = grp.variables["lon"][:]

        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
        lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

        # IMERG HH: precipitation(time, lon, lat) -> transpose to (time, lat, lon)
        precip = grp.variables["precipitation"][:, lon_sl, lat_sl]
        precip = np.transpose(precip, (0, 2, 1)).astype(np.float32)

        nc.close()
    finally:
        os.unlink(tmp_path)

    return {"idx": idx, "data": precip}


# --- Phase 1: init ---


def init_store(args):
    """Create empty template Icechunk store on GCS.

    Probes one granule to get the EA lat/lon grid, then creates an empty
    template with the full time range on the GCS-backed Icechunk store.
    """
    import dask.array as da
    import earthaccess
    import icechunk
    import shutil
    import tempfile
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating IMERG HH EA template on GCS")
    logger.info("=" * 60)
    start = time.time()

    authenticate_earthdata()
    results, download_urls = search_imerg_hh(args.start_date, args.end_date)

    if not results:
        logger.error("No granules found!")
        return

    # Download first granule to probe EA grid
    logger.info("Downloading first granule to probe EA grid...")
    probe_dir = tempfile.mkdtemp(prefix="imerg_hh_probe_")
    probe_files = earthaccess.download([results[0]], probe_dir)
    logger.info(f"  Probing: {probe_files[0]}")

    import netCDF4 as nc4
    nc = nc4.Dataset(str(probe_files[0]))
    grp = nc.groups["Grid"] if "Grid" in nc.groups else nc
    lat_all = grp.variables["lat"][:]
    lon_all = grp.variables["lon"][:]
    nc.close()
    shutil.rmtree(probe_dir, ignore_errors=True)

    lat_mask = (lat_all >= EA_LAT_MIN) & (lat_all <= EA_LAT_MAX)
    lon_mask = (lon_all >= EA_LON_MIN) & (lon_all <= EA_LON_MAX)
    lat_ea = lat_all[lat_mask]
    lon_ea = lon_all[lon_mask]

    n_lat = len(lat_ea)
    n_lon = len(lon_ea)
    n_time = len(results)

    logger.info(f"  EA lat: {n_lat} pts [{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    logger.info(f"  EA lon: {n_lon} pts [{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")
    logger.info(f"  Time: {n_time} half-hourly steps")

    # Build time coordinates from granule metadata
    times = []
    for r in results:
        t_start = r["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        times.append(pd.Timestamp(t_start))
    time_coords = pd.DatetimeIndex(sorted(times)).tz_localize(None)
    logger.info(f"  Time range: {time_coords[0]} to {time_coords[-1]}")

    shape = (n_time, n_lat, n_lon)
    size_gb = n_time * n_lat * n_lon * 4 / (1024**3)
    logger.info(f"  Template shape: {shape}")
    logger.info(f"  Total size: {size_gb:.2f} GB")

    chunk_time = min(ZARR_CHUNK_TIME, n_time)
    chunks = (chunk_time, n_lat, n_lon)

    template = xr.Dataset(
        {
            IMERG_VAR: (
                ("time", "lat", "lon"),
                da.zeros(shape, chunks=shape, dtype=np.float32),
                {
                    "long_name": "Half-hourly precipitation rate (Final)",
                    "units": "mm/hr",
                    "source": f"{IMERG_SHORT_NAME} v{IMERG_VERSION}",
                },
            ),
        },
        coords={
            "time": time_coords,
            "lat": ("lat", lat_ea.astype(np.float64), {"units": "degrees_north"}),
            "lon": ("lon", lon_ea.astype(np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "title": "IMERG Half-Hourly Final — East Africa Subset",
            "source": f"{IMERG_SHORT_NAME} v{IMERG_VERSION}",
            "region": "East Africa",
            "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
            "temporal_resolution": "30 minutes",
        },
    )
    logger.info(f"  Template:\n{template}")

    # Set up GCS Icechunk store
    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    storage = get_gcs_storage(gcs_prefix)

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new GCS repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing GCS repository (will overwrite)")

    session = repo.writable_session("main")
    template.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding={
            IMERG_VAR: {
                "chunks": chunks,
                "fill_value": float(FILL_VALUE),
            },
        },
        consolidated=False,
    )
    session.commit("initialize IMERG HH EA template")
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("INIT COMPLETE")
    logger.info(f"  GCS: gs://{GCS_BUCKET}/{gcs_prefix}")
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Chunks: {chunks}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


# --- Phase 2: fill ---


def fill_store(args):
    """Fill EA template using Coiled Dask cluster writing to GCS Icechunk.

    Workers download granules from NASA GES DISC, read with netcdf4, subset
    to EA, return numpy arrays. Coordinator writes to GCS-backed Icechunk
    and commits in batches.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: Populating IMERG HH EA store via Coiled → GCS")
    logger.info("=" * 60)
    overall_start = time.time()

    authenticate_earthdata()
    results, download_urls = search_imerg_hh(args.start_date, args.end_date)

    if not download_urls:
        logger.error("No download URLs found!")
        return

    n_granules = len(download_urls)
    logger.info(f"  {n_granules} granules to process")

    username, password = get_earthdata_credentials()

    # Open GCS-backed Icechunk store
    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    storage = get_gcs_storage(gcs_prefix)
    target_repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )

    # Read template time axis for date-based index mapping
    _session = target_repo.readonly_session("main")
    _ds = xr.open_zarr(_session.store, consolidated=False)
    template_times = pd.DatetimeIndex(_ds.time.values)
    logger.info(f"  Template time axis: {len(template_times)} slots")

    # Map each granule to its template time index by timestamp
    import re
    granule_indexed = []
    for i, result in enumerate(results):
        try:
            t_start = result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            granule_ts = pd.Timestamp(t_start).tz_localize(None)
        except (KeyError, TypeError):
            logger.warning(f"  Could not extract timestamp from granule {i}, skipping")
            continue

        matches = np.where(template_times == granule_ts)[0]
        if len(matches) == 0:
            logger.warning(f"  Granule timestamp {granule_ts} not in template, skipping")
            continue
        t_idx = int(matches[0])
        granule_indexed.append((t_idx, i, download_urls[i]))

    logger.info(f"  Mapped {len(granule_indexed)} granules to template indices")

    # Resume detection
    completed_indices = set()
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill batch "):
                try:
                    range_str = msg.split(":")[0].replace("fill batch ", "")
                    b_start, b_end = range_str.split("-")
                    for idx in range(int(b_start), int(b_end) + 1):
                        completed_indices.add(idx)
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    remaining = [(t_idx, i, url) for t_idx, i, url in granule_indexed
                 if t_idx not in completed_indices]

    if completed_indices:
        logger.info(f"  Already filled: {len(completed_indices)} indices")
    if not remaining:
        logger.info("  All granules already filled!")
        return

    logger.info(f"  Remaining: {len(remaining)} granules")

    # Build granule info dicts for workers
    granule_infos = []
    for t_idx, i, url in remaining:
        granule_infos.append({
            "idx": t_idx,
            "url": url,
            "username": username,
            "password": password,
            "lat_min": EA_LAT_MIN,
            "lat_max": EA_LAT_MAX,
            "lon_min": EA_LON_MIN,
            "lon_max": EA_LON_MAX,
        })

    # Launch Coiled cluster
    n_workers = args.n_workers
    use_cluster = not getattr(args, "no_cluster", False)

    if use_cluster:
        logger.info(f"  Launching Coiled cluster with {n_workers} workers...")
        cluster = coiled.Cluster(
            name=f"imerg-hh-gcs-{int(time.time()) % 10000}",
            n_workers=[min(5, n_workers), n_workers],
            worker_vm_types="n2-standard-4",
            package_sync=True,
            region=COILED_REGION,
            workspace=COILED_WORKSPACE,
            idle_timeout="30 minutes",
        )
        client = distributed.Client(cluster)
        client.wait_for_workers(n_workers=min(5, n_workers), timeout=300)
        logger.info(f"  Cluster ready: {client.dashboard_link}")

    # Process in batches
    COMMIT_BATCH = args.commit_batch
    total_written = 0
    total_failed = 0
    failed_indices = []

    for batch_start in range(0, len(granule_infos), COMMIT_BATCH):
        batch = granule_infos[batch_start: batch_start + COMMIT_BATCH]
        batch_idx_min = batch[0]["idx"]
        batch_idx_max = batch[-1]["idx"]
        logger.info(
            f"  Batch: indices {batch_idx_min}-{batch_idx_max} "
            f"({len(batch)} granules, {total_written}/{len(granule_infos)} done)"
        )

        # Re-open storage for each batch to avoid stale sessions
        storage = get_gcs_storage(gcs_prefix)
        target_repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default()
        )
        session = target_repo.writable_session("main")
        batch_ok = 0
        batch_fail = 0

        if use_cluster:
            # Submit to Coiled workers
            futures = {}
            for g_info in batch:
                future = client.submit(
                    worker_download_and_subset, g_info,
                    key=f"imerg-hh-{g_info['idx']}",
                )
                futures[future] = g_info["idx"]

            for future in distributed.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    data = result["data"]
                    t_idx = result["idx"]

                    ds_write = xr.Dataset({
                        IMERG_VAR: (("time", "lat", "lon"), data),
                    })
                    ds_write.to_zarr(
                        session.store,
                        region={"time": slice(t_idx, t_idx + data.shape[0])},
                        consolidated=False,
                    )
                    del result

                    batch_ok += 1
                    total_written += 1
                    logger.info(f"    Wrote granule t_idx={t_idx}")

                except Exception as e:
                    batch_fail += 1
                    total_failed += 1
                    failed_indices.append(idx)
                    logger.error(f"    Granule t_idx={idx} FAILED: {e}")
        else:
            # Sequential (no cluster)
            for g_info in batch:
                try:
                    result = worker_download_and_subset(g_info)
                    data = result["data"]
                    t_idx = result["idx"]

                    ds_write = xr.Dataset({
                        IMERG_VAR: (("time", "lat", "lon"), data),
                    })
                    ds_write.to_zarr(
                        session.store,
                        region={"time": slice(t_idx, t_idx + data.shape[0])},
                        consolidated=False,
                    )
                    del result

                    batch_ok += 1
                    total_written += 1
                    logger.info(f"    Wrote granule t_idx={t_idx}")

                except Exception as e:
                    batch_fail += 1
                    total_failed += 1
                    failed_indices.append(g_info["idx"])
                    logger.error(f"    Granule t_idx={g_info['idx']} FAILED: {e}")

        if batch_fail == 0:
            session.commit(
                f"fill batch {batch_idx_min}-{batch_idx_max}: "
                f"{batch_ok}/{len(batch)} OK"
            )
            logger.info(f"  Committed batch {batch_idx_min}-{batch_idx_max}")
        else:
            logger.warning(
                f"  Batch {batch_idx_min}-{batch_idx_max} had {batch_fail} failures, "
                f"NOT committed — will retry on resume"
            )

    if use_cluster:
        client.close()
        cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  GCS: gs://{GCS_BUCKET}/{gcs_prefix}")
    logger.info(f"  Granules written: {total_written}/{n_granules}")
    logger.info(f"  Failed: {total_failed} — {failed_indices[:20]}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


# --- Phase 3: verify ---


def verify_store(args):
    """Inspect the IMERG HH EA Icechunk store on GCS."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting IMERG HH EA store on GCS")
    logger.info("=" * 60)

    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    storage = get_gcs_storage(gcs_prefix)
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    if "time" in ds.dims:
        logger.info(f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]}")
        logger.info(f"  Steps: {ds.sizes['time']}")
    if "lat" in ds.dims:
        logger.info(f"  Lat: {float(ds.lat.values[0]):.2f} -> {float(ds.lat.values[-1]):.2f}")
    if "lon" in ds.dims:
        logger.info(f"  Lon: {float(ds.lon.values[0]):.2f} -> {float(ds.lon.values[-1]):.2f}")

    for var in ds.data_vars:
        da = ds[var]
        logger.info(f"\nVariable '{var}': dtype={da.dtype}, shape={da.shape}")

    if IMERG_VAR in ds.data_vars:
        logger.info("\nSpot-check: loading first 5 timesteps...")
        try:
            sample = ds[IMERG_VAR].isel(time=slice(0, 5)).load()
            valid = sample.values[~np.isnan(sample.values) & (sample.values != FILL_VALUE)]
            logger.info(f"  Valid values: {len(valid)}/{sample.values.size}")
            if len(valid) > 0:
                logger.info(f"  Min: {valid.min():.4f}, Max: {valid.max():.4f}, Mean: {valid.mean():.4f}")
        except Exception as e:
            logger.error(f"  Spot-check failed: {e}")

    try:
        commits = list(repo.ancestry(branch="main"))
        logger.info(f"\nCommit history ({len(commits)} commits):")
        for c in commits[:10]:
            logger.info(f"  {c.message}")
        if len(commits) > 10:
            logger.info(f"  ... and {len(commits) - 10} more")
    except Exception:
        pass

    logger.info("\nVerification complete.")


# --- CLI ---


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IMERG HH Final EA — Coiled Dask to GCS Icechunk",
    )
    sub = parser.add_subparsers(dest="command")

    # Shared GCS args
    gcs_args = {
        "gcs_prefix": dict(type=str, default=None,
                           help=f"GCS prefix (default: {GCS_PREFIX})"),
    }

    # -- init --
    p_init = sub.add_parser("init", help="Create empty EA template on GCS")
    p_init.add_argument("--start-date", type=str, required=True)
    p_init.add_argument("--end-date", type=str, required=True)
    p_init.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])

    # -- fill --
    p_fill = sub.add_parser("fill", help="Fill EA store using Coiled → GCS")
    p_fill.add_argument("--start-date", type=str, required=True)
    p_fill.add_argument("--end-date", type=str, required=True)
    p_fill.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])
    p_fill.add_argument("--n-workers", type=int, default=20)
    p_fill.add_argument("--commit-batch", type=int, default=96,
                        help="Granules per commit (default 96 = 2 days of HH)")
    p_fill.add_argument("--no-cluster", action="store_true",
                        help="Run sequentially without Coiled (for testing)")

    # -- verify --
    p_verify = sub.add_parser("verify", help="Inspect GCS store contents")
    p_verify.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])

    args = parser.parse_args()

    if args.command == "init":
        init_store(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
