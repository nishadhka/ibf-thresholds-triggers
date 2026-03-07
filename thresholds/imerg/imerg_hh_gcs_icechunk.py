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
# ]
# ///
"""
IMERG Half-Hourly Final East Africa — THREDDS to GCS Icechunk via Coiled
=========================================================================

Reads GPM IMERG Final Half-Hourly (GPM_3IMERGHH v07) precipitation data from
GES DISC THREDDS ncml daily aggregations, subsets to East Africa via server-side
OPeNDAP subsetting, writes directly to a GCS-backed Icechunk store.

Data access method: THREDDS OPeNDAP with ncml daily aggregation
  - Each ncml file aggregates 48 half-hourly granules into one day
  - Server-side subsetting: only EA bbox data is transferred
  - ~365 OPeNDAP requests per year (vs ~17,500 granule downloads)
  - No S3 credentials needed, no HTTPS rate limiting

Why not HTTPS granule download?
  - 20 concurrent workers downloading ~10 MB files from GES DISC triggers
    rate limiting: HTML error pages, truncated files, worker segfaults
  - S3 direct access (s3://gesdisc-cumulus-prod-protected/) requires AWS
    us-west-2 — blocked from GCP by IAM policy (s3-same-region-access-role)

Why not single-granule OPeNDAP?
  - Individual OPeNDAP requests to GES DISC hit concurrent connection limits
  - THREDDS ncml aggregation bundles 48 granules per request, reducing load

GCS bucket: cpc_awc
Store prefix: ea_imerg_ic_store (production), test_ea_imerg_ic_store (test)
THREDDS base: https://gpm1.gesdisc.eosdis.nasa.gov/thredds/dodsC/aggregation/

Subcommands:

  init     — Create empty template Icechunk store on GCS
  fill     — Populate with data using Coiled workers reading THREDDS
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

# THREDDS ncml aggregation base URL
THREDDS_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/thredds/dodsC/aggregation"
THREDDS_COLLECTION = f"{IMERG_SHORT_NAME}.{IMERG_VERSION}"

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
    """Authenticate with NASA Earthdata and set up OPeNDAP prerequisite files.

    THREDDS OPeNDAP requires .netrc and .dodsrc files.
    """
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

    # .dodsrc required by netcdf-c for OPeNDAP auth (absolute paths only)
    home = str(Path.home())
    dodsrc_content = (
        f"HTTP.COOKIEJAR={home}/.urs_cookies\n"
        f"HTTP.NETRC={home}/.netrc\n"
    )
    for dodsrc_path in [Path.home() / ".dodsrc", Path.cwd() / ".dodsrc"]:
        dodsrc_path.write_text(dodsrc_content)

    cookie_jar = Path.home() / ".urs_cookies"
    if not cookie_jar.exists():
        cookie_jar.touch()
    cookie_jar.chmod(0o600)

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


def build_thredds_day_urls(start_date: str, end_date: str):
    """Build THREDDS ncml OPeNDAP URLs for each day in the date range.

    Each ncml file aggregates 48 half-hourly granules for one day.
    URL pattern: .../aggregation/GPM_3IMERGHH.07/{year}/
                 GPM_3IMERGHH.07_Aggregation_{year}{doy}.ncml.ncml

    Returns list of (date, url) tuples.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days = pd.date_range(start, end, freq="D")

    day_urls = []
    for day in days:
        year = day.year
        doy = day.day_of_year
        ncml_name = f"{THREDDS_COLLECTION}_Aggregation_{year}{doy:03d}.ncml.ncml"
        url = f"{THREDDS_BASE}/{THREDDS_COLLECTION}/{year}/{ncml_name}"
        day_urls.append((day, url))

    logger.info(f"Built {len(day_urls)} THREDDS day URLs: {start_date} to {end_date}")
    return day_urls


# --- Worker function (runs on Coiled) ---


def worker_read_thredds_day(day_info):
    """Worker: read one day from THREDDS ncml, subset to EA, return numpy.

    Runs on Coiled workers. Opens the daily ncml aggregation via OPeNDAP,
    server-side subsets to EA bounding box, returns 48 half-hourly timesteps.

    Only the EA subset is transferred over the network (~550 KB per day
    vs ~480 MB for all 48 global granules).
    """
    from pathlib import Path

    import numpy as np
    import xarray as xr

    t_start_idx = day_info["t_start_idx"]
    url = day_info["url"]
    username = day_info["username"]
    password = day_info["password"]
    lat_min = day_info["lat_min"]
    lat_max = day_info["lat_max"]
    lon_min = day_info["lon_min"]
    lon_max = day_info["lon_max"]

    # Set up .netrc and .dodsrc on worker for OPeNDAP auth
    home = str(Path.home())
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        netrc_path.write_text(
            f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
        )
        netrc_path.chmod(0o600)

    dodsrc_content = (
        f"HTTP.COOKIEJAR={home}/.urs_cookies\n"
        f"HTTP.NETRC={home}/.netrc\n"
    )
    for dp in [Path.home() / ".dodsrc", Path.cwd() / ".dodsrc"]:
        if not dp.exists():
            dp.write_text(dodsrc_content)

    cookie_jar = Path.home() / ".urs_cookies"
    if not cookie_jar.exists():
        cookie_jar.touch()
        cookie_jar.chmod(0o600)

    # Open THREDDS ncml via OPeNDAP — server-side subset (with retry)
    import random
    import time as _time

    max_retries = 5
    for attempt in range(max_retries):
        try:
            ds = xr.open_dataset(url, engine="netcdf4", decode_timedelta=False)
            subset = ds["precipitation"].sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            ).load()
            ds.close()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 2)
                _time.sleep(wait)
            else:
                raise

    # IMERG native order in THREDDS is (time, lon, lat) — transpose to (time, lat, lon)
    subset = subset.transpose("time", "lat", "lon")
    data = subset.values.astype(np.float32)
    n_t = data.shape[0]

    return {
        "t_start_idx": t_start_idx,
        "n_t": n_t,
        "data": data,
    }


# --- Phase 1: init ---


def init_store(args):
    """Create empty template Icechunk store on GCS.

    Probes one THREDDS day to get the EA lat/lon grid, then creates an empty
    template with the full time range.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating IMERG HH EA template on GCS via THREDDS probe")
    logger.info("=" * 60)
    start = time.time()

    authenticate_earthdata()

    day_urls = build_thredds_day_urls(args.start_date, args.end_date)
    if not day_urls:
        logger.error("No days in range!")
        return

    # Probe first day via THREDDS to get EA grid
    probe_date, probe_url = day_urls[0]
    logger.info(f"  Probing THREDDS: {probe_url}")
    ds_probe = xr.open_dataset(probe_url, engine="netcdf4", decode_timedelta=False)

    lat_all = ds_probe.lat.values
    lon_all = ds_probe.lon.values

    lat_mask = (lat_all >= EA_LAT_MIN) & (lat_all <= EA_LAT_MAX)
    lon_mask = (lon_all >= EA_LON_MIN) & (lon_all <= EA_LON_MAX)
    lat_ea = lat_all[lat_mask]
    lon_ea = lon_all[lon_mask]

    n_hh_per_day = ds_probe.sizes["time"]  # should be 48
    ds_probe.close()

    n_lat = len(lat_ea)
    n_lon = len(lon_ea)
    n_days = len(day_urls)
    n_time = n_days * n_hh_per_day

    logger.info(f"  EA lat: {n_lat} pts [{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    logger.info(f"  EA lon: {n_lon} pts [{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")
    logger.info(f"  Days: {n_days}, HH/day: {n_hh_per_day}, Total steps: {n_time}")

    # Build time coordinates: 48 half-hourly steps per day
    time_coords = pd.date_range(
        args.start_date,
        periods=n_time,
        freq="30min",
    )
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
                    "access_method": "THREDDS ncml OPeNDAP aggregation",
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
            "thredds_base": THREDDS_BASE,
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
    """Fill EA template using Coiled workers reading THREDDS, writing to GCS.

    Each worker reads one day (48 HH steps) from THREDDS ncml OPeNDAP with
    server-side subsetting. Coordinator writes to GCS-backed Icechunk and
    commits in batches.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: THREDDS ncml → Coiled workers → GCS Icechunk")
    logger.info("=" * 60)
    overall_start = time.time()

    authenticate_earthdata()
    username, password = get_earthdata_credentials()

    day_urls = build_thredds_day_urls(args.start_date, args.end_date)
    n_days = len(day_urls)
    if n_days == 0:
        logger.error("No days in range!")
        return

    logger.info(f"  {n_days} days to process ({n_days * 48} HH steps)")

    # Open GCS-backed Icechunk store
    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    storage = get_gcs_storage(gcs_prefix)
    target_repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )

    # Read template time axis
    _session = target_repo.readonly_session("main")
    _ds = xr.open_zarr(_session.store, consolidated=False)
    template_times = pd.DatetimeIndex(_ds.time.values)
    logger.info(f"  Template time axis: {len(template_times)} slots")

    # Map each day to its starting time index in the template
    day_infos = []
    for day, url in day_urls:
        # Find the first HH step for this day in the template
        day_start = pd.Timestamp(day).normalize()
        matches = np.where(template_times >= day_start)[0]
        if len(matches) == 0:
            logger.warning(f"  Day {day.date()} not in template, skipping")
            continue
        t_start_idx = int(matches[0])
        day_infos.append({
            "t_start_idx": t_start_idx,
            "day": str(day.date()),
            "url": url,
            "username": username,
            "password": password,
            "lat_min": EA_LAT_MIN,
            "lat_max": EA_LAT_MAX,
            "lon_min": EA_LON_MIN,
            "lon_max": EA_LON_MAX,
        })

    logger.info(f"  Mapped {len(day_infos)} days to template indices")

    # Resume detection — only consider commits after the most recent init
    completed_indices = set()
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("initialize "):
                break  # stop at last init — earlier commits are stale
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

    remaining = [d for d in day_infos if d["t_start_idx"] not in completed_indices]
    if completed_indices:
        logger.info(f"  Already filled: {len(completed_indices)} day-start indices")
    if not remaining:
        logger.info("  All days already filled!")
        return

    logger.info(f"  Remaining: {len(remaining)} days")

    # Launch Coiled cluster
    n_workers = args.n_workers
    use_cluster = not getattr(args, "no_cluster", False)

    if use_cluster:
        logger.info(f"  Launching Coiled cluster with {n_workers} workers...")
        cluster = coiled.Cluster(
            name=f"imerg-hh-thredds-{int(time.time()) % 10000}",
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

    # Process in batches (days per commit)
    COMMIT_BATCH = args.commit_batch  # days per commit
    total_written = 0
    total_failed = 0
    failed_days = []

    for batch_start in range(0, len(remaining), COMMIT_BATCH):
        batch = remaining[batch_start: batch_start + COMMIT_BATCH]
        batch_idx_min = batch[0]["t_start_idx"]
        batch_idx_max = batch[-1]["t_start_idx"]
        logger.info(
            f"  Batch: days {batch[0]['day']}..{batch[-1]['day']} "
            f"(t_idx {batch_idx_min}-{batch_idx_max}, "
            f"{len(batch)} days, {total_written}/{len(remaining)} done)"
        )

        # Re-open storage for each batch
        storage = get_gcs_storage(gcs_prefix)
        target_repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default()
        )
        session = target_repo.writable_session("main")
        batch_ok = 0
        batch_fail = 0

        if use_cluster:
            futures = {}
            for d_info in batch:
                future = client.submit(
                    worker_read_thredds_day, d_info,
                    key=f"thredds-{d_info['day']}",
                )
                futures[future] = d_info

            for future in distributed.as_completed(futures):
                d_info = futures[future]
                try:
                    result = future.result()
                    data = result["data"]
                    t_start = result["t_start_idx"]
                    n_t = result["n_t"]

                    ds_write = xr.Dataset({
                        IMERG_VAR: (("time", "lat", "lon"), data),
                    })
                    ds_write.to_zarr(
                        session.store,
                        region={"time": slice(t_start, t_start + n_t)},
                        consolidated=False,
                    )
                    del result

                    batch_ok += 1
                    total_written += 1
                    logger.info(f"    Wrote day {d_info['day']} (t_idx={t_start}, {n_t} steps)")

                except Exception as e:
                    batch_fail += 1
                    total_failed += 1
                    failed_days.append(d_info["day"])
                    logger.error(f"    Day {d_info['day']} FAILED: {e}")
        else:
            for d_info in batch:
                try:
                    result = worker_read_thredds_day(d_info)
                    data = result["data"]
                    t_start = result["t_start_idx"]
                    n_t = result["n_t"]

                    ds_write = xr.Dataset({
                        IMERG_VAR: (("time", "lat", "lon"), data),
                    })
                    ds_write.to_zarr(
                        session.store,
                        region={"time": slice(t_start, t_start + n_t)},
                        consolidated=False,
                    )
                    del result

                    batch_ok += 1
                    total_written += 1
                    logger.info(f"    Wrote day {d_info['day']} (t_idx={t_start}, {n_t} steps)")

                except Exception as e:
                    batch_fail += 1
                    total_failed += 1
                    failed_days.append(d_info["day"])
                    logger.error(f"    Day {d_info['day']} FAILED: {e}")

        if batch_ok > 0:
            session.commit(
                f"fill batch {batch_idx_min}-{batch_idx_max}: "
                f"{batch_ok}/{len(batch)} OK"
            )
            logger.info(
                f"  Committed batch {batch[0]['day']}..{batch[-1]['day']} "
                f"({batch_ok} OK, {batch_fail} failed)"
            )
        else:
            logger.warning(f"  Batch all failed, nothing to commit")

    if use_cluster:
        client.close()
        cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  GCS: gs://{GCS_BUCKET}/{gcs_prefix}")
    logger.info(f"  Days written: {total_written}/{n_days}")
    logger.info(f"  Failed: {total_failed} — {failed_days[:20]}")
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
        logger.info("\nSpot-check: loading first 48 timesteps (1 day)...")
        try:
            sample = ds[IMERG_VAR].isel(time=slice(0, 48)).load()
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
        description="IMERG HH Final EA — THREDDS to GCS Icechunk via Coiled",
    )
    sub = parser.add_subparsers(dest="command")

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
    p_fill = sub.add_parser("fill", help="Fill via THREDDS → Coiled → GCS")
    p_fill.add_argument("--start-date", type=str, required=True)
    p_fill.add_argument("--end-date", type=str, required=True)
    p_fill.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])
    p_fill.add_argument("--n-workers", type=int, default=10)
    p_fill.add_argument("--commit-batch", type=int, default=10,
                        help="Days per commit batch (default 10)")
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
