#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "earthaccess>=0.12.0",
#     "h5netcdf>=1.3.0",
#     "h5py>=3.10.0",
#     "cftime>=1.6.0",
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
IMERG Half-Hourly East Africa — NASA Earthdata S3 → GCS Icechunk via Coiled
============================================================================

Reads GPM IMERG Half-Hourly v07 precipitation (Final / Late / Early) as
native HDF5 granules from NASA Earthdata Cloud via the `earthaccess`
client, subsets to East Africa, and writes to a GCS-backed Icechunk store.

Data access
  - 48 HDF5 granules per day, ~10 MB each
  - `earthaccess.search_data` finds granule URLs; `earthaccess.open`
    returns fsspec file objects that stream bytes directly
  - From us-west-2 AWS: direct S3 reads (s3://gesdisc-cumulus-prod-protected/)
  - From other regions (e.g. Coiled GCP workers): HTTPS from
    data.gesdisc.earthdata.nasa.gov — bearer-token auth handled by earthaccess

Why earthaccess rather than THREDDS?
  NASA GES DISC's THREDDS aggregation endpoint is being phased out and has
  been repeatedly unstable (5xx responses, server-side DNS failures).  Direct
  HDF5 access via earthaccess is the supported path going forward.

GCS bucket: cpc_awc
Store prefix: ea_imerg_ic_store (production), test_ea_imerg_ic_store (test)
Granule host: data.gesdisc.earthdata.nasa.gov (HTTPS) or
              s3://gesdisc-cumulus-prod-protected/ (from us-west-2)

Subcommands:

  init     — Create empty template Icechunk store on GCS
  fill     — Populate with data using Coiled workers reading S3 granules
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

IMERG_VAR = "precipitation"

# IMERG product hierarchy: Final > Late > Early
# Higher quality products should replace lower quality ones
IMERG_PRODUCTS = {
    "final": {"short_name": "GPM_3IMERGHH", "version": "07", "label": "Final"},
    "late":  {"short_name": "GPM_3IMERGHHL", "version": "07", "label": "Late"},
    "early": {"short_name": "GPM_3IMERGHHE", "version": "07", "label": "Early"},
}
PRODUCT_QUALITY = {"final": 3, "late": 2, "early": 1}
DEFAULT_PRODUCT = "final"

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
# Use NaN as fill value so consumers can distinguish "never written" from
# "real zero rainfall".  Earlier code used -9999.9 but extend_store wrote
# materialised zeros over the unfilled region, defeating that signal.
FILL_VALUE = np.float32(np.nan)

# Coiled
COILED_WORKSPACE = "e4drr"
COILED_REGION = "us-east1"


# --- Helpers ---


def authenticate_earthdata():
    """Authenticate with NASA Earthdata Cloud via earthaccess.

    Reads EARTHDATA_USERNAME / EARTHDATA_PASSWORD from the environment and
    caches a bearer token in ~/.netrc for subsequent fsspec/HTTP clients.
    No .dodsrc/OPeNDAP rigging required — we stream HDF5 granules directly.
    """
    import earthaccess

    auth = earthaccess.login(strategy="environment")
    logger.info("Earthdata authentication successful")
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


def build_day_list(start_date: str, end_date: str, product: str = DEFAULT_PRODUCT):
    """Enumerate (day, short_name) tuples for each day in the date range.

    Granule URLs are discovered per-day via earthaccess.search_data at fill
    time — no URL construction here, since NASA switched away from
    predictable THREDDS ncml paths to Earthdata Cloud's CMR catalogue.
    """
    prod = IMERG_PRODUCTS[product]
    short_name = prod["short_name"]

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days = pd.date_range(start, end, freq="D")

    day_list = [(day, short_name) for day in days]
    logger.info(
        f"Built {len(day_list)} day entries ({product} / {short_name}): "
        f"{start_date} to {end_date}"
    )
    return day_list


# --- Worker function (runs on Coiled) ---


def worker_read_s3_day(day_info):
    """Worker: stream 48 HDF5 granules from NASA Earthdata for one day,
    subset to EA, return a numpy array in (time, lat, lon) order.

    Runs on Coiled workers or locally.  Uses `earthaccess.open` which
    chooses between direct S3 (in us-west-2) and authenticated HTTPS.
    Only the EA bbox is materialised in RAM — global rasters are read
    lazily and subset before `.load()`.
    """
    import os as _os

    import earthaccess
    import numpy as np
    import pandas as pd
    import xarray as xr

    t_start_idx = day_info["t_start_idx"]
    day_str = day_info["day"]
    short_name = day_info["short_name"]
    username = day_info["username"]
    password = day_info["password"]
    lat_min = day_info["lat_min"]
    lat_max = day_info["lat_max"]
    lon_min = day_info["lon_min"]
    lon_max = day_info["lon_max"]

    # Ensure worker env carries the creds so earthaccess can log in
    _os.environ.setdefault("EARTHDATA_USERNAME", username)
    _os.environ.setdefault("EARTHDATA_PASSWORD", password)
    earthaccess.login(strategy="environment")

    day = pd.Timestamp(day_str)
    day_end = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    import random
    import time as _time

    max_retries = 5
    for attempt in range(max_retries):
        try:
            results = earthaccess.search_data(
                short_name=short_name,
                version="07",
                temporal=(
                    day.strftime("%Y-%m-%dT%H:%M:%S"),
                    day_end.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            if len(results) != 48:
                raise RuntimeError(
                    f"Expected 48 granules for {day.date()} / {short_name}, "
                    f"got {len(results)}"
                )
            files = earthaccess.open(results)
            ds = xr.open_mfdataset(
                files,
                engine="h5netcdf",
                group="Grid",
                combine="nested",
                concat_dim="time",
                decode_times=False,  # skip cftime decoding; we don't need
                                     # the times — indices are predetermined
            )
            subset = ds["precipitation"].sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            ).load()
            ds.close()
            break
        except Exception:
            if attempt < max_retries - 1:
                _time.sleep((2 ** attempt) + random.uniform(0, 2))
            else:
                raise

    # HDF5 native order is (time, lon, lat) — transpose to (time, lat, lon)
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

    Probes one HDF5 granule via earthaccess to get the EA lat/lon grid,
    then creates an empty template with the full time range.
    """
    import dask.array as da
    import earthaccess
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating IMERG HH EA template on GCS via earthaccess probe")
    logger.info("=" * 60)
    start = time.time()

    authenticate_earthdata()

    day_list = build_day_list(args.start_date, args.end_date, product=DEFAULT_PRODUCT)
    if not day_list:
        logger.error("No days in range!")
        return

    # Probe first day via earthaccess to get the native grid
    probe_date, probe_short = day_list[0]
    logger.info(
        f"  Probing {probe_short} granules for {probe_date.date()} ..."
    )
    probe_end = probe_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    probe_results = earthaccess.search_data(
        short_name=probe_short, version="07",
        temporal=(
            probe_date.strftime("%Y-%m-%dT%H:%M:%S"),
            probe_end.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    if len(probe_results) != 48:
        raise RuntimeError(
            f"Probe expected 48 granules for {probe_date.date()}, "
            f"got {len(probe_results)}"
        )
    probe_files = earthaccess.open([probe_results[0]])
    ds_probe = xr.open_dataset(
        probe_files[0], engine="h5netcdf", group="Grid", decode_times=False,
    )

    lat_all = ds_probe.lat.values
    lon_all = ds_probe.lon.values

    lat_mask = (lat_all >= EA_LAT_MIN) & (lat_all <= EA_LAT_MAX)
    lon_mask = (lon_all >= EA_LON_MIN) & (lon_all <= EA_LON_MAX)
    lat_ea = lat_all[lat_mask]
    lon_ea = lon_all[lon_mask]
    ds_probe.close()

    n_hh_per_day = 48  # IMERG HH is always 48 half-hours per day
    n_lat = len(lat_ea)
    n_lon = len(lon_ea)
    n_days = len(day_list)
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
                    "long_name": "Half-hourly precipitation rate",
                    "units": "mm/hr",
                    "source": "IMERG v07 (Final/Late/Early)",
                    "access_method": "NASA Earthdata Cloud (HDF5 granules via earthaccess)",
                },
            ),
        },
        coords={
            "time": time_coords,
            "lat": ("lat", lat_ea.astype(np.float64), {"units": "degrees_north"}),
            "lon": ("lon", lon_ea.astype(np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "title": "IMERG Half-Hourly — East Africa Subset",
            "source": "IMERG v07 (Final/Late/Early)",
            "region": "East Africa",
            "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
            "temporal_resolution": "30 minutes",
            "access_method": "earthaccess (HDF5 via CMR)",
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
    """Fill EA template using Coiled workers reading Earthdata S3/HTTPS,
    writing to GCS.

    Each worker resolves 48 HH granules for a day via earthaccess,
    subsets to EA, and returns a numpy array.  Coordinator writes to
    GCS-backed Icechunk and commits in batches.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    product = getattr(args, "product", DEFAULT_PRODUCT)
    prod_info = IMERG_PRODUCTS[product]
    prod_tag = f"[{prod_info['short_name']}]"

    logger.info("=" * 60)
    logger.info(
        f"FILL: Earthdata S3/HTTPS → GCS Icechunk "
        f"(product: {product} / {prod_info['short_name']})"
    )
    logger.info("=" * 60)
    overall_start = time.time()

    authenticate_earthdata()
    username, password = get_earthdata_credentials()

    day_urls = build_day_list(args.start_date, args.end_date, product=product)
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
    for day, short_name in day_urls:
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
            "short_name": short_name,
            "username": username,
            "password": password,
            "lat_min": EA_LAT_MIN,
            "lat_max": EA_LAT_MAX,
            "lon_min": EA_LON_MIN,
            "lon_max": EA_LON_MAX,
        })

    logger.info(f"  Mapped {len(day_infos)} days to template indices")

    # Resume detection — product-quality-aware
    # Track which indices have been filled and by which product quality level.
    # Skip indices already filled by same or higher quality product.
    # Allow overwriting indices filled by a lower quality product.
    filled_quality = {}  # t_start_idx -> max quality level
    current_quality = PRODUCT_QUALITY[product]
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("initialize "):
                break  # stop at last init — earlier commits are stale
            if msg.startswith("fill batch "):
                try:
                    # Parse product tag from commit: "fill batch X-Y [GPM_3IMERGHHE]: N/M OK"
                    commit_quality = 0
                    for pname, pinfo in IMERG_PRODUCTS.items():
                        if f"[{pinfo['short_name']}]" in msg:
                            commit_quality = PRODUCT_QUALITY[pname]
                            break
                    if commit_quality == 0:
                        # Old commits without product tag are assumed Final
                        commit_quality = PRODUCT_QUALITY["final"]

                    range_part = msg.split("[")[0] if "[" in msg else msg.split(":")[0]
                    range_str = range_part.replace("fill batch ", "").strip().rstrip("-")
                    b_start, b_end = range_str.split("-")
                    for idx in range(int(b_start), int(b_end) + 1):
                        filled_quality[idx] = max(filled_quality.get(idx, 0), commit_quality)
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    # Skip days already filled by same or higher quality.  --force disables
    # this — useful after a tail-truncate repair, where the historical
    # batch commits still reference indices whose data was dropped.
    force = bool(getattr(args, "force", False))
    remaining = []
    for d in day_infos:
        idx = d["t_start_idx"]
        existing_q = filled_quality.get(idx, 0)
        if not force and existing_q >= current_quality:
            continue  # already filled by same or better product
        remaining.append(d)

    skip_count = len(day_infos) - len(remaining)
    if skip_count:
        logger.info(f"  Already filled (same/higher quality): {skip_count} days")
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
            name=f"imerg-hh-s3-{int(time.time()) % 10000}",
            n_workers=n_workers,
            worker_vm_types="n2-standard-4",
            package_sync=True,
            region=COILED_REGION,
            workspace=COILED_WORKSPACE,
        )
        client = distributed.Client(cluster)
        client.wait_for_workers(n_workers=n_workers, timeout=300)
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
                    worker_read_s3_day, d_info,
                    key=f"earthaccess-{d_info['day']}",
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
                    result = worker_read_s3_day(d_info)
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
                f"fill batch {batch_idx_min}-{batch_idx_max} "
                f"{prod_tag}: {batch_ok}/{len(batch)} OK"
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
            valid = sample.values[~np.isnan(sample.values)]
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


# --- Phase 4: extend ---


def extend_store(args):
    """Extend the template time axis to cover new dates.

    Appends new empty time slots to the existing store without touching
    existing data. Use this before filling Late/Early data for dates
    beyond the original template range.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("EXTEND: Growing template time axis")
    logger.info("=" * 60)

    gcs_prefix = args.gcs_prefix or GCS_PREFIX
    storage = get_gcs_storage(gcs_prefix)
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    current_end = pd.Timestamp(ds.time.values[-1])
    new_end = pd.Timestamp(args.end_date)
    lat = ds.lat.values
    lon = ds.lon.values
    n_lat = len(lat)
    n_lon = len(lon)

    logger.info(f"  Current end: {current_end}")
    logger.info(f"  Requested end: {new_end}")

    if new_end <= current_end:
        logger.info("  Template already covers this date range, nothing to extend.")
        return

    # New time slots: start at the next 30-min boundary after current_end.
    # Build whole days only (48 HH steps each) to keep day boundaries aligned
    # with chunk boundaries.  Earlier code used
    #     pd.date_range(next_day, new_end, freq="30min")
    # which is *inclusive* on both ends — that produced an extra midnight
    # timestamp at the tail (e.g. ...2026-03-09T00:00).  The next extend's
    # `(current_end + 30min).normalize()` then rounded back to that same
    # midnight, creating duplicate timestamps in the time axis.
    next_slot = current_end + pd.Timedelta(minutes=30)
    if next_slot.time() != pd.Timestamp("2000-01-01").time():
        raise RuntimeError(
            f"Cannot extend: existing time axis ends at {current_end}, "
            f"which is not a clean half-hour-before-midnight boundary. "
            f"Repair the time axis first."
        )
    last_day = pd.Timestamp(args.end_date).normalize()
    n_new_days = (last_day - next_slot.normalize()).days + 1
    if n_new_days <= 0:
        logger.info("  No new time slots needed.")
        return
    n_new = n_new_days * 48
    new_times = pd.date_range(next_slot, periods=n_new, freq="30min")

    logger.info(f"  Adding {n_new} time slots ({n_new_days} days)")
    logger.info(f"  New range: {new_times[0]} to {new_times[-1]}")

    # Create append dataset with empty data — compute=False on to_zarr means
    # only the array metadata is written; the new chunks remain unwritten and
    # therefore read back as the array's fill_value (NaN), letting consumers
    # distinguish "never filled" from "real zero rainfall".
    shape = (n_new, n_lat, n_lon)
    chunk_time = min(ZARR_CHUNK_TIME, n_new)
    chunks = (chunk_time, n_lat, n_lon)

    append_ds = xr.Dataset(
        {
            IMERG_VAR: (
                ("time", "lat", "lon"),
                da.zeros(shape, chunks=shape, dtype=np.float32),
            ),
        },
        coords={
            "time": new_times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )

    # Append to existing store — compute=False so the data chunks are not
    # materialised (only the time-coord and array length grow).
    session = repo.writable_session("main")
    append_ds.to_zarr(
        session.store,
        mode="a",
        append_dim="time",
        compute=False,
        consolidated=False,
    )
    session.commit(f"extend template to {args.end_date} (+{n_new_days} days)")

    logger.info(f"  Extended! New total: {ds.sizes['time'] + n_new} time slots")
    logger.info("=" * 60)


# --- CLI ---


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IMERG HH EA — Earthdata S3 → GCS Icechunk via Coiled",
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
    p_fill = sub.add_parser("fill", help="Fill via earthaccess → Coiled → GCS")
    p_fill.add_argument("--start-date", type=str, required=True)
    p_fill.add_argument("--end-date", type=str, required=True)
    p_fill.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])
    p_fill.add_argument("--product", type=str, default=DEFAULT_PRODUCT,
                        choices=list(IMERG_PRODUCTS.keys()),
                        help="IMERG product: final (default), late, or early")
    p_fill.add_argument("--n-workers", type=int, default=10)
    p_fill.add_argument("--commit-batch", type=int, default=10,
                        help="Days per commit batch (default 10)")
    p_fill.add_argument("--no-cluster", action="store_true",
                        help="Run sequentially without Coiled (for testing)")
    p_fill.add_argument("--force", action="store_true",
                        help="Re-fill days even if commit history says they "
                             "are already covered (use after a tail repair).")

    # -- extend --
    p_extend = sub.add_parser("extend",
                              help="Extend template time axis to cover new dates")
    p_extend.add_argument("--end-date", type=str, required=True,
                          help="New end date (must be after current end)")
    p_extend.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])

    # -- verify --
    p_verify = sub.add_parser("verify", help="Inspect GCS store contents")
    p_verify.add_argument("--gcs-prefix", **gcs_args["gcs_prefix"])

    args = parser.parse_args()

    if args.command == "init":
        init_store(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "extend":
        extend_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
