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
#     "huggingface_hub>=0.20.0",
# ]
# ///
"""
IMERG Half-Hourly Final East Africa — Coiled Dask to Icechunk on HuggingFace
=============================================================================

Downloads GPM IMERG Final Half-Hourly (GPM_3IMERGHH v07) precipitation data,
subsets to East Africa, writes to Icechunk store, uploads to HuggingFace.

Workers on Coiled download granules directly from NASA GES DISC using
authenticated HTTPS, read with netcdf4-python, subset to EA, and return
numpy arrays. Coordinator writes to Icechunk and commits in batches.

Follows the same fork-merge pattern as cmorph_east_africa_icechunk.py.

Subcommands:

  init     — Create empty template Icechunk store (local)
  fill     — Populate with real data using Coiled (5 workers)
  verify   — Inspect store contents
  upload   — Push local store to HuggingFace

Usage:
    # Step 1: Init template (1 day = 48 half-hourly granules)
    uv run --python 3.12 imerg_hh_ea_icechunk.py init \\
        --start-date 2024-12-01 --end-date 2024-12-01

    # Step 2: Fill using 5 Coiled workers
    uv run --python 3.12 imerg_hh_ea_icechunk.py fill \\
        --start-date 2024-12-01 --end-date 2024-12-01 \\
        --n-workers 5

    # Step 3: Verify
    uv run --python 3.12 imerg_hh_ea_icechunk.py verify

    # Step 4: Upload to HuggingFace
    uv run --python 3.12 imerg_hh_ea_icechunk.py upload
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
        logging.FileHandler("imerg_hh_ea_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

IMERG_SHORT_NAME = "GPM_3IMERGHH"
IMERG_VERSION = "07"
IMERG_VAR = "precipitation"

# East Africa bounding box
EA_LAT_MIN = -12.0
EA_LAT_MAX = 23.0
EA_LON_MIN = 21.0
EA_LON_MAX = 53.0
EA_BBOX = (EA_LON_MIN, EA_LAT_MIN, EA_LON_MAX, EA_LAT_MAX)

# HuggingFace
HF_REPO = "E4DRR/icechunk-stores"
HF_STORE_PREFIX = "test2_imerg-v7-ea-store"

# Local store path
LOCAL_STORE = "./imerg_hh_ea_local"

# Chunk sizes: 48 timesteps (1 day) x full lat x full lon
ZARR_CHUNK_TIME = 48
FILL_VALUE = np.float32(-9999.9)

# Coiled
COILED_WORKSPACE = "e4drr"
COILED_REGION = "us-east1"


# --- Auth helpers ---


def authenticate_earthdata():
    """Authenticate with NASA Earthdata and set up .netrc/.dodsrc."""
    import earthaccess

    auth = earthaccess.login(strategy="environment")
    logger.info("Earthdata authentication successful")

    # Ensure .netrc exists
    from pathlib import Path as P
    netrc_path = P.home() / ".netrc"
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


# --- Search ---


def search_imerg_hh(start_date: str, end_date: str):
    """Search for IMERG Half-Hourly granules."""
    import earthaccess

    logger.info(f"Searching IMERG HH Final: {start_date} to {end_date}")
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

    This runs on Coiled workers. Downloads the granule using authenticated
    HTTPS, reads with netcdf4-python, subsets to EA, returns data.

    Args:
        granule_info: dict with idx, url, username, password
    """
    import os
    import tempfile
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

    # Download via authenticated HTTPS with redirects
    session = requests.Session()
    session.auth = (username, password)
    response = session.get(url, allow_redirects=True, timeout=120)
    response.raise_for_status()

    # Write to temp file and read with netcdf4
    with tempfile.NamedTemporaryFile(suffix=".HDF5", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        nc = nc4.Dataset(tmp_path)

        # IMERG HH uses Grid group
        grp = nc.groups["Grid"] if "Grid" in nc.groups else nc

        lat = grp.variables["lat"][:]
        lon = grp.variables["lon"][:]

        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
        lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

        # IMERG HH: precipitation(time, lon, lat) — transpose to (time, lat, lon)
        precip = grp.variables["precipitation"][:, lon_sl, lat_sl]
        precip = np.transpose(precip, (0, 2, 1)).astype(np.float32)

        nc.close()
    finally:
        os.unlink(tmp_path)

    return {"idx": idx, "data": precip}


# --- Phase 1: init ---


def init_store(args):
    """Create empty template Icechunk store."""
    import dask.array as da
    import earthaccess
    import icechunk
    import tempfile
    import shutil
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating IMERG HH EA template")
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
    n_time = len(results)  # each granule = 1 timestep (30 min)

    logger.info(f"  EA lat: {n_lat} pts [{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    logger.info(f"  EA lon: {n_lon} pts [{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")
    logger.info(f"  Time: {n_time} half-hourly steps")

    # Build time coordinates from granule temporal info
    times = []
    for r in results:
        t_start = r["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        times.append(pd.Timestamp(t_start))
    time_coords = pd.DatetimeIndex(sorted(times)).tz_localize(None)
    logger.info(f"  Time range: {time_coords[0]} to {time_coords[-1]}")

    shape = (n_time, n_lat, n_lon)
    size_mb = n_time * n_lat * n_lon * 4 / (1024**2)
    logger.info(f"  Template shape: {shape}")
    logger.info(f"  Total size: {size_mb:.1f} MB")

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
            "title": "IMERG Half-Hourly Final - East Africa Subset",
            "source": f"{IMERG_SHORT_NAME} v{IMERG_VERSION}",
            "region": "East Africa",
            "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
            "temporal_resolution": "30 minutes",
        },
    )
    logger.info(f"  Template:\n{template}")

    store_path = args.local or LOCAL_STORE
    logger.info(f"Using local storage: {store_path}")
    storage = icechunk.local_filesystem_storage(path=store_path)

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

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
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Chunks: {chunks}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


# --- Phase 2: fill (Coiled) ---


def fill_store(args):
    """Fill EA template using Coiled Dask cluster.

    Workers download granules from NASA GES DISC with authenticated HTTPS,
    read with netcdf4, subset to EA, return numpy arrays.
    Coordinator writes to Icechunk and commits in batches.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: Populating IMERG HH EA store via Coiled")
    logger.info("=" * 60)
    overall_start = time.time()

    authenticate_earthdata()
    results, download_urls = search_imerg_hh(args.start_date, args.end_date)

    if not download_urls:
        logger.error("No download URLs found!")
        return

    n_granules = len(download_urls)
    logger.info(f"  {n_granules} granules to process")

    # Get credentials to pass to workers
    username, password = get_earthdata_credentials()

    # Open target Icechunk store
    store_path = args.local or LOCAL_STORE
    target_storage = icechunk.local_filesystem_storage(path=store_path)
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default()
    )

    # Resume detection
    completed_up_to = -1
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill batch "):
                try:
                    range_str = msg.split(":")[0].replace("fill batch ", "")
                    _, b_end = range_str.split("-")
                    b_end_int = int(b_end)
                    if b_end_int > completed_up_to:
                        completed_up_to = b_end_int
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    start_idx = completed_up_to + 1
    if start_idx > 0:
        logger.info(f"  Resuming from granule {start_idx} (0-{completed_up_to} done)")

    remaining = list(range(len(download_urls)))[start_idx:]
    if not remaining:
        logger.info("  All granules already filled!")
        return

    logger.info(f"  Remaining: {len(remaining)} granules")

    # Build granule info dicts for workers
    granule_infos = []
    for idx in remaining:
        granule_infos.append({
            "idx": idx,
            "url": download_urls[idx],
            "username": username,
            "password": password,
            "lat_min": EA_LAT_MIN,
            "lat_max": EA_LAT_MAX,
            "lon_min": EA_LON_MIN,
            "lon_max": EA_LON_MAX,
        })

    # Launch Coiled cluster
    n_workers = args.n_workers
    logger.info(f"  Launching Coiled cluster with {n_workers} workers...")
    cluster = coiled.Cluster(
        name=f"imerg-hh-ea-{int(time.time()) % 10000}",
        n_workers=[min(3, n_workers), n_workers],
        worker_vm_types="n2-standard-4",
        package_sync=True,
        region=COILED_REGION,
        workspace=COILED_WORKSPACE,
        idle_timeout="15 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(3, n_workers), timeout=300)
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
            f"  Batch: granules {batch_idx_min}-{batch_idx_max} "
            f"({len(batch)} granules, {total_written}/{len(granule_infos)} done)"
        )

        # Submit to Coiled workers
        futures = {}
        for g_info in batch:
            future = client.submit(
                worker_download_and_subset, g_info,
                key=f"imerg-hh-{g_info['idx']}",
            )
            futures[future] = g_info["idx"]

        # Collect results and write to Icechunk
        session = target_repo.writable_session("main")
        batch_ok = 0
        batch_fail = 0

        for future in distributed.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                data = result["data"]
                n_t = data.shape[0]
                t_start = result["idx"] * n_t
                t_end = t_start + n_t

                ds_write = xr.Dataset({
                    IMERG_VAR: (("time", "lat", "lon"), data),
                })
                ds_write.to_zarr(
                    session.store,
                    region={"time": slice(t_start, t_end)},
                    consolidated=False,
                )
                del result

                batch_ok += 1
                total_written += 1
                logger.info(f"    Wrote granule {idx}")

            except Exception as e:
                batch_fail += 1
                total_failed += 1
                failed_indices.append(idx)
                logger.error(f"    Granule {idx} FAILED: {e}")

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

    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  Granules written: {total_written}/{n_granules}")
    logger.info(f"  Failed: {total_failed} — {failed_indices[:20]}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)


# --- Phase 3: verify ---


def verify_store(args):
    """Inspect the IMERG HH EA Icechunk store."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting IMERG HH EA store")
    logger.info("=" * 60)

    store_path = args.local or LOCAL_STORE
    storage = icechunk.local_filesystem_storage(path=store_path)
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
    except Exception:
        pass

    logger.info("\nVerification complete.")


# --- Phase 4: upload ---


def upload_store(args):
    """Upload local Icechunk store to HuggingFace."""
    from huggingface_hub import HfApi

    logger.info("=" * 60)
    logger.info(f"UPLOAD: Pushing to HF {HF_REPO}/{HF_STORE_PREFIX}")
    logger.info("=" * 60)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
    if not hf_token:
        logger.error("HF token not set! Add HF_TOKEN or hf to .env")
        return

    store_path = Path(args.local or LOCAL_STORE)
    if not store_path.exists():
        logger.error(f"Store not found: {store_path}")
        return

    # Safety check
    files = list(store_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    BLOCKED = {".env", ".netrc", ".dodsrc"}
    bad = [f for f in files if f.name in BLOCKED or f.suffix in {".py", ".ipynb"}]
    if bad:
        logger.error(f"ABORT: Store contains non-store files: {[str(b) for b in bad]}")
        return

    total_size = sum(f.stat().st_size for f in files)
    logger.info(f"  Store: {len(files)} files, {total_size / 1024:.1f} KB")

    api = HfApi(token=hf_token)
    api.upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=str(store_path),
        path_in_repo=HF_STORE_PREFIX,
        commit_message=f"Upload {HF_STORE_PREFIX} ({len(files)} files, {total_size/1024:.0f} KB)",
    )
    logger.info("Upload complete!")


# --- CLI ---


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IMERG Half-Hourly Final EA — Coiled Dask to Icechunk on HF",
    )
    sub = parser.add_subparsers(dest="command")

    # -- init --
    p_init = sub.add_parser("init", help="Create empty EA template store")
    p_init.add_argument("--start-date", type=str, required=True)
    p_init.add_argument("--end-date", type=str, required=True)
    p_init.add_argument("--local", type=str, default=None)

    # -- fill --
    p_fill = sub.add_parser("fill", help="Fill EA store using Coiled cluster")
    p_fill.add_argument("--start-date", type=str, required=True)
    p_fill.add_argument("--end-date", type=str, required=True)
    p_fill.add_argument("--local", type=str, default=None)
    p_fill.add_argument("--n-workers", type=int, default=5)
    p_fill.add_argument("--commit-batch", type=int, default=48,
                        help="Granules per commit (default 48 = 1 day)")

    # -- verify --
    p_verify = sub.add_parser("verify", help="Inspect store contents")
    p_verify.add_argument("--local", type=str, default=None)

    # -- upload --
    p_upload = sub.add_parser("upload", help="Upload store to HuggingFace")
    p_upload.add_argument("--local", type=str, default=None)

    args = parser.parse_args()

    if args.command == "init":
        init_store(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "verify":
        verify_store(args)
    elif args.command == "upload":
        upload_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
