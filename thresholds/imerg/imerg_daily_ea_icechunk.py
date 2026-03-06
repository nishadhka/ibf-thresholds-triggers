#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
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
IMERG Daily Early East Africa — OPeNDAP Stream to Icechunk on HuggingFace
==========================================================================

Streams GPM IMERG Daily Early (GPM_3IMERGDE v07) precipitation data via
OPeNDAP using netcdf4-python, subsets to East Africa, and writes to an
Icechunk store on HuggingFace (E4DRR/icechunk-stores as imerg-v7-ea-store).

Uses Dask/Coiled cluster for parallel OPeNDAP reads and writing.

Subcommands:

  init     — Create empty template Icechunk store on HuggingFace
  fill     — Populate with real data via OPeNDAP using Dask/Coiled

Usage:
    # Step 1: Authenticate with Earthdata and create template
    uv run imerg_daily_ea_icechunk.py init \\
        --start-date 2000-06-01 --end-date 2024-12-31

    # Step 2: Fill with data from OPeNDAP (needs Coiled cluster)
    uv run imerg_daily_ea_icechunk.py fill \\
        --start-date 2000-06-01 --end-date 2024-12-31 \\
        --n-workers 20

    # Step 3: Verify
    uv run imerg_daily_ea_icechunk.py verify
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("imerg_daily_ea_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---

IMERG_SHORT_NAME = "GPM_3IMERGDE"
IMERG_VERSION = "07"
IMERG_VAR = "precipitation"

# East Africa bounding box (same as CMORPH pipeline)
EA_LAT_MIN = -12.0
EA_LAT_MAX = 23.0
EA_LON_MIN = 21.0
EA_LON_MAX = 53.0
EA_BBOX = (EA_LON_MIN, EA_LAT_MIN, EA_LON_MAX, EA_LAT_MAX)

# HuggingFace Icechunk store
HF_REPO = "E4DRR/icechunk-stores"
HF_STORE_PREFIX = "imerg-v7-ea-store"

# Chunk sizes for Icechunk store: 30 days x full lat x full lon
ZARR_CHUNK_TIME = 30
FILL_VALUE = np.float32(-9999.9)

# Coiled cluster defaults
COILED_WORKSPACE = "e4drr"
COILED_REGION = "us-east1"


# --- Helpers ---


def authenticate_earthdata():
    """Authenticate with NASA Earthdata and set up OPeNDAP prerequisite files.

    netcdf4-python OPeNDAP access requires both .netrc and .dodsrc files.
    earthaccess.login() creates .netrc; we create .dodsrc ourselves.
    """
    import earthaccess
    from pathlib import Path

    auth = earthaccess.login(strategy="environment")
    logger.info("Earthdata authentication successful")

    # Ensure .netrc exists (earthaccess should have created it)
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        username = os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("EARTHDATA_PASSWORD")
        netrc_path.write_text(
            f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
        )
        netrc_path.chmod(0o600)
        logger.info(f"Created {netrc_path}")

    # Create .dodsrc in home and working directory (required by netcdf4-python)
    # Must use absolute paths — netcdf-c does not expand ~
    home = str(Path.home())
    dodsrc_content = (
        f"HTTP.COOKIEJAR={home}/.urs_cookies\n"
        f"HTTP.NETRC={home}/.netrc\n"
    )
    for dodsrc_path in [Path.home() / ".dodsrc", Path.cwd() / ".dodsrc"]:
        dodsrc_path.write_text(dodsrc_content)
        logger.info(f"Created {dodsrc_path}")

    # Create cookie jar file if it doesn't exist
    cookie_jar = Path.home() / ".urs_cookies"
    if not cookie_jar.exists():
        cookie_jar.touch()
    cookie_jar.chmod(0o600)

    return auth


def search_imerg_daily(start_date: str, end_date: str):
    """Search for IMERG Daily Early granules and return OPeNDAP URLs."""
    import earthaccess

    logger.info(f"Searching IMERG Daily Early: {start_date} to {end_date}")
    results = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(start_date, end_date),
        bounding_box=EA_BBOX,
    )
    logger.info(f"Found {len(results)} granules")

    # Extract OPeNDAP URLs
    opendap_urls = []
    for item in results:
        for urls in item["umm"]["RelatedUrls"]:
            if "OPENDAP" in urls.get("Description", "").upper():
                opendap_urls.append(urls["URL"])

    logger.info(f"Extracted {len(opendap_urls)} OPeNDAP URLs")
    return results, opendap_urls


def read_imerg_opendap_subset(url: str, lat_min: float, lat_max: float,
                               lon_min: float, lon_max: float):
    """Read a single IMERG granule via OPeNDAP using netCDF4, subset to EA.

    IMERG daily files use a Grid group and have dimensions (time, lon, lat).
    """
    import netCDF4 as nc4

    try:
        nc = nc4.Dataset(url)

        # IMERG daily uses Grid group
        if "Grid" in nc.groups:
            grp = nc.groups["Grid"]
        else:
            grp = nc

        lat = grp.variables["lat"][:]
        lon = grp.variables["lon"][:]

        # Find EA indices
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

        lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
        lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)

        # IMERG daily: precipitation(time, lon, lat) — note lon/lat order
        precip = grp.variables[IMERG_VAR][:, lon_slice, lat_slice]

        # Transpose to (time, lat, lon) for consistency
        precip = np.transpose(precip, (0, 2, 1))

        lat_subset = lat[lat_slice]
        lon_subset = lon[lon_slice]

        nc.close()
        return precip.astype(np.float32), lat_subset, lon_subset

    except Exception as e:
        logger.error(f"Failed to read {url}: {e}")
        raise


def get_hf_icechunk_storage():
    """Create Icechunk storage config for HuggingFace."""
    import icechunk

    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
    if not hf_token:
        raise ValueError(
            "Missing HF token! Set it in .env or environment:\n"
            "  HF_TOKEN=hf_your_token_here  (or hf=hf_your_token_here)"
        )

    storage = icechunk.s3_storage(
        bucket=f"hf://datasets/{HF_REPO}",
        prefix=HF_STORE_PREFIX,
        region="us-east-1",
        access_key_id=hf_token,
        secret_access_key=hf_token,
        endpoint_url="https://s3.us-east-1.amazonaws.com",
    )
    return storage


# --- Phase 1: init ---


def init_ea_store(args):
    """Create empty template Icechunk store on HuggingFace.

    Probes one OPeNDAP granule to get the EA lat/lon grid, then creates
    an empty template with the full time range.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating IMERG Daily EA template on HuggingFace")
    logger.info("=" * 60)
    start = time.time()

    # Authenticate and search
    authenticate_earthdata()
    results, opendap_urls = search_imerg_daily(args.start_date, args.end_date)

    if not opendap_urls:
        logger.error("No OPeNDAP URLs found!")
        return

    # Download and probe first granule for lat/lon grid
    import earthaccess
    import tempfile

    logger.info("Downloading first granule to probe EA grid...")
    probe_dir = tempfile.mkdtemp(prefix="imerg_probe_")
    probe_files = earthaccess.download([results[0]], probe_dir)
    logger.info(f"  Probing: {probe_files[0]}")
    import netCDF4 as nc4
    nc = nc4.Dataset(str(probe_files[0]))
    grp = nc.groups["Grid"] if "Grid" in nc.groups else nc
    lat_all = grp.variables["lat"][:]
    lon_all = grp.variables["lon"][:]
    nc.close()
    lat_mask = (lat_all >= EA_LAT_MIN) & (lat_all <= EA_LAT_MAX)
    lon_mask = (lon_all >= EA_LON_MIN) & (lon_all <= EA_LON_MAX)
    lat_ea = lat_all[lat_mask]
    lon_ea = lon_all[lon_mask]
    # Cleanup probe
    import shutil
    shutil.rmtree(probe_dir, ignore_errors=True)
    n_lat = len(lat_ea)
    n_lon = len(lon_ea)
    logger.info(f"  EA lat: {n_lat} pts [{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    logger.info(f"  EA lon: {n_lon} pts [{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")

    # Build time coordinate from search results (one timestep per daily granule)
    n_time = len(results)
    time_coords = pd.date_range(args.start_date, periods=n_time, freq="D")
    logger.info(f"  Time: {n_time} days [{time_coords[0]} .. {time_coords[-1]}]")

    shape = (n_time, n_lat, n_lon)
    size_gb = n_time * n_lat * n_lon * 4 / (1024**3)
    logger.info(f"  Template shape: {shape}")
    logger.info(f"  Total size: {size_gb:.2f} GB")

    # Chunk sizes
    chunk_time = min(ZARR_CHUNK_TIME, n_time)
    chunks = (chunk_time, n_lat, n_lon)

    # Create template dataset
    template = xr.Dataset(
        {
            IMERG_VAR: (
                ("time", "lat", "lon"),
                da.zeros(shape, chunks=shape, dtype=np.float32),
                {
                    "long_name": "Daily accumulated precipitation (Early)",
                    "units": "mm/day",
                    "source": "GPM IMERG Daily Early v07",
                },
            ),
        },
        coords={
            "time": time_coords,
            "lat": ("lat", lat_ea.astype(np.float64), {"units": "degrees_north"}),
            "lon": ("lon", lon_ea.astype(np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "title": "IMERG Daily Early — East Africa Subset",
            "source": f"{IMERG_SHORT_NAME} v{IMERG_VERSION}",
            "region": "East Africa",
            "bbox": f"[{EA_LON_MIN}, {EA_LAT_MIN}, {EA_LON_MAX}, {EA_LAT_MAX}]",
        },
    )
    logger.info(f"  Template:\n{template}")

    # Set up Icechunk store
    if args.local:
        logger.info(f"Using local storage: {args.local}")
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        logger.info(f"Using HuggingFace: {HF_REPO}/{HF_STORE_PREFIX}")
        storage = get_hf_icechunk_storage()

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    # Write metadata only (compute=False)
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
    session.commit("initialize IMERG Daily EA template")
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("INIT COMPLETE")
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Chunks: {chunks}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)

    return {"status": "success", "shape": shape, "chunks": chunks, "elapsed_sec": elapsed}


# --- Phase 2: fill ---


def _read_imerg_file_ea(idx, file_path):
    """Read a downloaded IMERG NetCDF file, subset to EA, return numpy."""
    import netCDF4 as nc4
    import numpy as np

    nc = nc4.Dataset(str(file_path))

    if "Grid" in nc.groups:
        grp = nc.groups["Grid"]
    else:
        grp = nc

    lat = grp.variables["lat"][:]
    lon = grp.variables["lon"][:]

    lat_idx = np.where((lat >= EA_LAT_MIN) & (lat <= EA_LAT_MAX))[0]
    lon_idx = np.where((lon >= EA_LON_MIN) & (lon <= EA_LON_MAX))[0]
    lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

    # IMERG daily: precipitation(time, lon, lat)
    precip = grp.variables["precipitation"][:, lon_sl, lat_sl]
    precip = np.transpose(precip, (0, 2, 1)).astype(np.float32)

    nc.close()
    return {"idx": idx, "data": precip}


def _write_result_to_icechunk(session, result):
    """Write a single granule result to the Icechunk session."""
    import xarray as xr

    idx = result["idx"]
    data = result["data"]
    n_t = data.shape[0]
    t_start = idx * n_t
    t_end = t_start + n_t

    ds_write = xr.Dataset({
        IMERG_VAR: (("time", "lat", "lon"), data),
    })
    ds_write.to_zarr(
        session.store,
        region={"time": slice(t_start, t_end)},
        consolidated=False,
    )


def fill_ea_store(args):
    """Fill EA template with IMERG data.

    Downloads granules via earthaccess, reads with netcdf4-python, subsets
    to EA, writes to Icechunk. Uses Coiled cluster unless --no-cluster.
    """
    import earthaccess
    import icechunk
    import tempfile
    from pathlib import Path

    logger.info("=" * 60)
    logger.info("FILL: Populating IMERG EA store (download + netcdf4 subset)")
    logger.info("=" * 60)
    overall_start = time.time()

    # Authenticate and search
    authenticate_earthdata()
    results, opendap_urls = search_imerg_daily(args.start_date, args.end_date)

    n_granules = len(results)
    if n_granules == 0:
        logger.error("No granules found!")
        return
    logger.info(f"  {n_granules} granules to process")

    # Open target Icechunk store + resume detection
    if args.local:
        target_storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        target_storage = get_hf_icechunk_storage()

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

    remaining_results = list(enumerate(results))[start_idx:]
    if not remaining_results:
        logger.info("  All granules already filled!")
        return {"status": "success", "message": "already complete"}
    logger.info(f"  Remaining: {len(remaining_results)} granules")

    COMMIT_BATCH = args.commit_batch
    total_written = 0
    total_failed = 0
    failed_indices = []
    use_cluster = not getattr(args, "no_cluster", False)

    # Download directory
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if use_cluster:
        # --- Coiled cluster path: download locally, scatter files to workers ---
        import coiled
        import distributed

        n_workers = args.n_workers
        cluster = coiled.Cluster(
            name=f"imerg-ea-fill-{int(time.time()) % 10000}",
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

        for batch_start in range(0, len(remaining_results), COMMIT_BATCH):
            batch = remaining_results[batch_start: batch_start + COMMIT_BATCH]
            batch_idx_min = batch[0][0]
            batch_idx_max = batch[-1][0]
            logger.info(
                f"  Batch: granules {batch_idx_min}-{batch_idx_max} "
                f"({len(batch)} granules, {total_written}/{len(remaining_results)} done)"
            )

            # Download batch
            batch_results_only = [r for _, r in batch]
            downloaded = earthaccess.download(batch_results_only, str(download_dir))
            logger.info(f"    Downloaded {len(downloaded)} files")

            # Submit reads to cluster
            futures = {}
            for (idx, _), fpath in zip(batch, downloaded):
                future = client.submit(
                    _read_imerg_file_ea, idx, fpath,
                    key=f"imerg-{idx}",
                )
                futures[future] = idx

            session = target_repo.writable_session("main")
            batch_ok = 0
            batch_fail = 0

            for future in distributed.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    _write_result_to_icechunk(session, result)
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

    else:
        # --- Sequential local path (no cluster) ---
        logger.info("  Running sequentially (--no-cluster)")

        for batch_start in range(0, len(remaining_results), COMMIT_BATCH):
            batch = remaining_results[batch_start: batch_start + COMMIT_BATCH]
            batch_idx_min = batch[0][0]
            batch_idx_max = batch[-1][0]
            logger.info(
                f"  Batch: granules {batch_idx_min}-{batch_idx_max} "
                f"({len(batch)} granules, {total_written}/{len(remaining_results)} done)"
            )

            # Download batch
            batch_results_only = [r for _, r in batch]
            downloaded = earthaccess.download(batch_results_only, str(download_dir))
            logger.info(f"    Downloaded {len(downloaded)} files")

            session = target_repo.writable_session("main")
            batch_ok = 0
            batch_fail = 0

            for (idx, _), fpath in zip(batch, downloaded):
                try:
                    result = _read_imerg_file_ea(idx, fpath)
                    _write_result_to_icechunk(session, result)
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

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  Granules written: {total_written}/{n_granules}")
    logger.info(f"  Failed: {total_failed} — {failed_indices[:20]}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)

    return {
        "status": "success" if not failed_indices else "partial",
        "written": total_written,
        "total": n_granules,
        "failed": failed_indices,
        "elapsed_min": elapsed / 60,
    }


# --- Phase 3: verify ---


def verify_store(args):
    """Inspect the IMERG EA Icechunk store."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting IMERG EA store")
    logger.info("=" * 60)

    if args.local:
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        storage = get_hf_icechunk_storage()

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    if "time" in ds.dims:
        logger.info(f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]}")
    if "lat" in ds.dims:
        logger.info(f"  Lat: {float(ds.lat.values[0]):.2f} -> {float(ds.lat.values[-1]):.2f}")
    if "lon" in ds.dims:
        logger.info(f"  Lon: {float(ds.lon.values[0]):.2f} -> {float(ds.lon.values[-1]):.2f}")

    for var in ds.data_vars:
        da = ds[var]
        logger.info(f"\nVariable '{var}': dtype={da.dtype}, shape={da.shape}")

    # Spot-check
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

    # Commit history
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
        description="IMERG Daily Early EA — OPeNDAP to Icechunk on HuggingFace",
    )
    sub = parser.add_subparsers(dest="command")

    # -- init --
    p_init = sub.add_parser("init", help="Create empty EA template store")
    p_init.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    p_init.add_argument("--end-date", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    p_init.add_argument("--local", type=str, default=None,
                        help="Local filesystem path (overrides HuggingFace)")

    # -- fill --
    p_fill = sub.add_parser("fill",
                            help="Fill EA store via OPeNDAP using Coiled")
    p_fill.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    p_fill.add_argument("--end-date", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    p_fill.add_argument("--local", type=str, default=None,
                        help="Local filesystem path (overrides HuggingFace)")
    p_fill.add_argument("--n-workers", type=int, default=20)
    p_fill.add_argument("--commit-batch", type=int, default=30,
                        help="Number of granules per Icechunk commit batch")
    p_fill.add_argument("--no-cluster", action="store_true",
                        help="Run sequentially without Coiled (for local testing)")
    p_fill.add_argument("--download-dir", type=str, default="./imerg_downloads",
                        help="Directory for downloaded IMERG files")

    # -- verify --
    p_verify = sub.add_parser("verify", help="Inspect store contents")
    p_verify.add_argument("--local", type=str, default=None,
                          help="Local filesystem path (overrides HuggingFace)")

    args = parser.parse_args()

    if args.command == "init":
        init_ea_store(args)
    elif args.command == "fill":
        fill_ea_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
