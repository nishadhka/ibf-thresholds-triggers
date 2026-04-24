#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netcdf4",
#     "icechunk>=0.1",
#     "zarr>=3",
#     "python-dotenv",
# ]
# ///
"""
Upload CMORPH East Africa Return Period NetCDF → Icechunk store on source.coop
==============================================================================

Reads the local CMORPH RP NetCDF (cmorph_ea_return_periods.nc) produced by
cmorph_return_periods.py and writes it as a pan-chunk Icechunk store to
source.coop at:

    s3://us-west-2.opendata.source.coop/
        e4drr-project/observations/cmorph_rp_icechunk

Pan chunk layout (one duration slab × full spatial extent):
    return_period_precip  (duration, return_period, lat, lon) → (1, 6, 550, 474)
    dist_location         (duration, lat, lon)                → (1, 550, 474)
    dist_scale            (duration, lat, lon)                → (1, 550, 474)
    annual_maxima         (duration, year, lat, lon)          → (1, 1, 550, 474)

Credentials are read from .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_SESSION_TOKEN).  The store is then readable anonymously via:

    storage = icechunk.s3_storage(
        bucket="us-west-2.opendata.source.coop",
        prefix="e4drr-project/observations/cmorph_rp_icechunk",
        region="us-west-2",
        anonymous=True,
    )

Usage:
    # Upload from default NC path
    uv run cmorph_rp_to_source_coop.py

    # Override NC path
    uv run cmorph_rp_to_source_coop.py \\
        --nc-path /path/to/cmorph_ea_return_periods.nc

    # Write to local Icechunk store instead (for testing)
    uv run cmorph_rp_to_source_coop.py \\
        --store-path ./cmorph_rp_icechunk_local

    # Verify uploaded store (anonymous read)
    uv run cmorph_rp_to_source_coop.py --verify
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "e4drr-project/observations/cmorph_rp_icechunk"
S3_REGION = "us-west-2"

DEFAULT_NC = Path(__file__).parent.parent.parent.parent / "bn-ibf/flood_ibf/cmorph_ea_return_periods.nc"


def upload(args):
    import icechunk
    import xarray as xr

    nc_path = Path(args.nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    logger.info("=" * 60)
    logger.info("UPLOAD: CMORPH RP NetCDF → Icechunk on source.coop")
    logger.info("=" * 60)
    start = time.time()

    # ── Load NetCDF ──
    logger.info(f"  Reading: {nc_path} ({nc_path.stat().st_size / (1024**2):.0f} MB)")
    ds = xr.open_dataset(nc_path)
    logger.info(f"  Dimensions: {dict(ds.sizes)}")
    logger.info(f"  Variables:  {list(ds.data_vars)}")

    n_dur  = ds.sizes["duration"]
    n_rp   = ds.sizes["return_period"]
    n_lat  = ds.sizes["lat"]
    n_lon  = ds.sizes["lon"]
    n_year = ds.sizes["year"]

    # ── Pan chunk encoding ──
    # One duration slab at full spatial extent per chunk.
    encoding = {
        "return_period_precip": {"chunks": (1, n_rp, n_lat, n_lon)},
        "dist_location":        {"chunks": (1, n_lat, n_lon)},
        "dist_scale":           {"chunks": (1, n_lat, n_lon)},
        "annual_maxima":        {"chunks": (1, 1, n_lat, n_lon)},
    }
    for var, enc in encoding.items():
        if var in ds.data_vars:
            sz = np.prod(enc["chunks"]) * 4 / (1024**2)
            logger.info(f"  {var}: chunks={enc['chunks']}  ({sz:.1f} MB/chunk)")

    # ── Set up Icechunk storage ──
    if args.store_path:
        logger.info(f"  Target: local → {args.store_path}")
        storage = icechunk.local_filesystem_storage(path=args.store_path)
    else:
        access_key   = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("SOURCE_COOP_ACCESS_KEY_ID")
        secret_key   = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SOURCE_COOP_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN") or os.environ.get("SOURCE_COOP_SESSION_TOKEN")

        if not access_key or not secret_key:
            raise RuntimeError(
                "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN) "
                "in .env or environment for source.coop write access."
            )

        logger.info(f"  Target: s3://{S3_BUCKET}/{S3_PREFIX}")
        storage = icechunk.s3_storage(
            bucket=S3_BUCKET,
            prefix=S3_PREFIX,
            region=S3_REGION,
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=session_token,
        )

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new Icechunk repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (overwriting)")

    # ── Write ──
    logger.info("  Writing to Icechunk store...")
    session = repo.writable_session("main")
    ds.to_zarr(
        session.store,
        mode="w",
        consolidated=False,
        encoding={k: v for k, v in encoding.items() if k in ds.data_vars},
    )
    session.commit(
        f"CMORPH EA return periods — {n_dur} durations × {n_rp} RPs "
        f"× {n_lat} lat × {n_lon} lon  |  {n_year} years (1998–2024)"
    )

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info(f"  Durations:      {list(ds.duration.values)}")
    logger.info(f"  Return periods: {list(ds.return_period.values)} years")
    logger.info(f"  Grid:           {n_lat} lat × {n_lon} lon")
    logger.info(f"  Annual maxima:  {n_year} years")
    logger.info(f"  Time:           {elapsed:.1f}s")
    logger.info("=" * 60)
    ds.close()


def verify(args):
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Reading cmorph_rp_icechunk from source.coop")
    logger.info("=" * 60)

    if args.store_path:
        storage = icechunk.local_filesystem_storage(path=args.store_path)
    else:
        logger.info(f"  s3://{S3_BUCKET}/{S3_PREFIX} (anonymous)")
        storage = icechunk.s3_storage(
            bucket=S3_BUCKET,
            prefix=S3_PREFIX,
            region=S3_REGION,
            anonymous=True,
        )

    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\n{ds}")
    logger.info(f"\nDimensions:  {dict(ds.sizes)}")
    logger.info(f"Variables:   {list(ds.data_vars)}")
    logger.info(f"Durations:   {list(ds.duration.values)}")
    logger.info(f"Return periods: {list(ds.return_period.values)} years")
    logger.info(f"Years:       {list(ds.year.values[:3])} … {list(ds.year.values[-3:])}")

    # Spot check: Nairobi (~-1.3°S 36.8°E), 24hr, 10-yr RP
    logger.info("\n--- Spot check: Nairobi (~1.3°S, 36.8°E) ---")
    for dur in ["24hr", "7day"]:
        for rp in [10, 50]:
            val = float(
                ds["return_period_precip"]
                .sel(duration=dur, return_period=rp)
                .sel(lat=-1.3, lon=36.8, method="nearest")
                .values
            )
            logger.info(f"  {dur} / {rp}-yr RP: {val:.2f} mm")

    logger.info("\nVerification complete.")
    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Upload CMORPH RP NetCDF to Icechunk on source.coop"
    )
    parser.add_argument(
        "--nc-path", type=str, default=str(DEFAULT_NC),
        help=f"Path to cmorph_ea_return_periods.nc (default: {DEFAULT_NC})",
    )
    parser.add_argument(
        "--store-path", type=str, default=None,
        help="Local Icechunk path for testing. Default: writes to source.coop S3.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the uploaded store (anonymous read from source.coop).",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args)
    else:
        upload(args)


if __name__ == "__main__":
    main()
