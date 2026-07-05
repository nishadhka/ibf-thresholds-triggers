#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#     "virtualizarr>=2.7", "obstore", "obspec-utils",
#     "h5py", "hdf5plugin", "h5netcdf", "fsspec", "s3fs", "numpy",
# ]
# ///
"""
CMORPH S3 -> Icechunk virtual store: DAILY-COMMIT rewrite (the #884 fix)
=========================================================================

Corrects the three failed strategies of cmorph_s3_to_gcs_icechunk_parallel.py
(see VirtualiZarr discussion #884 and ICECHUNK_PARALLEL_ERRORS_ANALYSIS.md):

  batch mode   -- per-worker branches needed merging; GCS 429s on refs
  collect mode -- coordinator accumulated ~82 KB kerchunk JSON x 236K files
                  -> OOM (exit 137) at ~28K files
  direct mode  -- 20 workers committing one branch -> 70-80% ConflictError

The fix (pattern proven in grib-index-kerchunk/ecmwf/icechunk-par):

  1. SINGLE writer. Workers only virtualize files and return kerchunk dicts;
     they never touch the Icechunk repo. No branches, no merges, no conflicts,
     no GCS rate limits, and (until publish) no cloud credentials at all.
  2. TEMP LOCAL store, ONE DAY per commit. The coordinator loops day by day:
     collect the day's 24 file refs (~2 MB), reconstruct, xr.concat,
     to_icechunk(append_dim="time"), commit, free. Coordinator RAM is one
     day's refs regardless of archive size (measured ~260 MB flat).
  3. MANIFEST SPLITTING along time (Tom Nicholas' #884 suggestion): each
     commit rewrites only the last ~60-day manifest shard, not one
     ever-growing manifest -- this is what makes 9,862 sequential appends
     O(1) each instead of O(n).
  4. RESUME for free: on restart the loop reads the store's last timestep
     and skips completed days (also removes any STS credential-window worry).
  5. PUBLISH at the end by plain file sync (Icechunk's local layout is
     byte-identical to its object-store layout):
         gsutil -m rsync -r <local-store> gs://<bucket>/<prefix>

Validated on 2020-01-01..02 (48 files): 4-6 s/day/commit sequential-local,
coordinator peak RSS 266 MB, stored int16 bit-exact vs direct S3 netCDF reads.

Usage:
    # local backend (no cluster; threads virtualize the day's files)
    uv run cmorph_daily_commit_icechunk.py \
        --start-date 2020-01-01 --end-date 2020-12-31 \
        --store ./cmorph_icechunk_local --backend local

    # Coiled backend (workers virtualize; coordinator still commits locally)
    uv run cmorph_daily_commit_icechunk.py \
        --start-date 1998-01-01 --end-date 2024-12-31 \
        --store ./cmorph_icechunk_local --backend coiled --n-workers 20
"""

import argparse
import json
import logging
import os
import resource
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import icechunk

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

S3_PREFIX = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
STANDARD_GRID = (1649, 4948)
DROP_VARS = ["lat_bounds", "lon_bounds", "time_bounds", "nv"]
# manifest shard = 2880 timesteps = 60 days; appends touch only the last shard
MANIFEST_SPLIT_TIME = 2880


# ── worker: one file -> kerchunk dict (Coiled-serializable, ~82 KB) ──────────

def virtualize_file_worker(s3_url: str) -> dict:
    """Runs on a (Coiled or local) worker. Never touches the Icechunk repo."""
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import from_url

    try:
        store = from_url(S3_PREFIX, region=S3_REGION, skip_signature=True)
        registry = ObjectStoreRegistry({S3_PREFIX: store})
        vds = open_virtual_dataset(url=s3_url, parser=HDFParser(),
                                   registry=registry, drop_variables=DROP_VARS)
        shape = tuple(vds["cmorph"].shape)
        refs = vds.virtualize.to_kerchunk(format="dict")
        vds.close()
        return {"s3_url": s3_url, "status": "success", "shape": shape, "refs": refs}
    except Exception as e:
        return {"s3_url": s3_url, "status": "error", "error": str(e)}


# ── coordinator helpers ──────────────────────────────────────────────────────

def sanitize_refs(refs: dict) -> dict:
    """Kerchunk stores float _FillValue on the float64 lat/lon coords, which
    xarray's zarr backend refuses to decode -- drop those entries."""
    for key, val in refs["refs"].items():
        if key.endswith(".zattrs"):
            attrs = json.loads(val)
            if isinstance(attrs.get("_FillValue"), float):
                del attrs["_FillValue"]
                refs["refs"][key] = json.dumps(attrs)
    return refs


def reconstruct_vds(refs: dict) -> xr.Dataset:
    """kerchunk dict -> virtual dataset (time loaded for concat/append order)."""
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import KerchunkJSONParser
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import from_url, LocalStore

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sanitize_refs(refs), f)
        tmp = f.name
    try:
        registry = ObjectStoreRegistry({
            S3_PREFIX: from_url(S3_PREFIX, region=S3_REGION, skip_signature=True),
            "file://": LocalStore(),
        })
        return open_virtual_dataset(url=f"file://{tmp}", parser=KerchunkJSONParser(),
                                    registry=registry, loadable_variables=["time"])
    finally:
        os.unlink(tmp)


def open_or_create_repo(store_path: str) -> tuple[icechunk.Repository, bool]:
    from icechunk import (ManifestSplitCondition, ManifestSplitDimCondition,
                          ManifestSplittingConfig)
    storage = icechunk.local_filesystem_storage(store_path)
    auth = icechunk.containers_credentials(
        {S3_PREFIX: icechunk.s3_anonymous_credentials()})
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth), False
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        S3_PREFIX, icechunk.s3_store(region=S3_REGION, anonymous=True)))
    config.manifest = icechunk.ManifestConfig(
        splitting=ManifestSplittingConfig.from_dict({
            ManifestSplitCondition.AnyArray(): {
                ManifestSplitDimCondition.DimensionName("time"): MANIFEST_SPLIT_TIME}}))
    repo = icechunk.Repository.create(storage, config,
                                      authorize_virtual_chunk_access=auth)
    return repo, True


def last_done_date(repo) -> pd.Timestamp | None:
    """Resume support: last timestep already committed, or None if empty."""
    try:
        ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        return pd.Timestamp(ds.time.values[-1]) if ds.sizes.get("time") else None
    except Exception:
        return None


def day_files(fs, day: pd.Timestamp) -> list[str]:
    pat = (f"noaa-cdr-precip-cmorph-pds/data/30min/8km/"
           f"{day.year}/{day.month:02d}/{day.day:02d}/*.nc")
    return sorted(f"s3://{f}" for f in fs.glob(pat))


# ── main loop: one day, one commit ───────────────────────────────────────────

def run(args):
    fs = fsspec.filesystem("s3", anon=True)
    repo, created = open_or_create_repo(args.store)
    store_empty = created or last_done_date(repo) is None

    resume_after = last_done_date(repo)
    if resume_after is not None:
        logger.info(f"Resuming: store already contains data through {resume_after}")

    client = None
    if args.backend == "coiled":
        import coiled
        from dask.distributed import Client
        cluster = coiled.Cluster(
            name=f"cmorph-daily-{int(time.time()) % 10000}",
            n_workers=args.n_workers, worker_vm_types="n2-standard-2",
            package_sync=True, region="us-east1", workspace="e4drr",
            idle_timeout="20 minutes")
        client = Client(cluster)
        logger.info(f"Coiled cluster ready: {client.dashboard_link}")

    days = pd.date_range(args.start_date, args.end_date, freq="D")
    skipped_files: list[dict] = []
    done = 0
    try:
        for day in days:
            if resume_after is not None and day.date() <= resume_after.date():
                continue
            urls = day_files(fs, day)
            if not urls:
                logger.warning(f"{day.date()}: no files on S3, skipping")
                continue

            t0 = time.time()
            if client is not None:
                results = client.gather(client.map(virtualize_file_worker, urls))
            else:
                with ThreadPoolExecutor(args.local_threads) as ex:
                    results = list(ex.map(virtualize_file_worker, urls))

            good, day_skipped = [], []
            for r in results:
                if r["status"] != "success":
                    day_skipped.append(r)
                elif r["shape"][1:] != STANDARD_GRID:
                    day_skipped.append({**r, "error": f"non-standard grid {r['shape']}"})
                else:
                    good.append(r)
            skipped_files.extend(
                {k: v for k, v in s.items() if k != "refs"} for s in day_skipped)
            if not good:
                logger.error(f"{day.date()}: 0/{len(urls)} usable files, skipping day")
                continue

            vdss = [reconstruct_vds(r["refs"]) for r in good]
            day_vds = xr.concat(vdss, dim="time", coords="minimal", compat="override")
            session = repo.writable_session("main")
            day_vds.virtualize.to_icechunk(
                session.store, append_dim=None if store_empty else "time")
            snap = session.commit(
                f"{day.date()}: {len(good)} files, {day_vds.sizes['time']} timesteps")
            store_empty = False
            del results, good, vdss, day_vds

            done += 1
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            logger.info(f"{day.date()}: committed {snap[:12]}.. "
                        f"({time.time()-t0:.1f}s, coordinator peak RSS {rss:.0f} MB)")
    finally:
        if client is not None:
            client.close()
            client.cluster.close()
        if skipped_files:
            with open("cmorph_skipped_files.json", "w") as f:
                json.dump(skipped_files, f, indent=2)
            logger.warning(f"{len(skipped_files)} files skipped "
                           f"-> cmorph_skipped_files.json")

    logger.info(f"Done: {done} day-commits into {args.store}")
    logger.info("Publish when finished (layout is identical on object storage):")
    logger.info(f"  gsutil -m rsync -r {args.store} gs://<bucket>/<prefix>")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--store", default="./cmorph_icechunk_local",
                   help="LOCAL temp Icechunk store (publish by file sync afterwards)")
    p.add_argument("--backend", choices=["local", "coiled"], default="local")
    p.add_argument("--n-workers", type=int, default=20, help="Coiled workers")
    p.add_argument("--local-threads", type=int, default=8,
                   help="virtualization threads for --backend local")
    run(p.parse_args())


if __name__ == "__main__":
    main()
