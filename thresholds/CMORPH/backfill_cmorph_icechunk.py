# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "virtualizarr>=2.7", "obstore", "pandas", "pyarrow", "gcsfs",
# ]
# ///
"""Backfill the CMORPH Icechunk store from the GCS Parquet VDS catalog -- the
CMORPH counterpart of grib-index-kerchunk/gefs/backfill_gefs_icechunk.py.

  gs://cpc_awc/cmorph_catalog/catalog.parquet   (236,688 rows, 9,862 days,
      1998..2024, one row per 30-min NetCDF, `kerchunk_refs` already computed)
    -> build_cmorph_icechunk.py per day -> commit one day (48 half-hour steps)

Single sequential writer (no Coiled, no branches) -> zero ConflictError/429s.
Resumable: days already in the store's `time` axis are skipped; appends stay
chronological. The 234 MB catalog is downloaded once to --work-dir so each
per-day build reads it locally (fast pyarrow row-filter), not re-fetched 9,862x.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json
  uv run backfill_cmorph_icechunk.py --store gs://cpc_awc/icechunk/cmorph-s3-nc
  # options: --start 20200101 --end 20201231 --limit 5 --dry-run
  # full run: nohup uv run backfill_cmorph_icechunk.py \
  #     --store gs://cpc_awc/icechunk/cmorph-s3-nc > /dev/null 2>&1 &
  #     (tail -f backfill_cmorph.log)
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

CATALOG = "gs://cpc_awc/cmorph_catalog/catalog.parquet"
BUILDER = Path(__file__).parent / "build_cmorph_icechunk.py"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(Path(__file__).parent / "backfill_cmorph.log"),
              logging.StreamHandler()])
log = logging.getLogger("backfill_cmorph")


def fetch_catalog(catalog: str, sa_key: str | None, work_dir: Path) -> str:
    """Download the GCS catalog once to a local file (fast repeated filtering)."""
    if not catalog.startswith("gs://"):
        return catalog
    import gcsfs
    fs = gcsfs.GCSFileSystem(
        token=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / "cmorph_catalog.parquet"
    if not local.exists():
        log.info(f"Downloading catalog {catalog} -> {local}")
        fs.get(catalog[5:], str(local))
    return str(local)


def all_days(local_catalog: str) -> list[str]:
    """Distinct YYYYMMDD in the catalog, chronological (success rows only)."""
    tbl = pads.dataset(local_catalog, format="parquet").to_table(
        columns=["year", "month", "day", "status"])
    df = tbl.to_pandas()
    df = df[df.status == "success"].drop_duplicates(["year", "month", "day"])
    days = pd.to_datetime(df[["year", "month", "day"]]).sort_values()
    return [d.strftime("%Y%m%d") for d in days]


def dates_done(store: str, sa_key: str | None) -> set[str]:
    import icechunk
    import xarray as xr
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        storage = icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    else:
        storage = icechunk.local_filesystem_storage(store)
    if not icechunk.Repository.exists(storage):
        return set()
    auth = icechunk.containers_credentials(
        {"s3://noaa-cdr-precip-cmorph-pds/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    try:
        ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        return {pd.Timestamp(t).strftime("%Y%m%d")
                for t in pd.DatetimeIndex(ds.time.values).normalize().unique()}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--start", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None, help="max BATCHES this run")
    ap.add_argument("--batch-days", type=int, default=60,
                    help="days appended per commit (~1 manifest shard = 60)")
    ap.add_argument("--work-dir", default="/tmp/cmorph_backfill")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    local_cat = fetch_catalog(args.catalog, args.sa_key, Path(args.work_dir))
    days = all_days(local_cat)
    if args.start:
        days = [d for d in days if d >= args.start]
    if args.end:
        days = [d for d in days if d <= args.end]

    done = dates_done(args.store, args.sa_key)
    todo = [d for d in days if d not in done]
    # group the contiguous todo list into batches of --batch-days (one commit each)
    batches = [todo[i:i + args.batch_days] for i in range(0, len(todo), args.batch_days)]
    log.info(f"catalog: {len(days)} days in scope | in store: {len(days)-len(todo)} "
             f"| to build: {len(todo)} days in {len(batches)} batch(es) of "
             f"{args.batch_days}" + (f" (limited to {args.limit})" if args.limit else ""))
    if args.limit:
        batches = batches[:args.limit]
    if args.dry_run:
        if batches:
            log.info(f"first batch: {batches[0][0]}..{batches[0][-1]}  "
                     f"last batch: {batches[-1][0]}..{batches[-1][-1]}")
        return

    t_start, times, fails = time.time(), [], []
    for i, batch in enumerate(batches, 1):
        t0 = time.time()
        cmd = [sys.executable, str(BUILDER), "--start", batch[0], "--end", batch[-1],
               "--catalog", local_cat, "--store", args.store]
        if args.sa_key:
            cmd += ["--sa-key", args.sa_key]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"{batch[0]}..{batch[-1]}")
            log.error(f"[{i}/{len(batches)}] {batch[0]}..{batch[-1]}: FAILED -- "
                      f"{(r.stdout[-400:] + r.stderr[-400:]).strip()}")
            continue
        dt = time.time() - t0
        times.append(dt)
        eta_h = (len(batches) - i) * (sum(times[-10:]) / len(times[-10:])) / 3600
        log.info(f"[{i}/{len(batches)}] {r.stdout.strip().splitlines()[-1]} "
                 f"({dt:.0f}s, ETA {eta_h:.1f} h)")

    log.info(f"done: {len(times)} batches built, {len(fails)} failed in "
             f"{(time.time()-t_start)/3600:.2f} h")
    if fails:
        log.info(f"failed batches (re-run to retry): {fails}")


if __name__ == "__main__":
    main()
