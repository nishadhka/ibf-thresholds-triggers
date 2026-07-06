# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "virtualizarr>=2.7", "obstore", "pandas", "pyarrow", "gcsfs",
# ]
# ///
"""CMORPH Parquet VDS catalog -> Icechunk virtual store (BATCH builder).

The CMORPH counterpart of grib-index-kerchunk/gefs/build_gefs_icechunk.py, and
the Coiled-free replacement for cmorph_s3_to_gcs_icechunk_parallel.py.

The CMORPH "par" is a single Parquet VDS catalog on GCS
(gs://cpc_awc/cmorph_catalog/catalog.parquet): one row per 30-min NetCDF file,
column `kerchunk_refs` (a `{"version":1,"refs":{...}}` JSON string) already holds
the virtual references -- nothing to virtualize from S3, no Coiled needed.

This builder appends a *range* of days in ONE commit. Committing per ~60-day
manifest shard (instead of per day) is what makes the backfill fast: each shard
is rewritten once, not once per day as it fills. Still a single sequential
writer -> zero ConflictError/429s. Reconstruction of the batch's files is
threaded. Reuses the #884 OOM fix: manifest splitting along `time`, virtual
refs only (no data movement) -> RAM holds one batch of refs (~a few hundred MB).

Usage (one shard):
  export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json
  uv run build_cmorph_icechunk.py --start 19990919 --end 19991117 \
      --catalog gs://cpc_awc/cmorph_catalog/catalog.parquet \
      --store gs://cpc_awc/icechunk/cmorph-s3-nc
  # single day: --date 20200101   (alias for --start=--end=date)
"""
import argparse
import json
import os
import tempfile
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr
import icechunk
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.compute as pc

S3_PREFIX = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
STANDARD_GRID = (1649, 4948)
DROP_VARS = ["lat_bounds", "lon_bounds", "time_bounds", "nv"]
# manifest shard = 2880 timesteps = 60 days; appends touch only the last shard
MANIFEST_SPLIT_TIME = 2880


def resolve_storage(store: str, sa_key: str | None):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        return icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    return icechunk.local_filesystem_storage(store)


def sanitize_refs(refs: dict) -> dict:
    """Kerchunk stores a float _FillValue on the float64 lat/lon coords, which
    xarray's zarr backend refuses to decode -- drop those entries."""
    for key, val in refs["refs"].items():
        if key.endswith(".zattrs"):
            attrs = json.loads(val)
            if isinstance(attrs.get("_FillValue"), float):
                del attrs["_FillValue"]
                refs["refs"][key] = json.dumps(attrs)
    return refs


def reconstruct_vds(refs: dict) -> xr.Dataset:
    """One stored kerchunk dict -> virtual dataset (time loaded for concat order)."""
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
        vds = open_virtual_dataset(
            url=f"file://{tmp}", parser=KerchunkJSONParser(),
            registry=registry, loadable_variables=["time"])
        return vds.drop_vars([v for v in DROP_VARS if v in vds.variables],
                             errors="ignore")
    finally:
        os.unlink(tmp)


def open_or_create_repo(storage):
    from icechunk import (ManifestSplitCondition, ManifestSplitDimCondition,
                          ManifestSplittingConfig)
    auth = icechunk.containers_credentials(
        {S3_PREFIX: icechunk.s3_anonymous_credentials()})
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(
            storage, authorize_virtual_chunk_access=auth), False
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        S3_PREFIX, icechunk.s3_store(region=S3_REGION, anonymous=True)))
    config.manifest = icechunk.ManifestConfig(
        splitting=ManifestSplittingConfig.from_dict({
            ManifestSplitCondition.AnyArray(): {
                ManifestSplitDimCondition.DimensionName("time"): MANIFEST_SPLIT_TIME}}))
    repo = icechunk.Repository.create(
        storage, config, authorize_virtual_chunk_access=auth)
    return repo, True


def last_done_time(repo) -> pd.Timestamp | None:
    try:
        ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        return pd.Timestamp(ds.time.values[-1]) if ds.sizes.get("time") else None
    except Exception:
        return None


def read_range_refs(catalog: str, sa_key: str | None, start: str, end: str,
                    after: pd.Timestamp | None) -> pd.DataFrame:
    """All catalog rows in [start,end] (chronological, success only, > after)."""
    fs = None
    path = catalog
    if catalog.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(
            token=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        path = catalog[5:]
    lo = datetime(int(start[:4]), int(start[4:6]), int(start[6:]))
    hi = datetime(int(end[:4]), int(end[4:6]), int(end[6:])) + timedelta(days=1)
    if after is not None and after.to_pydatetime() >= lo:
        lo = after.to_pydatetime() + timedelta(microseconds=1)
    flt = ((pc.field("datetime") >= pa.scalar(lo, pa.timestamp("us")))
           & (pc.field("datetime") < pa.scalar(hi, pa.timestamp("us")))
           & (pc.field("status") == "success"))
    tbl = pads.dataset(path, filesystem=fs, format="parquet").to_table(
        columns=["datetime", "kerchunk_refs"], filter=flt)
    return tbl.to_pandas().sort_values("datetime").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYYMMDD (inclusive)")
    ap.add_argument("--end", help="YYYYMMDD (inclusive)")
    ap.add_argument("--date", help="single day (alias for --start=--end=date)")
    ap.add_argument("--catalog", default="gs://cpc_awc/cmorph_catalog/catalog.parquet")
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    start = args.start or args.date
    end = args.end or args.date
    if not (start and end):
        ap.error("need --start/--end or --date")
    t0 = _time.time()

    storage = resolve_storage(args.store, args.sa_key)
    repo, created = open_or_create_repo(storage)
    last = last_done_time(repo)

    df = read_range_refs(args.catalog, args.sa_key, start, end, last)
    if df.empty:
        raise SystemExit(f"{start}..{end}: nothing to do (after {last})")

    refs_list = [json.loads(s) for s in df.kerchunk_refs]
    with ThreadPoolExecutor(args.threads) as ex:
        vdss = list(ex.map(reconstruct_vds, refs_list))
    good, skipped = [], []
    for v in vdss:
        (good if tuple(v["cmorph"].shape[1:]) == STANDARD_GRID else skipped).append(v)
    if not good:
        raise SystemExit(f"{start}..{end}: 0 usable files")

    batch_vds = xr.concat(good, dim="time", coords="minimal", compat="override")
    session = repo.writable_session("main")
    batch_vds.virtualize.to_icechunk(
        session.store, append_dim=None if last is None else "time")
    d0 = pd.Timestamp(df.datetime.iloc[0]).strftime("%Y%m%d")
    d1 = pd.Timestamp(df.datetime.iloc[-1]).strftime("%Y%m%d")
    snap = session.commit(
        f"{d0}..{d1}: {len(good)} files, {batch_vds.sizes['time']} timesteps")
    print(f"{d0}..{d1}: committed {len(good)} files / {batch_vds.sizes['time']} steps "
          f"-> {str(snap)[:12]} ({_time.time()-t0:.1f}s)"
          + (f" | skipped {len(skipped)} off-grid" if skipped else ""))


if __name__ == "__main__":
    main()
