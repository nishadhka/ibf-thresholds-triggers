# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "virtualizarr>=2.7", "obstore", "pandas", "pyarrow", "gcsfs",
# ]
# ///
"""CMORPH Parquet VDS catalog -> Icechunk virtual store (per-DAY builder).

The CMORPH counterpart of grib-index-kerchunk/gefs/build_gefs_icechunk.py, and
the Coiled-free replacement for cmorph_s3_to_gcs_icechunk_parallel.py.

The CMORPH "par" is a single Parquet VDS catalog on GCS
(gs://cpc_awc/cmorph_catalog/catalog.parquet): one row per 30-min NetCDF file,
column `kerchunk_refs` (a `{"version":1,"refs":{...}}` JSON string) already holds
the virtual references. So there is NOTHING to virtualize from S3 and NO Coiled
cluster is needed -- we read the day's rows, reconstruct virtual datasets from
the stored refs, and append one day (48 half-hour steps) to the Icechunk store.

Reuses the proven #884 OOM fix (cmorph_daily_commit_icechunk.py): one commit per
day, manifest splitting along `time` (60-day shards) so each append is O(1),
virtual chunk container over the anonymous NOAA S3 bucket (no data movement).

Usage (single day):
  export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json
  uv run build_cmorph_icechunk.py --date 20200101 \
      --catalog gs://cpc_awc/cmorph_catalog/catalog.parquet \
      --store gs://cpc_awc/icechunk/cmorph-s3-nc
  # local catalog + local store (smoke test):
  uv run build_cmorph_icechunk.py --date 20200101 \
      --catalog /tmp/cmorph_catalog.parquet --store /tmp/cmorph_store
"""
import argparse
import json
import os
import tempfile
import time as _time

import pandas as pd
import xarray as xr
import icechunk
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


def last_done_date(repo) -> pd.Timestamp | None:
    try:
        ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        return pd.Timestamp(ds.time.values[-1]) if ds.sizes.get("time") else None
    except Exception:
        return None


def read_day_refs(catalog: str, sa_key: str | None, date: str) -> list[dict]:
    """Read one day's rows from the Parquet VDS catalog, chronological."""
    y, m, d = int(date[:4]), int(date[4:6]), int(date[6:])
    fs = None
    if catalog.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(
            token=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        catalog = catalog[5:]
    dset = pads.dataset(catalog, filesystem=fs, format="parquet")
    flt = (pc.field("year") == y) & (pc.field("month") == m) & (pc.field("day") == d)
    tbl = dset.to_table(columns=["datetime", "status", "kerchunk_refs"], filter=flt)
    df = tbl.to_pandas().sort_values("datetime")
    df = df[df.status == "success"]
    return [json.loads(s) for s in df.kerchunk_refs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--catalog", default="gs://cpc_awc/cmorph_catalog/catalog.parquet")
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    args = ap.parse_args()
    t0 = _time.time()

    storage = resolve_storage(args.store, args.sa_key)
    repo, created = open_or_create_repo(storage)
    last = last_done_date(repo)
    day = pd.Timestamp(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}")
    if last is not None and day.date() <= last.date():
        raise SystemExit(f"{args.date} already in store (through {last.date()})")

    refs_list = read_day_refs(args.catalog, args.sa_key, args.date)
    if not refs_list:
        raise SystemExit(f"{args.date}: no rows in catalog")

    vdss, skipped = [], []
    for refs in refs_list:
        vds = reconstruct_vds(refs)
        if tuple(vds["cmorph"].shape[1:]) != STANDARD_GRID:
            skipped.append(tuple(vds["cmorph"].shape))
            continue
        vdss.append(vds)
    if not vdss:
        raise SystemExit(f"{args.date}: 0 usable files (skipped {skipped})")

    day_vds = xr.concat(vdss, dim="time", coords="minimal", compat="override")
    session = repo.writable_session("main")
    day_vds.virtualize.to_icechunk(
        session.store, append_dim=None if last is None else "time")
    snap = session.commit(
        f"{args.date}: {len(vdss)} files, {day_vds.sizes['time']} timesteps")
    print(f"{args.date}: committed {len(vdss)} files / {day_vds.sizes['time']} steps "
          f"-> {str(snap)[:12]} ({_time.time()-t0:.1f}s)"
          + (f" | skipped grids {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
