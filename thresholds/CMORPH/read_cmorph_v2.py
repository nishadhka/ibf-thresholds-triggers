#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "dask", "pandas", "numpy",
#   "gcsfs", "netcdf4",
# ]
# ///
"""Read cmorph-s3-nc-v2 as ONE continuous 1998-2024 series, with an optional subset.

The store keeps three groups because CMORPH re-chunked the NetCDFs twice and a
zarr array has exactly one chunking (see CMORPH_ICECHUNK_S3_NC_UPLOAD.md):

    /            (1, 501, 1506)   1998-01-01 .. 2020-06-30
    cmorph_832   (1, 832, 2497)   2020-07-01 .. 2022-08-31, 2023-06-01 .. 2023-07-31
    cmorph_825   (1, 825, 2474)   2022-09-01 .. 2023-05-31, 2023-08-01 .. 2024-12-31

That split is a *storage* constraint, not a data one: lat/lon are identical
across the groups, so on read they concat cleanly along `time` into a single
series. Differing chunk shapes just become differing dask block sizes.

The subset is applied to each group BEFORE the concat, so only chunks that
intersect the region are fetched from S3.

COST -- read this before asking for a long time range
-----------------------------------------------------
The store is chunked SPATIALLY: one chunk spans a big lat/lon tile but only
ONE 30-min step (e.g. root = `(1, 501, 1506)`). So the read cost is

    fetches ~= n_timesteps  x  n_chunks_intersecting_your_region

and it is driven by TIME, not by how small your box is. A point costs about the
same per step as a small box -- it just wastes more of each chunk.

Measured on this VM (two EA reads: 96 fetches -> 155 s, 384 -> 249 s):

    ~124 s fixed  +  ~0.33 s per chunk fetch

    EA box, 8 days        384 fetches      ~4 min     ok
    EA box, one season   4,416 fetches    ~26 min     ok, the intended use
    full record, a point 473,376 fetches  ~43 h       <-- no
    full record, EA box  867,792 fetches  ~78 h       <-- no (and ~1.1 TB out)

So: this store is the right tool for **any region over a bounded window**
(days -> a season). For **multi-year series at a point/small box**, use the
materialized, pencil-chunked EA store (`cmorph_east_africa_icechunk.py`, chunked
`(473376, 5, 5)`) -- it does a full 27-year point series in ~0.5 s. That is the
tradeoff the two stores exist to cover.

`--estimate` costs a request without fetching; reads projected over
MAX_ETA_MIN are refused unless you pass --force.

Examples
--------
    # East Africa, one season, daily totals -> NetCDF   (the intended use)
    uv run read_cmorph_v2.py --bbox -16 26 18 56 \
        --start 2024-03-01 --end 2024-05-31 --daily --out ea_mam2024.nc

    # a point, one month
    uv run read_cmorph_v2.py --point -1.29 36.82 --start 2024-03-01 --end 2024-03-31 --daily

    # what would the full record cost? (no data fetched)
    uv run read_cmorph_v2.py --bbox -16 26 18 56 --estimate
"""

import argparse
import json

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

STORE = "gs://cpc_awc/icechunk/cmorph-s3-nc-v2"
S3_PREFIX = "s3://noaa-cdr-precip-cmorph-pds/"
GROUPS = [None, "cmorph_832", "cmorph_825"]  # None = root


def open_store(store: str, sa_key: str | None):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        cfg = {}
        if sa_key:
            cfg["service_account_key"] = json.dumps(json.load(open(sa_key)))
        storage = icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"), config=cfg or None)
    else:
        storage = icechunk.local_filesystem_storage(store)
    auth = icechunk.containers_credentials(
        {S3_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    return repo.readonly_session("main").store


def to_native_lon(lon: float) -> float:
    """The store's lon axis is 0..360; accept -180..180 input too."""
    return lon + 360 if lon < 0 else lon


def _subset_groups(store: str, sa_key: str | None,
                   bbox: tuple[float, float, float, float] | None,
                   point: tuple[float, float] | None,
                   start: str | None, end: str | None) -> list[xr.Dataset]:
    """Each group, subset. Lazy -- no concat, so no dask graph is built yet."""
    zstore = open_store(store, sa_key)

    parts = []
    for group in GROUPS:
        ds = xr.open_zarr(zstore, group=group, consolidated=False)

        if bbox is not None:
            lat0, lat1, lon0, lon1 = bbox
            ds = ds.sel(lat=slice(lat0, lat1),
                        lon=slice(to_native_lon(lon0), to_native_lon(lon1)))
        elif point is not None:
            ds = ds.sel(lat=point[0], lon=to_native_lon(point[1]), method="nearest")

        if start or end:
            ds = ds.sel(time=slice(start, end))

        if ds.sizes.get("time", 0):        # group may fall entirely outside --start/--end
            parts.append(ds)

    if not parts:
        raise SystemExit("no data in the requested time range")
    return parts


def open_cmorph(store: str = STORE, sa_key: str | None = None,
                bbox: tuple[float, float, float, float] | None = None,
                point: tuple[float, float] | None = None,
                start: str | None = None, end: str | None = None) -> xr.Dataset:
    """All three chunk-regime groups, subset then concatenated into one series.

    NB: concatenating the *full* record builds a dask graph with one task per
    30-min step per chunk (~1.9M for an EA box) -- slow to even construct. Use
    `estimate_cost()` first; keep the time range bounded.
    """
    parts = _subset_groups(store, sa_key, bbox, point, start, end)

    out = xr.concat(parts, dim="time", coords="minimal", compat="override")
    out = out.sortby("time")               # groups interleave in wall-clock time

    t = pd.DatetimeIndex(out.time.values)
    if not t.is_monotonic_increasing or t.duplicated().any():
        raise SystemExit(f"BUG: non-monotonic or duplicated time "
                         f"(dups={t.duplicated().sum()})")
    return out


# Measured on this VM against GCS/S3: two materialising reads of the same EA box,
#   96 fetches -> 155 s, 384 fetches -> 249 s  =>  ~124 s fixed + ~0.33 s per fetch.
# The fixed part is env + store open + reading the time coords; the rest is chunk I/O.
BASELINE_S = 124.0
SEC_PER_FETCH = 0.33
MAX_ETA_MIN = 30.0        # refuse beyond this without --force


def estimate_cost(parts: list[xr.Dataset], daily: bool = False) -> dict:
    """Projected read cost, from METADATA ONLY -- no concat, no dask graph.

    The store is chunked spatially (one chunk = a big lat/lon tile x ONE 30-min
    step), so fetches ~= n_timesteps x n_chunks_intersecting_region: the cost is
    driven by the LENGTH OF THE TIME RANGE, not by how small the box is.
    """
    fetches = timesteps = 0
    scan_values = 0
    days = set()
    space = 1
    for ds in parts:
        n_t = ds.sizes["time"]
        chunks = ds.cmorph.encoding.get("chunks")     # zarr chunk shape, per group
        per_step = 1
        for dim in ("lat", "lon"):
            if dim in ds.cmorph.dims and chunks:
                axis = ds.cmorph.dims.index(dim)
                per_step *= int(np.ceil(ds.sizes[dim] / chunks[axis]))
        timesteps += n_t
        fetches += n_t * per_step
        scan_values += int(np.prod([ds.sizes[d] for d in ds.cmorph.dims]))
        space = int(np.prod([ds.sizes[d] for d in ("lat", "lon") if d in ds.dims]))
        days.update(pd.DatetimeIndex(ds.time.values).normalize().unique())

    # what actually lands on disk: --daily collapses 48 steps/day into 1
    out_values = len(days) * space if daily else scan_values
    return {"timesteps": timesteps, "fetches": fetches,
            "eta_min": (BASELINE_S + SEC_PER_FETCH * fetches) / 60,
            "scan_gb": scan_values * 8 / 1e9,
            "out_gb": out_values * 8 / 1e9}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--sa-key", default="coiled-data-e4drr_202505.json")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="e.g. East Africa: -16 26 18 56")
    ap.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="nearest-gridpoint time series, e.g. -1.29 36.82")
    ap.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--daily", action="store_true",
                    help="resample the 30-min rate (mm/hr) to daily totals (mm/day)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the projected read cost and exit (fetches nothing)")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the request is projected to be very expensive")
    ap.add_argument("--out", help="write NetCDF here (otherwise just summarise)")
    args = ap.parse_args()

    if args.bbox and args.point:
        ap.error("--bbox and --point are mutually exclusive")

    bbox = tuple(args.bbox) if args.bbox else None
    point = tuple(args.point) if args.point else None

    # cost first, from metadata only -- concat of a long range is itself slow to build
    parts = _subset_groups(args.store, args.sa_key, bbox, point, args.start, args.end)
    c = estimate_cost(parts, daily=args.daily)
    if args.estimate:
        print(f"timesteps    : {c['timesteps']:,}")
        print(f"S3 fetches   : {c['fetches']:,}")
        print(f"rough ETA    : {c['eta_min']:.0f} min "
              f"(~{BASELINE_S:.0f}s fixed + {SEC_PER_FETCH}s/fetch, measured)")
        print(f"scanned      : {c['scan_gb']:.1f} GB")
        print(f"written      : {c['out_gb']:.1f} GB" + (" (--daily)" if args.daily else ""))
        return

    # the fetch count is what kills you; size is just a heads-up
    if c["eta_min"] > MAX_ETA_MIN and not args.force:
        raise SystemExit(
            f"REFUSING: ~{c['fetches']:,} S3 fetches -> rough ETA ~{c['eta_min'] / 60:.1f} h.\n"
            f"  This store is chunked SPATIALLY, so cost scales with the TIME range\n"
            f"  (~{SEC_PER_FETCH}s per 30-min step, per intersecting chunk).\n"
            f"  For multi-year series use the pencil-chunked EA store instead\n"
            f"  (cmorph_east_africa_icechunk.py -- full 27-yr point series in ~0.5 s).\n"
            f"  Narrow --start/--end, --estimate to cost it, or --force to proceed.")
    if args.out and c["out_gb"] > 5:
        print(f"NOTE: this will write ~{c['out_gb']:.1f} GB to {args.out}")

    ds = open_cmorph(args.store, args.sa_key, bbox, point, args.start, args.end)

    if args.daily:
        # cmorph is a rate in mm/hr on 30-min steps -> each step contributes 0.5 h
        ds = (ds.cmorph * 0.5).resample(time="1D").sum().to_dataset(name="precip")
        ds["precip"].attrs.update(units="mm/day", long_name="CMORPH daily precipitation")

    t = pd.DatetimeIndex(ds.time.values)
    var = "precip" if args.daily else "cmorph"
    print(f"{var}{tuple(ds[var].shape)}  dims={dict(ds[var].sizes)}")
    print(f"  time : {t.min()} .. {t.max()}  n={len(t):,}  monotonic={t.is_monotonic_increasing}")
    if "lat" in ds.dims:
        print(f"  space: lat {float(ds.lat.min()):.2f}..{float(ds.lat.max()):.2f} "
              f"({ds.sizes['lat']})  lon {float(ds.lon.min()):.2f}..{float(ds.lon.max()):.2f} "
              f"({ds.sizes['lon']})")

    if args.out:
        print(f"  materialising -> {args.out} ...")
        ds.to_netcdf(args.out)
        v = ds[var].values
        print(f"  wrote {args.out}: finite={100 * np.isfinite(v).mean():.1f}% "
              f"max={np.nanmax(v):.1f} {ds[var].attrs.get('units', '')}")


if __name__ == "__main__":
    main()
