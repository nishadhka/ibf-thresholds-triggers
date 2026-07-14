#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "gcsfs", "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "numpy"]
# ///
"""Mirror the CMORPH Icechunk store from GCS to source.coop by plain object copy.

CMORPH counterpart of grib-index-kerchunk/icechunk-dask/mirror_gcs_to_source_coop.py.

Why a plain object copy is enough
---------------------------------
`cmorph-s3-nc-v2` is a *virtual* store: its chunks are references into the public
`s3://noaa-cdr-precip-cmorph-pds/` bucket, not copies of the data. So the store
prefix itself is tiny (~163 MB / ~1.4k objects) and a byte-for-byte copy of it is
a fully-functional published repo. Nothing about the virtual references changes;
only the store metadata (repo pointer + snapshots + manifests + transactions +
native coord chunks) moves. Readers need anonymous GET/LIST on source.coop plus
anonymous GET on the NOAA bucket -- both are public.

The three chunk-regime groups (root / cmorph_832 / cmorph_825) need no special
handling: they are just nodes inside the snapshots being copied.

Correctness
-----------
The top-level `repo` file is the entry point naming the branch tips. It is
uploaded LAST, after every snapshot/manifest it can reference is already present,
so the published store is never seen pointing at a not-yet-copied object.

`overwritten/` holds superseded `repo`-file versions from ref rotation; Icechunk
never reads it on open, so it is skipped by default.

Resumable
---------
Skips destination objects already present with the same size. source.coop hands
out 1-hour STS credentials, so if the token expires mid-run: refresh .env,
`source .env`, re-run -- it picks up where it stopped. (At ~163 MB this should
finish well inside one token.)

Usage
-----
  export GOOGLE_APPLICATION_CREDENTIALS=$PWD/coiled-data-e4drr_202505.json  # GCS read
  source .env                                                               # source.coop STS
  uv run mirror_cmorph_to_source_coop.py                  # uses the defaults below
  uv run mirror_cmorph_to_source_coop.py --verify-only    # check a published store

Published as:
  https://source.coop/e4drr-project/observations/s3-noaa-cdr-cmorph-icechunk-vd
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import gcsfs
from botocore.config import Config
from botocore.exceptions import ClientError

GCS_STORE = "gs://cpc_awc/icechunk/cmorph-s3-nc-v2"
BUCKET = "e4drr-project"
DEST = "observations/s3-noaa-cdr-cmorph-icechunk-vd"
NOAA_PREFIX = "s3://noaa-cdr-precip-cmorph-pds/"
GROUPS = [None, "cmorph_832", "cmorph_825"]

# Functional icechunk prefixes, in copy order. `repo` is handled separately
# (uploaded last). `overwritten` is intentionally excluded (see module docstring).
DATA_SUBDIRS = ["manifests", "snapshots", "transactions", "chunks"]


def list_source(fs, store, subdirs):
    """(relpath, size) for every file object under the store's data subdirs."""
    out = []
    for sub in subdirs:
        p = f"{store}/{sub}"
        if not fs.exists(p):
            continue
        for k, v in fs.find(p, detail=True).items():
            if v.get("type") == "file":
                out.append((k[len(store):].lstrip("/"), v.get("size", 0)))
    return out


def verify_published(bucket: str, dest: str, endpoint: str) -> None:
    """Open the PUBLISHED store and read a timestep from each group.

    This is the check that matters: it proves the copied metadata resolves and
    that the virtual refs still reach NOAA's public S3 from the new location.
    """
    import icechunk
    import numpy as np
    import pandas as pd
    import xarray as xr

    storage = icechunk.s3_storage(
        bucket=bucket, prefix=dest, endpoint_url=endpoint,
        region="us-west-2", anonymous=True)
    auth = icechunk.containers_credentials(
        {NOAA_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    store = repo.readonly_session("main").store

    for group in GROUPS:
        ds = xr.open_zarr(store, group=group, consolidated=False)
        t = pd.DatetimeIndex(ds.time.values)
        # materialise one timestep -> forces a virtual read from noaa-cdr-precip-cmorph-pds
        v = ds.cmorph.isel(time=len(t) // 2).values
        print(f"  [{group or 'root'}] {t.min()} .. {t.max()}  n={len(t):,}  "
              f"chunks={ds.cmorph.encoding.get('chunks')}  "
              f"virtual_read_finite={100 * np.isfinite(v).mean():.1f}%")
    print("VERIFY OK -- published store opens anonymously and virtual refs resolve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-store", default=GCS_STORE)
    ap.add_argument("--dest", default=DEST, help="key prefix under the S3 bucket")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--sa-key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    ap.add_argument("--include-overwritten", action="store_true",
                    help="also copy overwritten/ (superseded ref versions; not needed)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip the copy; just open+read the published store")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-mirror read-back check")
    args = ap.parse_args()

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise SystemExit("AWS_ENDPOINT_URL unset -- did you `source .env`? "
                         "(source.coop STS creds expire after 1 h)")

    if args.verify_only:
        verify_published(args.bucket, args.dest.rstrip("/"), endpoint)
        return

    assert args.gcs_store.startswith("gs://"), "--gcs-store must be gs://..."
    store = args.gcs_store[5:].rstrip("/")
    dest = args.dest.rstrip("/")

    fs = gcsfs.GCSFileSystem(token=args.sa_key)
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      config=Config(s3={"addressing_style": "path"},
                                    max_pool_connections=args.threads + 4))

    # destination inventory (name -> size) for resume/skip
    have = {}
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=args.bucket, Prefix=dest + "/"):
            for o in page.get("Contents", []):
                have[o["Key"]] = o["Size"]
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("ExpiredToken", "AccessDenied",
                                                       "InvalidAccessKeyId"):
            raise SystemExit(
                "source.coop credentials are expired or invalid "
                f"({e.response['Error']['Code']}).\n"
                "  They are 1-hour STS tokens: refresh them in .env, then "
                "`source .env` and re-run.\n"
                "  The mirror is resumable -- it skips whatever is already uploaded.")
        raise
    print(f"destination already holds {len(have)} objects under {dest}/")

    subdirs = DATA_SUBDIRS + (["overwritten"] if args.include_overwritten else [])
    print("listing source objects ...", flush=True)
    t_list = time.time()
    src = list_source(fs, store, subdirs)
    total_bytes = sum(sz for _, sz in src)
    todo = [(rel, sz) for rel, sz in src if have.get(f"{dest}/{rel}") != sz]
    todo_bytes = sum(sz for _, sz in todo)
    print(f"source: {len(src)} objects ({total_bytes/1e6:.1f} MB), "
          f"listed in {time.time()-t_list:.0f}s")
    print(f"to upload: {len(todo)} objects ({todo_bytes/1e6:.1f} MB); "
          f"{len(src)-len(todo)} already in sync")

    if args.dry_run:
        print("dry-run: nothing uploaded")
        return
    if not todo and f"{dest}/repo" in have:
        print("already fully mirrored")
    else:
        def put(rel):
            data = fs.cat_file(f"{store}/{rel}")   # single GET; single PUT (no multipart)
            s3.put_object(Bucket=args.bucket, Key=f"{dest}/{rel}", Body=data)
            return len(data)

        t0 = time.time()
        done = done_bytes = 0
        errors = []
        expired = False
        with ThreadPoolExecutor(args.threads) as ex:
            futs = {ex.submit(put, rel): rel for rel, _ in todo}
            for fut in as_completed(futs):
                try:
                    done_bytes += fut.result()
                    done += 1
                except Exception as e:
                    errors.append((futs[fut], str(e)))
                    if "ExpiredToken" in str(e) or "AccessDenied" in str(e):
                        expired = True
                if done and done % 200 == 0:
                    el = time.time() - t0
                    rate = done / el
                    eta = (len(todo) - done) / rate if rate else float("nan")
                    print(f"  {done}/{len(todo)} objs ({done_bytes/1e6:.1f} MB) "
                          f"{rate:.0f} obj/s  ETA {eta/60:.1f} min", flush=True)
                if expired:
                    break

        if expired:
            print(f"\nSTS token expired/denied after {done} uploads. "
                  f"Refresh .env (source .env) and re-run -- it resumes from here.")
            raise SystemExit(2)
        if errors:
            print(f"{len(errors)} errors, first: {errors[0]}")
            raise SystemExit(1)

        # Upload the branch-pointer `repo` file LAST, now that everything it can
        # reference is present at the destination.
        repo_key = f"{dest}/repo"
        if have.get(repo_key) != fs.info(f"{store}/repo").get("size"):
            s3.put_object(Bucket=args.bucket, Key=repo_key,
                          Body=fs.cat_file(f"{store}/repo"))
            print("uploaded repo pointer (last)")
        else:
            print("repo pointer already in sync")

        print(f"\nuploaded {len(todo)} data objects in {time.time()-t0:.1f}s "
              f"-> s3://{args.bucket}/{dest}")

    print("MIRROR COMPLETE")
    if not args.no_verify:
        print("\nverifying the published store ...")
        verify_published(args.bucket, dest, endpoint)


if __name__ == "__main__":
    main()
