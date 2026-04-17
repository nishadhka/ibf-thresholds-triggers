#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=0.1.0",
#     "zarr>=3.0.0",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Repair the source.coop IMERG-HH icechunk store.

Background
----------
extend_store() in imerg_hh_gcs_icechunk.py used
    pd.date_range(next_day, new_end, freq="30min")  # *both ends inclusive*
which produced an extra 00:00 timestamp at the tail of every extension.
The next extend's `(current_end + 30min).normalize()` then rounded back to
that midnight, creating duplicate timestamps and shifting subsequent day
boundaries by one half-hour step.

Combined with extend_store() materialising literal zeros (no compute=False),
the store ended up with:

  * duplicate time entries at idx 451632 (2026-03-06T00:00:00, ×2) and
    idx 451681 (2026-03-07T00:00:00, ×2)
  * literal-zero values for 2026-03-06 (whole day) and 2026-03-10 (whole day)
  * off-by-one writes for 2026-03-07..2026-03-09 (data is shifted +30 min
    relative to the timestamps the consumer reads)

This script truncates the array+time-coord at the first duplicate index
(451632 — first slot of 2026-03-06), then re-extends and re-fills via the
*fixed* extend_store / fill_store pipeline.

Safety
------
Default mode is --dry-run: prints what it would do, writes nothing.
Pass --apply to actually mutate the source.coop store.  Requires
SOURCE_COOP_ACCESS_KEY_ID / SOURCE_COOP_SECRET_ACCESS_KEY (and optionally
SOURCE_COOP_SESSION_TOKEN) in env or .env, *plus* EARTHDATA_USERNAME /
EARTHDATA_PASSWORD for the THREDDS re-fill step.

Usage
-----
    # 1. Dry run — inspect the corrupt indices and proposed truncation point
    uv run --python 3.12 repair_imerg_tail.py

    # 2. Truncate only (does not re-fill)
    uv run --python 3.12 repair_imerg_tail.py --apply --truncate-only

    # 3. Truncate + extend to 2026-03-10 + re-fill via Late
    uv run --python 3.12 repair_imerg_tail.py --apply \\
        --refill-end 2026-03-10 --product late
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_PREFIX = "e4drr-project/observations/imerg_hh_icechunk"


def open_repo(write: bool):
    import icechunk

    if write:
        access_key = (
            os.getenv("SOURCE_COOP_ACCESS_KEY_ID")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        session_token = (
            os.getenv("SOURCE_COOP_SESSION_TOKEN")
            or os.getenv("AWS_SESSION_TOKEN")
        )
        if not access_key or not secret_key:
            raise RuntimeError(
                "Write mode needs SOURCE_COOP_ACCESS_KEY_ID and "
                "SOURCE_COOP_SECRET_ACCESS_KEY (or AWS_*) in env / .env."
            )
        storage = icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=SOURCE_COOP_PREFIX,
            region="us-west-2",
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=session_token,
        )
    else:
        storage = icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=SOURCE_COOP_PREFIX,
            region="us-west-2",
            anonymous=True,
        )
    return icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )


def find_first_duplicate_idx(times: pd.DatetimeIndex) -> int | None:
    """Return the index of the *first* occurrence of the earliest duplicate.

    Truncating to this length drops both the original and the duplicate
    (and everything after), leaving the array ending at a clean
    half-hour-before-midnight boundary that extend_store can resume from.
    """
    deltas = times[1:] - times[:-1]
    bad = np.where(deltas == pd.Timedelta(0))[0]
    if len(bad) == 0:
        return None
    return int(bad[0])  # first occurrence — truncate drops it and the dup


def diagnose(repo) -> dict:
    import xarray as xr

    sess = repo.readonly_session("main")
    ds = xr.open_zarr(sess.store, consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    dup_idx = find_first_duplicate_idx(times)
    n = len(times)
    diag = {
        "n_steps": n,
        "first_time": times[0],
        "last_time": times[-1],
        "first_dup_idx": dup_idx,
        "first_dup_time": times[dup_idx] if dup_idx is not None else None,
        "n_lat": int(ds.sizes["lat"]),
        "n_lon": int(ds.sizes["lon"]),
    }
    return diag


def truncate_tail(repo, truncate_at: int, message: str) -> None:
    """Resize the precipitation array and time coord to length=truncate_at."""
    import zarr

    sess = repo.writable_session("main")
    root = zarr.open_group(sess.store, mode="r+", zarr_format=3)
    precip = root["precipitation"]
    time_arr = root["time"]

    old_t = precip.shape[0]
    if old_t == truncate_at:
        print(f"  [skip] precipitation already at length {old_t}")
        return
    if truncate_at > old_t:
        raise RuntimeError(
            f"truncate_at={truncate_at} > current length {old_t}"
        )

    new_shape = (truncate_at,) + precip.shape[1:]
    print(f"  precipitation: resize {precip.shape} -> {new_shape}")
    precip.resize(new_shape)
    print(f"  time         : resize ({time_arr.shape[0]},) -> ({truncate_at},)")
    time_arr.resize((truncate_at,))

    sess.commit(message)
    print(f"  committed: {message}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Actually mutate the source.coop store. Default: dry-run.")
    p.add_argument("--truncate-only", action="store_true",
                   help="Only truncate the corrupt tail; skip extend + re-fill.")
    p.add_argument("--refill-end", type=str, default=None,
                   help="End date for the post-truncate extend+fill (YYYY-MM-DD).")
    p.add_argument("--fill-start", type=str, default=None,
                   help="Override the fill-window start (YYYY-MM-DD). "
                        "Default: day after the pre-extend tail timestamp.")
    p.add_argument("--product", type=str, default="late",
                   choices=["final", "late", "early"],
                   help="IMERG product to re-fill with (default: late).")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Coiled workers for re-fill (small batch — default 4).")
    p.add_argument("--no-cluster", action="store_true",
                   help="Run re-fill sequentially without Coiled.")
    args = p.parse_args()

    print("=" * 60)
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN (no writes)'}")
    print("=" * 60)

    print("\n[1/3] Diagnose")
    repo_ro = open_repo(write=False)
    diag = diagnose(repo_ro)
    print(f"  n_steps      : {diag['n_steps']}")
    print(f"  first_time   : {diag['first_time']}")
    print(f"  last_time    : {diag['last_time']}")
    print(f"  first_dup_idx: {diag['first_dup_idx']}")
    print(f"  first_dup_t  : {diag['first_dup_time']}")
    print(f"  spatial dims : ({diag['n_lat']}, {diag['n_lon']})")
    if diag["first_dup_idx"] is None:
        print("  -> no duplicate timestamps; nothing to truncate.")
        if not args.refill_end:
            return

    truncate_at = diag["first_dup_idx"]
    if truncate_at is not None:
        truncate_keep_through = pd.Timestamp(diag["first_dup_time"]) - pd.Timedelta(minutes=30)
        print(f"\n  -> would truncate to length {truncate_at}")
        print(f"     keeping data through {truncate_keep_through}")

    print("\n[2/3] Truncate tail")
    if args.apply and truncate_at is not None:
        repo_rw = open_repo(write=True)
        truncate_tail(
            repo_rw, truncate_at,
            message=f"truncate corrupt tail at idx {truncate_at} "
                    f"(removes duplicate timestamps + zero-filled days)",
        )
    else:
        print("  [dry-run / nothing to truncate]")

    print("\n[3/3] Re-extend + re-fill")
    if not args.refill_end or args.truncate_only:
        print("  [skipped — pass --refill-end YYYY-MM-DD to enable]")
        print("\nDone.")
        return

    if not args.apply:
        print(f"  [dry-run] would extend to {args.refill_end} and "
              f"fill via {args.product}")
        print("\nDone.")
        return

    # Re-use the fixed extend / fill from imerg_hh_gcs_icechunk.py.
    # Note: that module is GCS-targeted by default; we monkey-patch its
    # storage helper to point at source.coop instead.
    sys.path.insert(0, str(Path(__file__).parent))
    import imerg_hh_gcs_icechunk as pipeline

    def s3_storage(_gcs_prefix=None):
        import icechunk
        return icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=SOURCE_COOP_PREFIX,
            region="us-west-2",
            access_key_id=os.getenv("SOURCE_COOP_ACCESS_KEY_ID")
                          or os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY")
                              or os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("SOURCE_COOP_SESSION_TOKEN")
                          or os.getenv("AWS_SESSION_TOKEN"),
        )

    pipeline.get_gcs_storage = s3_storage  # noqa: redirect to S3

    # Capture the tail timestamp *before* extending, so we know where the
    # data-less region starts.  diag2 is refreshed post-truncate (or is the
    # same as diag if this is a re-run after truncation already happened).
    diag2 = diagnose(open_repo(write=False))
    last_kept_pre_extend = pd.Timestamp(diag2["last_time"])

    extend_args = argparse.Namespace(
        gcs_prefix=None,
        end_date=args.refill_end,
    )
    print(f"  Extending time axis to {args.refill_end} ...")
    pipeline.extend_store(extend_args)

    # Fill window: day after the pre-extend tail up through refill-end,
    # unless --fill-start overrides it (useful on re-runs when the extend
    # has already happened).
    if args.fill_start:
        fill_start = pd.Timestamp(args.fill_start).normalize()
    else:
        fill_start = (last_kept_pre_extend + pd.Timedelta(minutes=30)).normalize()
    fill_args = argparse.Namespace(
        gcs_prefix=None,
        start_date=str(fill_start.date()),
        end_date=args.refill_end,
        product=args.product,
        n_workers=args.n_workers,
        commit_batch=5,
        no_cluster=args.no_cluster,
        force=True,  # bypass resume-detection — historical batch commits
                     # still reference the now-truncated tail indices
    )
    print(f"  Filling {fill_args.start_date} -> {fill_args.end_date} "
          f"via {args.product} ...")
    pipeline.fill_store(fill_args)

    print("\nDone.")


if __name__ == "__main__":
    main()
