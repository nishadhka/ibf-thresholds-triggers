#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-cloud-storage>=2.14.0",
#     "boto3>=1.34.0",
#     "icechunk>=0.1.0",
#     "xarray>=2024.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Transfer Icechunk stores from GCS to source.coop (S3).

Two-step process:
  1. Sync GCS store to a local staging directory
  2. Upload to source.coop S3 bucket via boto3

The local staging dir can be cleaned up after upload, or kept for
incremental re-uploads (only changed files are re-transferred).

Usage:
    # Dry run — list GCS files and sizes
    uv run gcs_to_source_coop_transfer.py rfe2 --dry-run

    # Full transfer (download GCS → local → upload to S3)
    uv run gcs_to_source_coop_transfer.py rfe2

    # All 4 datasets
    uv run gcs_to_source_coop_transfer.py all

    # Resume upload only (skip GCS download)
    uv run gcs_to_source_coop_transfer.py rfe2 --skip-download

    # Verify uploaded store
    uv run gcs_to_source_coop_transfer.py rfe2 --verify
"""

import argparse
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from google.cloud import storage as gcs_storage
from google.oauth2 import service_account

load_dotenv()

GCS_BUCKET = "cpc_awc"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202503.json"

S3_BUCKET = "us-west-2.opendata.source.coop"
S3_BASE_PREFIX = "e4drr-project/observations"

DATASETS = {
    "chirps_spi": {
        "gcs_prefix": "chirps_spi_ic_store",
        "s3_prefix": "chirps_spi_icechunk",
    },
    "gdo_fpar": {
        "gcs_prefix": "gdo_fpar_ic_store",
        "s3_prefix": "gdo_fpar_icechunk",
    },
    "gdo_sma": {
        "gcs_prefix": "gdo_sma_ic_store",
        "s3_prefix": "gdo_sma_icechunk",
    },
    "rfe2": {
        "gcs_prefix": "rfe2_ic_store",
        "s3_prefix": "rfe2_icechunk",
    },
}


def sync_gcs_to_local(bucket_name, prefix, staging_dir, sa_path):
    """Download all objects from GCS prefix to local staging directory."""
    creds = service_account.Credentials.from_service_account_file(str(sa_path))
    client = gcs_storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    total_size = sum(b.size for b in blobs)
    print(f"  GCS: {len(blobs)} objects, {total_size / (1024**3):.2f} GB")

    downloaded = 0
    skipped = 0
    start = time.time()

    for i, blob in enumerate(blobs):
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_path = staging_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded with matching size
        if local_path.exists() and local_path.stat().st_size == blob.size:
            skipped += 1
            continue

        blob.download_to_filename(str(local_path))
        downloaded += 1

        if (downloaded + skipped) % 500 == 0:
            elapsed = time.time() - start
            pct = 100 * (i + 1) / len(blobs)
            print(f"  [{pct:5.1f}%] {downloaded} downloaded, {skipped} skipped ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"  Download complete: {downloaded} new, {skipped} skipped ({elapsed:.0f}s)")
    return len(blobs)


def get_s3_client():
    """Create boto3 S3 client with source.coop credentials.

    Checks SOURCE_COOP_* env vars first, falls back to AWS_* env vars.
    """
    access_key = os.getenv("SOURCE_COOP_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("SOURCE_COOP_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "Set SOURCE_COOP_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID and "
            "SOURCE_COOP_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY "
            "in .env or environment"
        )

    return boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )


def upload_to_source_coop(staging_dir, s3_prefix):
    """Upload local staging directory to source.coop S3 bucket."""
    s3 = get_s3_client()
    full_prefix = f"{S3_BASE_PREFIX}/{s3_prefix}"

    # Collect all files
    files = sorted(f for f in staging_dir.rglob("*") if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Local: {len(files)} files, {total_size / (1024**3):.2f} GB")
    print(f"  Target: s3://{S3_BUCKET}/{full_prefix}/")

    uploaded = 0
    skipped = 0
    start = time.time()

    for i, local_file in enumerate(files):
        rel_path = local_file.relative_to(staging_dir)
        s3_key = f"{full_prefix}/{rel_path}"
        local_size = local_file.stat().st_size

        # Check if already uploaded with matching size
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            if head["ContentLength"] == local_size:
                skipped += 1
                if (uploaded + skipped) % 500 == 0:
                    elapsed = time.time() - start
                    pct = 100 * (i + 1) / len(files)
                    print(f"  [{pct:5.1f}%] {uploaded} uploaded, {skipped} skipped ({elapsed:.0f}s)")
                continue
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in ("404", "400", "403"):
                raise

        with open(local_file, "rb") as f:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f,
                ACL="bucket-owner-full-control",
            )
        uploaded += 1

        if (uploaded + skipped) % 500 == 0:
            elapsed = time.time() - start
            pct = 100 * (i + 1) / len(files)
            print(f"  [{pct:5.1f}%] {uploaded} uploaded, {skipped} skipped ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"  Upload complete: {uploaded} new, {skipped} skipped ({elapsed:.0f}s)")
    print(f"  URL: s3://{S3_BUCKET}/{full_prefix}/")


def verify_store(s3_prefix):
    """Open uploaded Icechunk store from S3 and print summary."""
    import icechunk
    import xarray as xr

    full_prefix = f"{S3_BASE_PREFIX}/{s3_prefix}"
    print(f"  Opening: s3://{S3_BUCKET}/{full_prefix}/")

    storage = icechunk.s3_storage(
        bucket=S3_BUCKET,
        prefix=full_prefix,
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    print(f"\n  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables:  {list(ds.data_vars)}")
    print(f"  Coords:     {list(ds.coords)}")

    # Spot-check: first and last time values
    if "time" in ds.coords:
        print(f"  Time range: {ds.time.values[0]} → {ds.time.values[-1]}")
        print(f"  Time steps: {len(ds.time)}")

    # Check a data variable has non-null values
    for var in list(ds.data_vars)[:1]:
        sample = ds[var].isel(time=0).values
        import numpy as np
        valid = np.count_nonzero(~np.isnan(sample))
        total = sample.size
        print(f"  {var} (t=0): {valid}/{total} valid pixels ({100*valid/total:.1f}%)")

    ds.close()
    print("  Verification passed.")


def dry_run(dataset_name, info, sa_path):
    """List GCS files and target S3 paths."""
    creds = service_account.Credentials.from_service_account_file(str(sa_path))
    client = gcs_storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix=info["gcs_prefix"]))
    total_size = sum(b.size for b in blobs)

    subdirs = {}
    for b in blobs:
        rel = b.name[len(info["gcs_prefix"]):].lstrip("/")
        subdir = rel.split("/")[0] if "/" in rel else "(root)"
        if subdir not in subdirs:
            subdirs[subdir] = {"count": 0, "size": 0}
        subdirs[subdir]["count"] += 1
        subdirs[subdir]["size"] += b.size

    full_s3 = f"s3://{S3_BUCKET}/{S3_BASE_PREFIX}/{info['s3_prefix']}/"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"  GCS source:  gs://{GCS_BUCKET}/{info['gcs_prefix']}/")
    print(f"  S3 target:   {full_s3}")
    print(f"  {len(blobs)} objects, {total_size / (1024**3):.2f} GB\n")
    print(f"  {'Subdirectory':<20} {'Files':>8} {'Size':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")
    for sd, si in sorted(subdirs.items()):
        sz = f"{si['size']/(1024**3):.2f} GB" if si["size"] > 1e9 else f"{si['size']/(1024**2):.1f} MB"
        print(f"  {sd:<20} {si['count']:>8} {sz:>12}")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Icechunk stores from GCS to source.coop"
    )
    parser.add_argument(
        "dataset",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to transfer (or 'all' for all datasets)",
    )
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE,
                        help="GCS service account JSON file")
    parser.add_argument("--staging-dir", type=str, default="./gcs_staging",
                        help="Local staging directory for GCS download")
    parser.add_argument("--dry-run", action="store_true",
                        help="List GCS files only, don't transfer")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GCS download, upload existing staging dir")
    parser.add_argument("--verify", action="store_true",
                        help="Verify uploaded store (open from S3 and check)")
    args = parser.parse_args()

    sa_path = Path(__file__).parent / args.sa_file
    if not args.verify and not sa_path.exists():
        print(f"ERROR: Service account file not found: {sa_path}")
        return

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        if args.dry_run:
            dry_run(name, info, sa_path)
            continue

        if args.verify:
            print(f"\n--- Verify: {name} ---")
            verify_store(info["s3_prefix"])
            continue

        staging_dir = Path(args.staging_dir) / info["gcs_prefix"]

        # Step 1: Download GCS → local staging
        if not args.skip_download:
            print(f"\n--- Step 1: Sync GCS → {staging_dir} ---")
            staging_dir.mkdir(parents=True, exist_ok=True)
            sync_gcs_to_local(GCS_BUCKET, info["gcs_prefix"], staging_dir, sa_path)
        else:
            print(f"\n--- Step 1: Skipped (using existing {staging_dir}) ---")

        # Step 2: Upload staging → source.coop S3
        print(f"\n--- Step 2: Upload → source.coop ---")
        upload_to_source_coop(staging_dir, info["s3_prefix"])

    print("\nDone.")


if __name__ == "__main__":
    main()
