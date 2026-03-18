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
Transfer IMERG Icechunk store from GCS to source.coop (S3).

The IMERG HH store is ~12 GB / 12,000+ objects — too large for a single
download pass with temporary credentials. This script batches the download
by GCS object prefix groups, keeping each batch under a configurable size
limit (default 2 GB). Upload resumes automatically (skips matching files).

Two-step process per batch:
  1. Download batch of GCS objects to local staging directory
  2. Upload to source.coop S3 bucket via boto3

Usage:
    # Dry run — show batches and sizes
    uv run gcs_to_source_coop_transfer.py --dry-run

    # Full transfer (all batches)
    uv run gcs_to_source_coop_transfer.py

    # Upload only (skip download, use existing staging dir)
    uv run gcs_to_source_coop_transfer.py --skip-download

    # Verify uploaded store
    uv run gcs_to_source_coop_transfer.py --verify

    # Custom batch size limit (bytes)
    uv run gcs_to_source_coop_transfer.py --batch-gb 1.5
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
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202503.json"

S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "e4drr-project/observations/imerg_hh_icechunk"


def list_gcs_blobs(sa_path):
    """List all blobs in the GCS store."""
    creds = service_account.Credentials.from_service_account_file(str(sa_path))
    client = gcs_storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(GCS_BUCKET)
    return list(bucket.list_blobs(prefix=GCS_PREFIX))


def partition_blobs(blobs, batch_gb):
    """Partition blobs into batches, each under batch_gb in size.

    Groups by top-level subdirectory (chunks, manifests, snapshots, etc.)
    then splits large groups into sub-batches.
    """
    batch_limit = int(batch_gb * 1024**3)

    # Group by top-level subdir
    groups = {}
    for b in blobs:
        rel = b.name[len(GCS_PREFIX):].lstrip("/")
        subdir = rel.split("/")[0] if "/" in rel else "(root)"
        groups.setdefault(subdir, []).append(b)

    batches = []
    for subdir, group_blobs in sorted(groups.items()):
        group_size = sum(b.size for b in group_blobs)

        if group_size <= batch_limit:
            batches.append({
                "name": subdir,
                "blobs": group_blobs,
                "size": group_size,
            })
        else:
            # Split large groups (e.g. chunks/) into sub-batches
            current = []
            current_size = 0
            part = 1
            for b in group_blobs:
                if current_size + b.size > batch_limit and current:
                    batches.append({
                        "name": f"{subdir}_part{part}",
                        "blobs": current,
                        "size": current_size,
                    })
                    part += 1
                    current = []
                    current_size = 0
                current.append(b)
                current_size += b.size
            if current:
                batches.append({
                    "name": f"{subdir}_part{part}",
                    "blobs": current,
                    "size": current_size,
                })

    return batches


def download_batch(batch, staging_dir, sa_path):
    """Download a batch of GCS blobs to local staging."""
    creds = service_account.Credentials.from_service_account_file(str(sa_path))
    client = gcs_storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(GCS_BUCKET)

    downloaded = 0
    skipped = 0
    start = time.time()

    for i, blob in enumerate(batch["blobs"]):
        # Re-fetch blob to get fresh handle
        blob = bucket.blob(blob.name)
        rel_path = blob.name[len(GCS_PREFIX):].lstrip("/")
        local_path = staging_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.reload()
        if local_path.exists() and local_path.stat().st_size == blob.size:
            skipped += 1
            continue

        blob.download_to_filename(str(local_path))
        downloaded += 1

        if (downloaded + skipped) % 500 == 0:
            elapsed = time.time() - start
            pct = 100 * (i + 1) / len(batch["blobs"])
            print(f"    [{pct:5.1f}%] {downloaded} downloaded, {skipped} skipped ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"    Download: {downloaded} new, {skipped} skipped ({elapsed:.0f}s)")


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


def upload_staging(staging_dir):
    """Upload entire staging directory to source.coop S3."""
    s3 = get_s3_client()

    files = sorted(f for f in staging_dir.rglob("*") if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Upload: {len(files)} files, {total_size / (1024**3):.2f} GB")
    print(f"  Target: s3://{S3_BUCKET}/{S3_PREFIX}/")

    uploaded = 0
    skipped = 0
    start = time.time()

    for i, local_file in enumerate(files):
        rel_path = local_file.relative_to(staging_dir)
        s3_key = f"{S3_PREFIX}/{rel_path}"
        local_size = local_file.stat().st_size

        # Check if already uploaded with matching size
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            if head["ContentLength"] == local_size:
                skipped += 1
                if (uploaded + skipped) % 500 == 0:
                    elapsed = time.time() - start
                    pct = 100 * (i + 1) / len(files)
                    print(f"    [{pct:5.1f}%] {uploaded} uploaded, {skipped} skipped ({elapsed:.0f}s)")
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
            print(f"    [{pct:5.1f}%] {uploaded} uploaded, {skipped} skipped ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"    Upload complete: {uploaded} new, {skipped} skipped ({elapsed:.0f}s)")
    print(f"    URL: s3://{S3_BUCKET}/{S3_PREFIX}/")


def verify_store():
    """Open uploaded Icechunk store from S3 and print summary."""
    import icechunk
    import xarray as xr

    print(f"  Opening: s3://{S3_BUCKET}/{S3_PREFIX}/")

    storage = icechunk.s3_storage(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
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

    if "time" in ds.coords:
        print(f"  Time range: {ds.time.values[0]} → {ds.time.values[-1]}")
        print(f"  Time steps: {len(ds.time)}")

    for var in list(ds.data_vars)[:1]:
        sample = ds[var].isel(time=0).values
        import numpy as np
        valid = np.count_nonzero(~np.isnan(sample))
        total = sample.size
        print(f"  {var} (t=0): {valid}/{total} valid pixels ({100*valid/total:.1f}%)")

    ds.close()
    print("  Verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer IMERG Icechunk store from GCS to source.coop"
    )
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE,
                        help="GCS service account JSON file")
    parser.add_argument("--staging-dir", type=str, default="./gcs_staging/imerg",
                        help="Local staging directory")
    parser.add_argument("--batch-gb", type=float, default=2.0,
                        help="Max batch size in GB for download (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show batches and sizes, don't transfer")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GCS download, upload existing staging dir")
    parser.add_argument("--verify", action="store_true",
                        help="Verify uploaded store from S3")
    args = parser.parse_args()

    if args.verify:
        print("\n=== Verify IMERG HH on source.coop ===")
        verify_store()
        return

    sa_path = Path(__file__).parent / ".." / "hf-gdo" / args.sa_file
    if not sa_path.exists():
        # Also check current directory
        sa_path = Path(__file__).parent / args.sa_file
    if not sa_path.exists():
        print(f"ERROR: Service account file not found: {sa_path}")
        return

    staging_dir = Path(args.staging_dir)

    if args.skip_download:
        print(f"\n=== Upload existing staging → source.coop ===")
        if not staging_dir.exists():
            print(f"ERROR: Staging dir not found: {staging_dir}")
            return
        upload_staging(staging_dir)
        print("\nDone.")
        return

    # List all blobs and partition into batches
    print(f"Listing gs://{GCS_BUCKET}/{GCS_PREFIX}/ ...")
    blobs = list_gcs_blobs(sa_path)
    total_size = sum(b.size for b in blobs)
    print(f"  {len(blobs)} objects, {total_size / (1024**3):.2f} GB")

    batches = partition_blobs(blobs, args.batch_gb)
    print(f"\n  Batches ({args.batch_gb} GB limit):")
    for i, batch in enumerate(batches):
        sz = f"{batch['size']/(1024**3):.2f} GB" if batch["size"] > 1e9 else f"{batch['size']/(1024**2):.1f} MB"
        print(f"    {i+1}. {batch['name']:<25} {len(batch['blobs']):>6} files  {sz:>10}")

    print(f"\n  S3 target: s3://{S3_BUCKET}/{S3_PREFIX}/")

    if args.dry_run:
        print("\n  (dry run — no transfer)")
        return

    # Download and upload in batches
    staging_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        sz = f"{batch['size']/(1024**3):.2f} GB" if batch["size"] > 1e9 else f"{batch['size']/(1024**2):.1f} MB"
        print(f"\n{'='*60}")
        print(f"Batch {i+1}/{len(batches)}: {batch['name']} ({len(batch['blobs'])} files, {sz})")
        print(f"{'='*60}")

        print(f"\n  Step 1: Download GCS → {staging_dir}")
        download_batch(batch, staging_dir, sa_path)

    # Upload all at once after downloading
    print(f"\n{'='*60}")
    print("Upload all to source.coop")
    print(f"{'='*60}")
    upload_staging(staging_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
