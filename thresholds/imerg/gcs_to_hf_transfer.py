#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-cloud-storage>=2.14.0",
#     "huggingface_hub>=0.20.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Transfer Icechunk store from GCS to HuggingFace.

Two-step process:
  1. Sync GCS store to a local staging directory (~12 GB)
  2. Upload to HF using upload_large_folder() which handles rate limits,
     chunking, and resumption automatically

The local staging dir can be cleaned up after upload, or kept for
incremental re-uploads (only changed files are re-transferred).

Usage:
    # Dry run — list GCS files and sizes
    uv run --python 3.12 gcs_to_hf_transfer.py --dry-run

    # Full transfer (download GCS → local staging → upload to HF)
    uv run --python 3.12 gcs_to_hf_transfer.py

    # Resume a previous upload (skips already-uploaded files)
    uv run --python 3.12 gcs_to_hf_transfer.py --skip-download

    # Custom paths
    uv run --python 3.12 gcs_to_hf_transfer.py \
        --gcs-prefix test_ea_imerg_ic_store \
        --hf-prefix test_imerg_ic_store \
        --staging-dir /tmp/imerg_staging
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage as gcs_storage
from google.oauth2 import service_account
from huggingface_hub import HfApi

load_dotenv()

GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "ea_imerg_ic_store"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"

HF_REPO = "E4DRR/icechunk-stores"
HF_PREFIX = "ea_imerg_ic_store"


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


def upload_to_hf(staging_dir, hf_repo, hf_prefix, hf_token):
    """Upload local staging directory to HuggingFace using upload_large_folder.

    upload_large_folder handles:
    - Rate limiting (waits and retries on 429)
    - Large file chunking via LFS
    - Resumption (tracks uploaded files, skips already-done)
    - Efficient multi-file commits
    """
    api = HfApi(token=hf_token)

    # Ensure repo exists
    try:
        api.repo_info(repo_id=hf_repo, repo_type="dataset")
        print(f"  HF repo exists: {hf_repo}")
    except Exception:
        print(f"  Creating HF repo: {hf_repo}")
        api.create_repo(repo_id=hf_repo, repo_type="dataset", private=False)

    print(f"  Uploading {staging_dir} → {hf_repo}/{hf_prefix}/")
    print(f"  (upload_large_folder handles rate limits and resumption)")

    start = time.time()
    api.upload_large_folder(
        repo_id=hf_repo,
        repo_type="dataset",
        folder_path=str(staging_dir),
        path_in_repo=hf_prefix,
    )
    elapsed = time.time() - start
    print(f"  Upload complete in {elapsed:.0f}s")
    print(f"  URL: https://huggingface.co/datasets/{hf_repo}/tree/main/{hf_prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Icechunk store from GCS to HuggingFace"
    )
    parser.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    parser.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    parser.add_argument("--hf-repo", type=str, default=HF_REPO)
    parser.add_argument("--hf-prefix", type=str, default=HF_PREFIX)
    parser.add_argument("--sa-file", type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--staging-dir", type=str, default="./gcs_staging",
                        help="Local staging directory for GCS download")
    parser.add_argument("--dry-run", action="store_true",
                        help="List GCS files only, don't transfer")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GCS download, upload existing staging dir")
    args = parser.parse_args()

    sa_path = Path(__file__).parent / args.sa_file
    if not sa_path.exists():
        print(f"ERROR: Service account file not found: {sa_path}")
        return

    staging_dir = Path(args.staging_dir)

    if args.dry_run:
        creds = service_account.Credentials.from_service_account_file(str(sa_path))
        client = gcs_storage.Client(credentials=creds, project=creds.project_id)
        bucket = client.bucket(args.gcs_bucket)
        blobs = list(bucket.list_blobs(prefix=args.gcs_prefix))
        total_size = sum(b.size for b in blobs)

        subdirs = {}
        for b in blobs:
            rel = b.name[len(args.gcs_prefix):].lstrip("/")
            subdir = rel.split("/")[0] if "/" in rel else "(root)"
            if subdir not in subdirs:
                subdirs[subdir] = {"count": 0, "size": 0}
            subdirs[subdir]["count"] += 1
            subdirs[subdir]["size"] += b.size

        print(f"GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}/")
        print(f"  {len(blobs)} objects, {total_size / (1024**3):.2f} GB\n")
        print(f"  {'Subdirectory':<20} {'Files':>8} {'Size':>12}")
        print(f"  {'-'*20} {'-'*8} {'-'*12}")
        for sd, info in sorted(subdirs.items()):
            sz = f"{info['size']/(1024**3):.2f} GB" if info["size"] > 1e9 else f"{info['size']/(1024**2):.1f} MB"
            print(f"  {sd:<20} {info['count']:>8} {sz:>12}")
        print(f"\n  Staging dir: {staging_dir}")
        print(f"  HF target:   {args.hf_repo}/{args.hf_prefix}/")
        return

    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
    if not hf_token:
        print("ERROR: HF_TOKEN not set in .env or environment")
        return

    # Step 1: Download GCS → local staging
    if not args.skip_download:
        print(f"\n=== Step 1: Sync GCS → {staging_dir} ===")
        staging_dir.mkdir(parents=True, exist_ok=True)
        sync_gcs_to_local(args.gcs_bucket, args.gcs_prefix, staging_dir, sa_path)
    else:
        print(f"\n=== Step 1: Skipped (using existing {staging_dir}) ===")

    # Step 2: Upload staging → HuggingFace
    print(f"\n=== Step 2: Upload {staging_dir} → HuggingFace ===")
    upload_to_hf(staging_dir, args.hf_repo, args.hf_prefix, hf_token)


if __name__ == "__main__":
    main()
