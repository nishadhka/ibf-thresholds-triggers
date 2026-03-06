#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=0.1.0",
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "huggingface_hub>=0.20.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Test Icechunk store upload to HuggingFace Hub.

Two approaches tested:
  1. huggingface_hub API: upload local store directory to HF dataset repo
  2. icechunk s3_storage: direct S3-compatible access (experimental)

Usage:
    # Set HF_TOKEN in .env or environment
    uv run --python 3.12 test_hf_icechunk.py --store ./imerg_ea_local
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_REPO = "E4DRR/icechunk-stores"
HF_STORE_PREFIX = "imerg-v7-ea-store"


def test_hf_hub_upload(store_path: str, dry_run: bool = True):
    """Test uploading local Icechunk store to HF via huggingface_hub API.

    This is the most reliable approach: upload the store directory as files
    to a HuggingFace dataset repository.
    """
    from huggingface_hub import HfApi

    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
    if not hf_token:
        print("SKIP: HF_TOKEN not set — cannot test upload")
        print("  Set HF_TOKEN=hf_... in .env or environment to test")
        return False

    api = HfApi(token=hf_token)

    # Check repo exists / create
    try:
        api.repo_info(repo_id=HF_REPO, repo_type="dataset")
        print(f"  HF repo exists: {HF_REPO}")
    except Exception as e:
        print(f"  HF repo {HF_REPO} not found: {e}")
        if not dry_run:
            api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=False)
            print(f"  Created repo: {HF_REPO}")
        return False

    store_dir = Path(store_path)
    if not store_dir.exists():
        print(f"  Store not found: {store_dir}")
        return False

    # Safety: reject if store dir contains sensitive files
    BLOCKED = {".env", ".netrc", ".dodsrc"}
    files = list(store_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    bad_files = [f for f in files if f.name in BLOCKED or f.suffix in {".py", ".ipynb"}]
    if bad_files:
        print(f"  ABORT: Store dir contains non-store files that should not be uploaded:")
        for bf in bad_files:
            print(f"    {bf}")
        print("  Only upload a directory that contains ONLY the Icechunk store.")
        return False
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Store: {len(files)} files, {total_size / 1024:.1f} KB")

    if dry_run:
        print("  DRY RUN — would upload to:")
        print(f"    repo: {HF_REPO}")
        print(f"    path_in_repo: {HF_STORE_PREFIX}/")
        print(f"    files: {len(files)}")
        return True

    # Upload
    print(f"  Uploading to {HF_REPO}/{HF_STORE_PREFIX}/...")
    api.upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=str(store_dir),
        path_in_repo=HF_STORE_PREFIX,
        commit_message=f"Upload IMERG v7 EA Icechunk store ({len(files)} files)",
    )
    print(f"  Upload complete!")
    return True


def test_icechunk_s3_hf(dry_run: bool = True):
    """Test icechunk s3_storage pointed at HuggingFace S3-compatible API.

    HuggingFace provides S3-compatible access to dataset repos.
    This is the direct approach but may have compatibility issues.
    """
    import icechunk

    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf")
    if not hf_token:
        print("SKIP: HF_TOKEN not set — cannot test S3 access")
        return False

    print("  Testing icechunk.s3_storage with HF endpoint...")
    try:
        storage = icechunk.s3_storage(
            bucket=HF_REPO,
            prefix=HF_STORE_PREFIX,
            endpoint_url="https://huggingface.co",
            access_key_id="",
            secret_access_key=hf_token,
            force_path_style=True,
            allow_http=False,
        )
        print(f"  Storage config created (not yet connected)")

        if not dry_run:
            config = icechunk.RepositoryConfig.default()
            try:
                repo = icechunk.Repository.open(storage, config=config)
                session = repo.readonly_session("main")
                print(f"  Connected to existing repo!")
            except Exception:
                repo = icechunk.Repository.create(storage, config=config)
                print(f"  Created new repo on HF!")
        return True
    except Exception as e:
        print(f"  S3-compatible HF access failed: {e}")
        print("  Fallback: use huggingface_hub upload approach instead")
        return False


def test_dask_worker_pattern():
    """Evaluate how Dask workers interact with the fill pipeline.

    The fill pipeline follows a fork-merge pattern (same as CMORPH):

    Coordinator (local):
      1. earthaccess.search_data() → list of granules
      2. earthaccess.download() → download batch of NetCDF files
      3. Submit _read_imerg_file_ea() to Dask workers
      4. Collect results, write to Icechunk session
      5. Commit batch

    Workers (Coiled):
      - Receive file path (on coordinator's filesystem)
      - ISSUE: Workers can't read coordinator's local files!

    Solution options:
      A. Download on workers: workers run earthaccess.download() themselves
         - Requires .netrc on each worker
         - earthaccess handles auth propagation
      B. Scatter data: coordinator downloads, scatters bytes to workers
         - More memory on coordinator
         - Works without auth on workers
      C. Coordinator reads sequentially (current --no-cluster path)
         - Simplest, no auth issues
         - Good for small batches, slow for large ones
      D. Download to shared storage (GCS), workers read from there
         - Most robust for large-scale runs

    Current implementation uses option A for cluster mode.
    For production, option D (GCS staging) is recommended.
    """
    print("  Dask worker pattern analysis:")
    print()
    print("  Current pipeline: earthaccess.download() → netcdf4 read → icechunk write")
    print("  NOT using OPeNDAP (GES DISC rate-limits concurrent connections)")
    print()
    print("  Cluster mode (Coiled):")
    print("    - Coordinator downloads batch of NetCDF granules")
    print("    - ISSUE: Downloaded files are on coordinator filesystem")
    print("    - Workers cannot access coordinator's local files!")
    print()
    print("  RECOMMENDED FIX for production:")
    print("    Option A: Workers download their own granules")
    print("      - Pass search results to workers")
    print("      - Workers call earthaccess.download() + read + subset")
    print("      - Return numpy arrays to coordinator")
    print("      - Coordinator writes to Icechunk")
    print()
    print("    Option B: Stage downloads on GCS/S3 shared storage")
    print("      - Coordinator downloads to GCS bucket")
    print("      - Workers read from GCS")
    print("      - Most robust for large-scale runs")
    print()
    print("  Current --no-cluster mode works correctly (sequential)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test HF Icechunk integration")
    parser.add_argument("--store", type=str, default="./imerg_ea_local")
    parser.add_argument("--prefix", type=str, default=None,
                        help="HF store prefix (overrides default)")
    parser.add_argument("--upload", action="store_true",
                        help="Actually upload (not dry run)")
    args = parser.parse_args()

    if args.prefix:
        global HF_STORE_PREFIX
        HF_STORE_PREFIX = args.prefix
        print(f"Using custom prefix: {HF_STORE_PREFIX}")

    print("=" * 60)
    print("TEST 1: HuggingFace Hub upload (recommended approach)")
    print("=" * 60)
    test_hf_hub_upload(args.store, dry_run=not args.upload)

    print()
    print("=" * 60)
    print("TEST 2: Icechunk S3-compatible HF access (experimental)")
    print("=" * 60)
    test_icechunk_s3_hf(dry_run=not args.upload)

    print()
    print("=" * 60)
    print("TEST 3: Dask worker pattern evaluation")
    print("=" * 60)
    test_dask_worker_pattern()


if __name__ == "__main__":
    main()
