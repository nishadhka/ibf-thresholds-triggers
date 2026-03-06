# Icechunk + HuggingFace Storage Strategy

## Why direct Dask worker writes to HuggingFace fail

HuggingFace provides S3-compatible API access to dataset repositories, but
Icechunk's `s3_storage()` cannot use it as a remote backend. Three configurations
were tested — all failed:

| Config | Endpoint | Bucket | Result |
|--------|----------|--------|--------|
| 1 | `https://huggingface.co` | `E4DRR/icechunk-stores` | Connection refused |
| 2 | `https://huggingface.co/api/datasets` | `E4DRR/icechunk-stores` | 403 Forbidden |
| 3 | `https://s3.us-east-1.amazonaws.com` (HF backend) | `hf-datasets-E4DRR` | Bucket not found |

**Root cause:** HuggingFace's storage layer is git-LFS backed, not a true
S3-compatible object store. Icechunk requires atomic object-level operations
(put, get, delete, list) that HF's API does not support. The S3 endpoint HF
exposes is read-oriented and does not support the write patterns Icechunk needs
for commits (writing manifests, chunk refs, and snapshot metadata atomically).

This means **Dask workers cannot commit directly to a HuggingFace-hosted
Icechunk store** — there is no viable remote storage configuration for it.

## Proposed solution: GCS for writes, HuggingFace for distribution

```
                    BACKFILL (one-time, large)
                    ─────────────────────────
Coiled Workers ──download+subset──► Coordinator ──write──► GCS Icechunk Store
                                                                  │
                                                           sync to HF
                                                                  │
                                                                  ▼
                                                     HuggingFace Dataset Repo
                                                     (free TB-scale storage)
                                                                  ▲
                                                                  │
                                                           sync to HF
                                                                  │
                    DAILY/WEEKLY UPDATE (small)
                    ───────────────────────────
            Single process ──download+subset+write──► Local Icechunk Store
                                                           │
                                                      upload delta
                                                           │
                                                           ▼
                                                     HuggingFace Dataset Repo
```

### Phase 1: Backfill via GCS-backed Icechunk

For the initial 2000–2024 archive (~9000 days of IMERG data), use the proven
CMORPH pattern:

1. **GCS Icechunk store** — `icechunk.gcs_storage(bucket="...", prefix="imerg-v7-ea")` with service account
2. **Coiled cluster** — 10-20 workers download granules, subset to EA, return numpy arrays
3. **Coordinator writes** — coordinator receives arrays, writes to GCS-backed Icechunk, commits in batches
4. **Sync to HuggingFace** — after backfill completes, upload the GCS store contents to HF using `huggingface_hub.upload_folder()`

This leverages GCS's true object store compatibility with Icechunk for parallel,
high-throughput writes during the heavy backfill phase.

### Phase 2: Operational updates via local Icechunk + HF upload

For daily or weekly updates (7 days = ~7 files for daily, ~336 for half-hourly),
a Dask cluster is overkill. A single-process sequential pipeline is sufficient:

1. **Local Icechunk store** — `icechunk.local_filesystem_storage()`
2. **Sequential download** — `earthaccess.download()` for the new granules
3. **Read + write** — netcdf4 read, subset, write to local Icechunk, commit
4. **Upload to HuggingFace** — `huggingface_hub.upload_folder()` syncs the delta

**Why no cluster for updates?**
- 7 daily granules process in ~12 seconds sequentially
- 336 half-hourly granules (1 week) process in ~5 minutes sequentially
- Cluster startup overhead alone is 2-3 minutes
- No concurrent write coordination needed

### Why HuggingFace for final storage?

| Feature | HuggingFace | GCS |
|---------|-------------|-----|
| Storage cost | Free (TB-scale for public datasets) | $0.02/GB/month |
| Read access | Public, no auth needed | Requires credentials |
| Icechunk write support | No (git-LFS backend) | Yes (true object store) |
| Community visibility | Dataset cards, previews, downloads | None |
| Versioning | Git-based (can squash history) | Object versioning |

HuggingFace is ideal as the **distribution endpoint** — free, public, and
community-friendly. GCS serves as the **write-time backend** only during
heavy backfill operations.

## Test plan: Local Icechunk to HuggingFace with incremental updates

To validate that the local→HF pattern supports incremental updates (not just
one-shot uploads), run this test:

### Test 1: Initial week

```bash
# 1. Init local store with 2 weeks of time slots
uv run --python 3.12 imerg_daily_ea_icechunk.py init \
    --start-date 2024-12-01 --end-date 2024-12-14 \
    --local ./imerg_ea_update_test

# 2. Fill week 1
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --local ./imerg_ea_update_test --no-cluster

# 3. Verify week 1
uv run --python 3.12 imerg_daily_ea_icechunk.py verify \
    --local ./imerg_ea_update_test

# 4. Upload to HF
uv run --python 3.12 test_hf_icechunk.py \
    --store ./imerg_ea_update_test --upload
```

### Test 2: Incremental update (week 2)

```bash
# 5. Fill week 2 into the SAME local store
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-08 --end-date 2024-12-14 \
    --local ./imerg_ea_update_test --no-cluster

# 6. Verify both weeks present
uv run --python 3.12 imerg_daily_ea_icechunk.py verify \
    --local ./imerg_ea_update_test

# 7. Re-upload (should sync only changed/new chunks)
uv run --python 3.12 test_hf_icechunk.py \
    --store ./imerg_ea_update_test --upload
```

### What this validates

- Icechunk store survives multiple fill+commit cycles
- `huggingface_hub.upload_folder()` handles incremental uploads (only changed files)
- Week 1 data is preserved after week 2 fill
- HF store is readable after incremental update
- Commit history in Icechunk tracks each batch

## Summary

| Scenario | Backend | Cluster | Upload to HF |
|----------|---------|---------|--------------|
| Initial backfill (2000-2024) | GCS Icechunk | Coiled 10-20 workers | After completion |
| Daily/weekly updates | Local Icechunk | None (sequential) | After each update |
| Read access (consumers) | HuggingFace | N/A | N/A |

The key insight: **use the right tool for each phase**. GCS + Dask for heavy
writes, local + sequential for light updates, HuggingFace for free public
distribution. No need to force direct HF writes — it is not technically possible
and the two-phase approach is more robust.
