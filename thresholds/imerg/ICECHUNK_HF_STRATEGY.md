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

## Test results: Incremental Icechunk to HuggingFace (validated 2026-03-06)

The incremental update pattern was tested end-to-end with a 2-week window
(2024-12-01 to 2024-12-14), filling week 1, uploading to HF, then filling
week 2 and re-uploading. Store name: `test3_imerg-update-test`.

### Step-by-step results

| Step | Action | Result |
|------|--------|--------|
| 1 | Init 2-week template (Dec 1-14) | `(14, 350, 320)` — 14 days x 350 lat x 320 lon, 7.4s |
| 2 | Fill week 1 (Dec 1-7) | 7/7 granules → indices 0-6, 2.7s write time |
| 3 | Verify week 1 | 560,000/560,000 valid values, min=0.0, max=210.72, mean=1.99 mm/day |
| 4 | Upload week 1 to HF | 19 files, 2.5 KB → `E4DRR/icechunk-stores/test3_imerg-update-test` |
| 5 | Fill week 2 (Dec 8-14) | 7/7 granules → indices 7-13, resume detected 0-6 as done |
| 6 | Verify both weeks | 4 commits in history, all 14 days present |
| 7 | Re-upload to HF | 29 files, 9.1 KB — only new/changed chunks uploaded |

### Key findings

1. **Incremental fill works** — week 2 was written to correct template indices
   (7-13) without overwriting week 1 data. Date-based index mapping ensures
   granules land in the right time slots regardless of search query range.

2. **Resume detection works** — on the week 2 fill, the pipeline correctly
   identified indices 0-6 as already committed and only processed 7-13.

3. **HF incremental upload works** — `huggingface_hub.upload_folder()` only
   transferred new/changed files. Store grew from 19→29 files (2.5→9.1 KB)
   but HF only uploaded the delta, not the full store.

4. **Icechunk commit history preserved** — 4 commits tracked:
   ```
   fill batch 7-13: 7/7 OK
   fill batch 0-6: 7/7 OK
   initialize IMERG Daily EA template
   Repository initialized
   ```

5. **Week 1 data intact after week 2 fill** — verification confirmed all 14
   days have valid precipitation values after both fill cycles.

### Bug fixed during testing

The original fill logic used sequential indices (0, 1, 2...) based on
`enumerate(search_results)`, which meant a second fill with a different date
range would write to indices 0-6 again, overwriting week 1. Fixed by:

- Reading the template's time coordinate after opening the store
- Mapping each granule's date (from UMM metadata `BeginningDateTime`) to the
  correct template time index
- Handling timezone mismatch (granule dates are UTC-aware, template is tz-naive)
  with `tz_localize(None)`

### HF store location

```
https://huggingface.co/datasets/E4DRR/icechunk-stores/tree/main/test3_imerg-update-test
```

## Test plan commands (reproducible)

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
    --store ./imerg_ea_update_test --prefix test3_imerg-update-test --upload
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

# 7. Re-upload (only changed/new chunks transferred)
uv run --python 3.12 test_hf_icechunk.py \
    --store ./imerg_ea_update_test --prefix test3_imerg-update-test --upload
```

### Plot any 7-day window

```bash
# Plot specific date range from the store
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_update_test \
    --start-date 2024-12-05 --end-date 2024-12-11 \
    --geojson ea_ghcf_simple.geojson \
    --outdir ./plots
```

## GCS → HuggingFace transfer: rate limit issue (tested 2026-03-07)

### Problem

HuggingFace enforces a **128 commits per hour** rate limit on dataset repos.
The GCS Icechunk store (`gs://cpc_awc/ea_imerg_ic_store`) contains **12,158
objects (11.83 GB)** — mostly chunk files in the 0.5–2.0 MB range.

Two transfer approaches were tested:

| Approach | Method | Result |
|----------|--------|--------|
| Streaming GCS→HF | `create_commit()` with 50-file batches | 6,400/12,158 uploaded (52.6%), then **429 Too Many Requests** — hit 128 commits/hour |
| Local staging + `upload_large_folder()` | Download GCS→local (12 GB), then bulk upload | Requires 12 GB local disk; `upload_large_folder()` handles rate limits but still constrained by the same 128 commits/hour |

### Rate limit details

```
429 Too Many Requests: you have exceeded the rate limit for repository
commits (128 per hour). You can retry this action in 43 minutes.
```

- Free HF accounts: 128 commits/hour, 1000 API requests per 5-minute window
- Each batch commit counts as 1 commit regardless of file count
- With 50 files/commit: 128 × 50 = 6,400 files/hour → full upload needs ~2 hours
- With `upload_large_folder()`: handles rate limit waiting internally but same throughput

### Store size breakdown

| Subdirectory | Files | Size |
|-------------|-------|------|
| chunks | 11,239 | 11.80 GB |
| manifests | 307 | 30.5 MB |
| snapshots | 306 | 0.4 MB |
| transactions | 305 | 0.1 MB |
| refs | 1 | < 1 KB |

### Options for completing the transfer

1. **Wait and retry** — run `gcs_to_hf_transfer.py --skip-download` after the
   rate limit resets (~43 min); `upload_large_folder()` resumes from where it
   stopped. May need 2-3 hourly runs to complete all 12,158 files.

2. **HuggingFace Pro/paid plan** — higher rate limits for commits.

3. **Keep data on GCS only** — the store is fully accessible at
   `gs://cpc_awc/ea_imerg_ic_store` via Icechunk's `gcs_storage()`. HF upload
   is for free public distribution but not strictly required for analysis.

4. **Use `gsutil rsync` to a GCS-hosted HF-compatible endpoint** — not currently
   supported by Icechunk.

### Current status

The partial upload (6,400 files, 52.6%) is live at:
```
https://huggingface.co/datasets/E4DRR/icechunk-stores/tree/main/ea_imerg_ic_store
```
This is **not a functional Icechunk store** — missing chunks will cause read
errors. Either complete the upload or delete the partial data from HF.

## Summary

| Scenario | Backend | Cluster | Upload to HF | Validated |
|----------|---------|---------|--------------|-----------|
| Initial backfill (2000-2024) | GCS Icechunk | Coiled 10 workers | Partial (52.6%, rate limited) | Yes — 9,315 days, 11.83 GB |
| Daily/weekly updates | Local Icechunk | None (sequential) | After each update | Yes (test3) |
| Read access (consumers) | GCS Icechunk | N/A | N/A | Yes |

The key insight: **use the right tool for each phase**. GCS + Dask for heavy
writes, local + sequential for light updates, HuggingFace for free public
distribution. No need to force direct HF writes — it is not technically possible
and the two-phase approach is more robust.

**Note:** HuggingFace's 128 commits/hour rate limit makes large store uploads
slow but not impossible — plan for multi-hour transfer windows or use
`upload_large_folder()` which handles retries automatically.
