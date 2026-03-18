# IMERG East Africa — Icechunk Pipeline

Processes GPM IMERG precipitation data (Daily Early + Half-Hourly Final),
subsets to East Africa, and stores in Icechunk repositories on GCS and
HuggingFace ([E4DRR/icechunk-stores](https://huggingface.co/datasets/E4DRR/icechunk-stores)).



  ┌─────────┬───────────────┬────────────┬──────────────────────────────────┐
  │ Product │  Short name   │  Latency   │             Quality              │
  ├─────────┼───────────────┼────────────┼──────────────────────────────────┤
  │ Early   │ GPM_3IMERGHHE │ ~4 hours   │ Near-real-time, lower quality    │
  ├─────────┼───────────────┼────────────┼──────────────────────────────────┤
  │ Late    │ GPM_3IMERGHHL │ ~14 hours  │ Intermediate                     │
  ├─────────┼───────────────┼────────────┼──────────────────────────────────┤
  │ Final   │ GPM_3IMERGHH  │ ~3.5       │ Research-quality,                │
  │         │               │ months     │ gauge-calibrated                 │
  └─────────┴───────────────┴────────────┴──────────────────────────────────┘
  
## Data access methods tested

Three approaches were tested for accessing IMERG data from NASA GES DISC.
Each failed at scale until the THREDDS ncml aggregation approach was found.

### 1. OPeNDAP (individual granules) — failed

Single-granule OPeNDAP access via netcdf4-python:
- GES DISC enforces strict concurrent connection limits (even 3-5 reads)
- Returns HTML error pages instead of data when rate limited
- Requires `.netrc` + `.dodsrc` with absolute paths (netcdf-c doesn't expand `~`)
- Unreliable for batch pipelines

### 2. HTTPS granule download (`earthaccess.download()`) — failed at scale

Download full HDF5 files via authenticated HTTPS, read locally with netcdf4:
- Works well for small batches (5 workers, ~50 granules)
- **Fails with 20 workers**: GES DISC rate limiting returns truncated files
  or HTML error pages. Workers crash with `NetCDF: HDF error` or SIGSEGV
  (signal 11) when trying to open corrupt downloads
- Each IMERG HH file is ~10 MB × 48 files/day × 20 workers = heavy load

### 3. S3 direct access — blocked from GCP

IMERG data is on S3 (`s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/`
in us-west-2). `earthaccess.get_s3_credentials(daac='GES_DISC')` returns valid
temporary creds, but:
- IAM role `s3-same-region-access-role` has an **explicit deny** for out-of-region
- Only works from AWS us-west-2 instances — blocked from GCP by policy
- See https://data.gesdisc.earthdata.nasa.gov/s3credentialsREADME

### 4. THREDDS ncml daily aggregation (current) — works

GES DISC provides THREDDS ncml aggregation files that bundle all 48 half-hourly
granules per day into a single OPeNDAP-accessible dataset:

```
https://gpm1.gesdisc.eosdis.nasa.gov/thredds/dodsC/aggregation/
  GPM_3IMERGHH.07/{year}/GPM_3IMERGHH.07_Aggregation_{year}{doy}.ncml.ncml
```

- **Server-side subsetting**: xarray sends EA bbox, server returns only the subset
- **~550 KB per day** transferred vs ~480 MB for 48 raw global granules
- **365 requests/year** instead of ~17,500 granule downloads
- Works from GCP — no S3 credentials or AWS region restrictions
- Each Coiled worker reads one day, returns `(48, 400, 345)` numpy array
- Available from 1998 to present

## Architecture

```
                    PRODUCTION PIPELINE (THREDDS → GCS → HF)
                    ─────────────────────────────────────────
THREDDS ncml (GES DISC)                              GCS Icechunk
  ├── day1.ncml ──► Worker 1 ──subset──► numpy ──┐   gs://cpc_awc/
  ├── day2.ncml ──► Worker 2 ──subset──► numpy ──┤   ea_imerg_ic_store
  ├── day3.ncml ──► Worker 3 ──subset──► numpy ──┤        │
  └── ...          (5-10 Coiled workers)          └──► Coordinator
                                                      writes + commits
                                                           │
                                                      sync to HF
                                                           ▼
                                                  HuggingFace Dataset Repo
```

## Region

East Africa bounding box (expanded):
- Latitude: -14.5 to 25.5
- Longitude: 19.5 to 54.0

Countries: BDI, DJI, ERI, ETH, KEN, RWA, SDN, SOM, SSD, TZA, UGA

## Requirements

- Python >= 3.12 (icechunk requirement)
- [uv](https://docs.astral.sh/uv/) for dependency management
- NASA Earthdata account with GES DISC access enabled
- GCS service account file (`coiled-data-e4drr_202505.json`)
- HuggingFace token (for upload to HF)

### .env file

```
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
hf=hf_your_token_here
```

## Scripts

| Script | Purpose |
|--------|---------|
| `imerg_hh_gcs_icechunk.py` | **Production**: THREDDS → Coiled → GCS Icechunk (Final/Late/Early) |
| `imerg_daily_ea_icechunk.py` | Daily Early pipeline: init → fill → verify (local/HF) |
| `imerg_hh_ea_icechunk.py` | Half-hourly HTTPS download pipeline (local/HF) |
| `plot_imerg_ea.py` | Plot precipitation maps from local or HuggingFace store |
| `plot_imerg_gcs_sample.py` | Plot random days from GCS Icechunk store |
| `check_missing_days.py` | Scan GCS store for missing/empty days |
| `gcs_to_hf_transfer.py` | Transfer GCS store to HuggingFace (rate-limit aware) |
| `gcs_to_source_coop_transfer.py` | Transfer GCS store to source.coop S3 (batched, ~12 GB) |
| `test_hf_icechunk.py` | Upload local store to HF, with safety checks |
| `download_imerg_daily.py` | Simple 7-day download script (standalone) |
| `ICECHUNK_HF_STRATEGY.md` | Architecture: why HF direct writes fail, GCS+HF strategy |

All scripts use PEP 723 inline metadata — `uv run --python 3.12` handles
dependencies automatically.

## Usage: THREDDS → GCS pipeline (production)

### 1. Init template on GCS

```bash
# Test: 1 month
uv run --python 3.12 imerg_hh_gcs_icechunk.py init \
    --start-date 2024-12-01 --end-date 2024-12-31 \
    --gcs-prefix test_ea_imerg_ic_store

# Production: full archive (Jun 2000 – Dec 2025)
uv run --python 3.12 imerg_hh_gcs_icechunk.py init \
    --start-date 2000-06-01 --end-date 2025-12-01
```

### 2. Fill via THREDDS with Coiled workers

```bash
# Test: 1 month, 10 workers
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-31 \
    --gcs-prefix test_ea_imerg_ic_store --n-workers 10

# Production: full archive (supports resume — re-run to fill gaps)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2000-06-01 --end-date 2025-12-01 \
    --n-workers 10 --commit-batch 30

# Sequential (no cluster, for debugging)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --gcs-prefix test_ea_imerg_ic_store --no-cluster
```

### 3. Verify

```bash
uv run --python 3.12 imerg_hh_gcs_icechunk.py verify \
    --gcs-prefix test_ea_imerg_ic_store

# Production store
uv run --python 3.12 imerg_hh_gcs_icechunk.py verify
```

### 4. Monitor backfill progress

The fill command logs to `imerg_hh_gcs_icechunk.log`. Monitor with:

```bash
# Watch live progress
tail -f imerg_hh_gcs_icechunk.log

# Count committed batches and failures
grep "Committed batch" imerg_hh_gcs_icechunk.log | wc -l
grep "FAILED" imerg_hh_gcs_icechunk.log | wc -l

# See latest committed batch and progress
grep "Committed batch" imerg_hh_gcs_icechunk.log | tail -3
grep "done)" imerg_hh_gcs_icechunk.log | tail -1

# Show batches with partial failures
grep "failed)" imerg_hh_gcs_icechunk.log | grep -v "0 failed"

# List all failed days (for targeted re-fill)
grep "FAILED" imerg_hh_gcs_icechunk.log | sed 's/.*Day \(.*\) FAILED.*/\1/' | sort -u
```

Coiled dashboard (requires `bokeh>=3.1.0`):

```bash
pip install "bokeh>=3.1.0"
# Dashboard URL printed at cluster startup in logs
grep "Cluster ready" imerg_hh_gcs_icechunk.log
```

## Usage: Daily Early pipeline (local/HF)

### Init + fill + verify + upload

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py init \
    --start-date 2024-12-01 --end-date 2024-12-14 \
    --local ./imerg_ea_local

uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --local ./imerg_ea_local --no-cluster

uv run --python 3.12 imerg_daily_ea_icechunk.py verify \
    --local ./imerg_ea_local

uv run --python 3.12 test_hf_icechunk.py \
    --store ./imerg_ea_local --prefix test3_imerg-update-test --upload
```

Supports incremental fills — run fill again with different dates, resume
detection skips completed indices.

## Plotting

```bash
# From local store (all days)
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_local --geojson ea_ghcf_simple.geojson

# From local store (specific 7-day window)
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_local \
    --start-date 2024-12-05 --end-date 2024-12-11 \
    --geojson ea_ghcf_simple.geojson --outdir ./plots

# From HuggingFace store
uv run --python 3.12 plot_imerg_ea.py \
    --hf-repo E4DRR/icechunk-stores \
    --hf-prefix test3_imerg-update-test \
    --start-date 2024-12-05 --end-date 2024-12-11 \
    --geojson ea_ghcf_simple.geojson --outdir ./plots
```

## Icechunk + HuggingFace storage strategy

Direct Icechunk writes to HuggingFace are **not possible** — HF's git-LFS
backend does not support the atomic object operations Icechunk needs. See
[ICECHUNK_HF_STRATEGY.md](ICECHUNK_HF_STRATEGY.md) for full analysis.

| Scenario | Write backend | Cluster | Then upload to HF |
|----------|---------------|---------|-------------------|
| Backfill (2000-2025) | GCS Icechunk | Coiled 5-10 workers | After completion |
| Daily/weekly updates | Local Icechunk | None (sequential) | After each fill |
| Read access | HuggingFace | N/A | N/A |

## Verified test results

### Test 1: Daily Early — single week fill + HF upload

```
Init:   (7, 350, 320) — 7 days x 350 lat x 320 lon
Fill:   7/7 granules, 0 failures, 12s
Verify: 560,000 valid values, mean 1.99 mm/day
Upload: → E4DRR/icechunk-stores/test1_imerg-v7-ea-store
```

### Test 2: Half-hourly — Coiled HTTPS download (5 workers, 1 day)

```
Init:   (48, 350, 320) — 48 half-hours x 350 lat x 320 lon
Fill:   48/48 granules on Coiled (5 workers) — worked with small batch
Upload: → E4DRR/icechunk-stores/test2_imerg-v7-ea-store
```

### Test 3: Incremental update — 2 weeks, 2 fill batches, HF delta upload

| Step | Action | Result |
|------|--------|--------|
| 1 | Init 2-week template | `(14, 350, 320)`, 7.4s |
| 2 | Fill week 1 (Dec 1-7) | 7/7 → indices 0-6 |
| 3 | Upload to HF | 19 files, 2.5 KB |
| 4 | Fill week 2 (Dec 8-14) | 7/7 → indices 7-13, resume skipped 0-6 |
| 5 | Re-upload to HF | 29 files, 9.1 KB (only delta) |
| 6 | Plot from HF (Dec 5-11) | 7 PNGs spanning both weeks |

### Test 4: GCS + HTTPS download (20 workers, 1 month) — failed

```
Init:   (1488, 400, 345) on gs://cpc_awc/test_ea_imerg_ic_store — OK
Fill:   20 workers downloading via HTTPS — rate limited
        Batch 96-191 committed (96 granules OK)
        Other batches: HDF errors, truncated files, worker segfaults
        Workers killed by signal 11 (SIGSEGV in netCDF4 on corrupt files)
Root cause: 20 concurrent HTTPS downloads overwhelm GES DISC servers
```

### Test 5: THREDDS ncml aggregation — verified locally

```
URL:    gpm1.gesdisc.eosdis.nasa.gov/thredds/dodsC/aggregation/
        GPM_3IMERGHH.07/2024/GPM_3IMERGHH.07_Aggregation_2024336.ncml.ncml
Result: xr.open_dataset() → server-side subset → (48, 345, 400) loaded
        6,624,000 valid values, min=0.0, max=41.39, mean=0.087 mm/hr
        Works from GCP, no rate limiting on single connection
```

### Test 6: THREDDS + GCS + Coiled (10 workers, 1 month) — passed

```
Init:   (1488, 400, 345) on gs://cpc_awc/test_ea_imerg_ic_store
Fill:   31 days via THREDDS ncml, 10 Coiled workers (5-10 adaptive)
        31/31 OK, 0 failures, ~5 min data + 3 min cluster startup
Verify: 6,624,000 valid values, min=0.0, max=41.39, mean=0.087 mm/hr
Fixes:  Transpose THREDDS (time, lon, lat) → (time, lat, lon) for template
        Resume detection stops at latest init commit
```

### S3 direct access test — blocked from GCP

```
Creds:  earthaccess.get_s3_credentials(daac='GES_DISC') → valid STS token
Bucket: s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/
Error:  AccessDenied — role s3-same-region-access-role has explicit deny
        for out-of-region requests (only works from AWS us-west-2)
```

## HuggingFace stores

| Store | Product | Period | Status |
|-------|---------|--------|--------|
| `test1_imerg-v7-ea-store` | IMERG Daily Early | 2024-12-01 to 2024-12-07 | 1-week test |
| `test2_imerg-v7-ea-store` | IMERG HH Final | 2024-12-01 (48 HH) | 1-day test |
| `test3_imerg-update-test` | IMERG Daily Early | 2024-12-01 to 2024-12-14 | Incremental test |

GCS stores:

| Store | Product | Period | Shape | Status |
|-------|---------|--------|-------|--------|
| `gs://cpc_awc/ea_imerg_ic_store` | IMERG HH Final | 2000-06-01 to 2025-12-01 | `(447120, 400, 345)` | Backfill in progress |
| `gs://cpc_awc/test_ea_imerg_ic_store` | IMERG HH Final | 2024-12-01 to 2024-12-31 | `(1488, 400, 345)` | Complete (test) |

All HF stores at: https://huggingface.co/datasets/E4DRR/icechunk-stores

## Production backfill status

**Store**: `gs://cpc_awc/ea_imerg_ic_store`
**Template**: `(447,120, 400, 345)` — 9,315 days × 48 HH × 400 lat × 345 lon = ~230 GB

### Run 1 — no retries (2026-03-06)

```
Workers:   5 adaptive (Coiled, n2-standard-4, us-east1)
Batches:   85 committed, 1628 failures
Result:    ~2,550 days committed (many batches dropped due to failures)
Duration:  ~3.5 hours
Issue:     No retry logic — THREDDS DAP errors caused entire batches to be dropped
           Server returns HTML error pages when rate-limited (NASA IT Security Banner)
```

### Run 2 — with retries + partial commits (2026-03-07, in progress)

```
Workers:   5-10 adaptive (Coiled, n2-standard-4, us-east1)
Batches:   100+ committed, ~300 failures
Result:    Filling gaps from run 1 + continuing through 2025
Duration:  ~8+ hours estimated
Fixes:     5 retries with exponential backoff + jitter per worker
           Partial batch commits (save successful days even if some fail)
           Resume detection stops at latest init (ignores stale pre-init commits)
```

### Backfill resilience

The pipeline handles transient THREDDS failures gracefully:

- **Worker retries**: 5 attempts with exponential backoff (1s, 2s, 4s, 8s + jitter)
- **Partial commits**: if 27/30 days succeed, those 27 are committed; 3 failures retry on resume
- **Resume detection**: re-run the same fill command to pick up failed days
- **Idempotent**: safe to run multiple times — already-filled days are skipped

After the initial backfill, run fill again to pick up any remaining gaps:

```bash
# Resume — automatically skips committed days
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2000-06-01 --end-date 2025-12-01 \
    --n-workers 10 --commit-batch 30
```

## Multi-product quality cascade (Final → Late → Early)

The store supports all three IMERG products in a single Icechunk repository.
Each product fills the same time slots but at different quality levels:

```
  Product    Short name       Latency    Quality   Priority
  ─────────  ───────────────  ─────────  ────────  ────────
  Final      GPM_3IMERGHH     ~3.5 mo    Highest   3
  Late       GPM_3IMERGHHL    ~14 hrs    Medium    2
  Early      GPM_3IMERGHHE    ~4 hrs     Lowest    1
```

### How it works

- **Commit tagging**: each fill commit includes the product tag:
  `fill batch 123-456 [GPM_3IMERGHHE]: 30/30 OK`
- **Quality-aware resume**: when filling with a product, days already filled
  by the same or higher quality product are skipped. Days filled by a lower
  quality product are overwritten.
- **Iterative replacement**: as Final data becomes available (~3.5 months
  after observation), re-run fill with `--product final` to replace
  Late/Early data with research-quality Final data.

### Usage: bring the store up to today

```bash
# Step 1: Extend the template to cover through today
uv run --python 3.12 imerg_hh_gcs_icechunk.py extend \
    --end-date 2026-03-08

# Step 2: Fill with Late data (available up to ~14 hours ago)
#   Only fills days not already covered by Final
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2025-12-02 --end-date 2026-03-06 \
    --product late --no-cluster --commit-batch 7

# Step 3: Fill remaining days with Early data (up to ~4 hours ago)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2026-03-07 --end-date 2026-03-08 \
    --product early --no-cluster --commit-batch 1

# Step 4 (later): When Final data is released, replace Late/Early
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2025-12-02 --end-date 2026-03-08 \
    --product final --no-cluster --commit-batch 30
```

### Icechunk commit history example

```
fill batch 447168-447216 [GPM_3IMERGHHE]: 1/1 OK     ← Early for today
fill batch 446784-447120 [GPM_3IMERGHHL]: 7/7 OK     ← Late for last week
fill batch 0-1440 [GPM_3IMERGHH]: 30/30 OK           ← Final for 2000-06
initialize IMERG HH EA template
```

When `--product final` is re-run later, it overwrites the Early/Late slots
because Final (quality=3) > Late (quality=2) > Early (quality=1). Days
already filled by Final are skipped.

## Way forward

1. **Backfill complete** — 9,315 days (2000-06 to 2025-12) filled with Final data

2. **Extend to present** — use `extend` + `fill --product late/early` to
   bring the store from 2025-12-01 up to today

3. **Upload to HuggingFace** — rate-limited to 128 commits/hour; use
   `gcs_to_hf_transfer.py` with `upload_large_folder()` for automatic
   rate limit handling (see `ICECHUNK_HF_STRATEGY.md`)

4. **Operational updates** — daily `--no-cluster` fill with Early/Late,
   periodic Final replacement as data becomes available

5. **Rechunk for time-series access**
   - Add a `rechunk` subcommand (like CMORPH pipeline)
   - Target pencil chunks: full-time x 5-lat x 5-lon for fast point queries

6. **Integration with thresholds pipeline**
   - Use the Icechunk store as input for GEV return period analysis

## source.coop (public S3 access)

The IMERG HH store is published to [source.coop](https://source.coop) at:

```
s3://us-west-2.opendata.source.coop/e4drr-project/observations/imerg_hh_icechunk/
```

### Read access (no credentials needed)

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/observations/imerg_hh_icechunk",
    region="us-west-2",
    anonymous=True,
)
repo = icechunk.Repository.open(
    storage, config=icechunk.RepositoryConfig.default()
)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```

### Transfer from GCS to source.coop

The store is ~12 GB / 12,000+ objects. `gcs_to_source_coop_transfer.py`
downloads in batches (default 2 GB each) to work within temporary credential
time limits:

```bash
# Set credentials in .env or environment
export SOURCE_COOP_ACCESS_KEY_ID=...    # or AWS_ACCESS_KEY_ID
export SOURCE_COOP_SECRET_ACCESS_KEY=... # or AWS_SECRET_ACCESS_KEY
export SOURCE_COOP_SESSION_TOKEN=...     # or AWS_SESSION_TOKEN

# Dry run — show batches and sizes
uv run gcs_to_source_coop_transfer.py --dry-run

# Full transfer (download + upload in batches)
uv run gcs_to_source_coop_transfer.py

# Resume upload only (skip download)
uv run gcs_to_source_coop_transfer.py --skip-download

# Verify uploaded store
uv run gcs_to_source_coop_transfer.py --verify

# Custom batch size
uv run gcs_to_source_coop_transfer.py --batch-gb 1.5
```
