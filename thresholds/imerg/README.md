# IMERG East Africa — Icechunk Pipeline

Processes GPM IMERG precipitation data (Daily Early + Half-Hourly Final),
subsets to East Africa, and stores in Icechunk repositories on GCS and
HuggingFace ([E4DRR/icechunk-stores](https://huggingface.co/datasets/E4DRR/icechunk-stores)).

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
  └── ...          (20 Coiled workers)           └──► Coordinator
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
| `imerg_hh_gcs_icechunk.py` | **Production**: THREDDS → Coiled → GCS Icechunk |
| `imerg_daily_ea_icechunk.py` | Daily Early pipeline: init → fill → verify (local/HF) |
| `imerg_hh_ea_icechunk.py` | Half-hourly HTTPS download pipeline (local/HF) |
| `plot_imerg_ea.py` | Plot precipitation maps from local or HuggingFace store |
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

# Production: full archive
uv run --python 3.12 imerg_hh_gcs_icechunk.py init \
    --start-date 2000-06-01 --end-date 2025-03-01
```

### 2. Fill via THREDDS with Coiled workers

```bash
# Test: 1 month, 20 workers
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-31 \
    --gcs-prefix test_ea_imerg_ic_store --n-workers 20

# Sequential (no cluster, for debugging)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --gcs-prefix test_ea_imerg_ic_store --no-cluster
```

### 3. Verify

```bash
uv run --python 3.12 imerg_hh_gcs_icechunk.py verify \
    --gcs-prefix test_ea_imerg_ic_store
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
| Backfill (2000-2025) | GCS Icechunk | Coiled 20 workers | After completion |
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

GCS store: `gs://cpc_awc/test_ea_imerg_ic_store` (Dec 2024 HH template)

All HF stores at: https://huggingface.co/datasets/E4DRR/icechunk-stores

## Way forward

1. **Test THREDDS + GCS pipeline** (next step)
   - Run `imerg_hh_gcs_icechunk.py` fill with THREDDS + 20 Coiled workers
   - 1-month test on `test_ea_imerg_ic_store`, then full 2000-2025 archive

2. **Full archive backfill (2000-06 to present)**
   - ~9,500 days × 48 HH steps = ~456,000 timesteps
   - 20 workers reading THREDDS, coordinator writing to GCS Icechunk
   - Upload completed store to HF

3. **Operational daily/weekly updates**
   - Sequential `--no-cluster` fill via THREDDS (1 day in ~5s)
   - Upload delta to HF

4. **Rechunk for time-series access**
   - Add a `rechunk` subcommand (like CMORPH pipeline)
   - Target pencil chunks: full-time x 5-lat x 5-lon for fast point queries

5. **Integration with thresholds pipeline**
   - Use the Icechunk store as input for GEV return period analysis
