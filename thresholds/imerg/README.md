# IMERG Daily Early — East Africa Icechunk Pipeline

Downloads GPM IMERG Daily Early (GPM_3IMERGDE v07) precipitation granules,
subsets to East Africa using netcdf4-python, and stores in an Icechunk
repository. The store is uploaded to HuggingFace for free public distribution
([E4DRR/icechunk-stores](https://huggingface.co/datasets/E4DRR/icechunk-stores)).

## How it works

```
earthaccess.search_data()       # Search NASA GES DISC for IMERG Daily granules
        │
earthaccess.download()          # Download NetCDF4 granules via HTTPS
        │
netcdf4-python read + subset    # Open with nc4.Dataset, subset to EA bbox
        │
Icechunk store write            # Write to versioned Zarr-compatible store
        │
huggingface_hub upload          # Push store to HF dataset repo
```

### Why granule download, not OPeNDAP?

OPeNDAP was the initial approach but failed in practice:

- **Concurrent connection limits** — GES DISC OPeNDAP servers enforce strict
  per-user connection limits. Batch processing (even 3-5 concurrent reads)
  quickly triggers rate limiting, returning HTML error pages instead of data.
- **Unreliable for pipelines** — intermittent `"NetCDF: Access failure"` errors
  require `.netrc` + `.dodsrc` configuration with absolute paths (netcdf-c
  does not expand `~`), and even with correct auth, connections are fragile.
- **No advantage for subsetting** — IMERG daily files are ~27 MB each. Full
  download + local subset is faster than server-side OPeNDAP subsetting when
  factoring in connection overhead and retry logic.

Downloading granules via `earthaccess.download()` (HTTPS) is far more reliable
for pipelines — this is the same approach used in the CMORPH pipeline.

## Region

East Africa bounding box (same as CMORPH pipeline):
- Latitude: -12.0 to 23.0
- Longitude: 21.0 to 53.0

Countries: BDI, DJI, ERI, ETH, KEN, RWA, SDN, SOM, SSD, TZA, UGA

## Requirements

- Python >= 3.12 (icechunk requirement)
- [uv](https://docs.astral.sh/uv/) for dependency management
- NASA Earthdata account with GES DISC access enabled
- HuggingFace token (for upload)

### .env file

```
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
hf=hf_your_token_here
```

## Scripts

| Script | Purpose |
|--------|---------|
| `imerg_daily_ea_icechunk.py` | Main pipeline: init → fill → verify |
| `imerg_hh_ea_icechunk.py` | Half-hourly IMERG pipeline (GPM_3IMERGHH) with Coiled |
| `plot_imerg_ea.py` | Plot precipitation maps from local or HuggingFace store |
| `test_hf_icechunk.py` | Upload local store to HF, with safety checks |
| `download_imerg_daily.py` | Simple 7-day download script (standalone) |
| `ICECHUNK_HF_STRATEGY.md` | Architecture: why HF direct writes fail, GCS+HF strategy |

All scripts use PEP 723 inline metadata — `uv run --python 3.12` handles
dependencies automatically. No separate `pip install` needed.

## Usage

### 1. Initialize empty template

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py init \
    --start-date 2024-12-01 --end-date 2024-12-14 \
    --local ./imerg_ea_local
```

### 2. Fill with data (sequential, no cluster)

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --local ./imerg_ea_local --no-cluster
```

Fill supports **incremental updates** — run again with a different date range
to add more data to the same store:

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-08 --end-date 2024-12-14 \
    --local ./imerg_ea_local --no-cluster
```

Granules are mapped to the correct template time index by date, so order and
range of fill calls does not matter. Resume detection skips already-committed
indices.

### 3. Fill with Coiled cluster (production)

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2000-06-01 --end-date 2024-12-31 \
    --local ./imerg_ea_local --n-workers 20
```

### 4. Verify

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py verify \
    --local ./imerg_ea_local
```

### 5. Upload to HuggingFace

```bash
uv run --python 3.12 test_hf_icechunk.py \
    --store ./imerg_ea_local \
    --prefix test3_imerg-update-test \
    --upload
```

Safety checks block upload if the store directory contains `.env`, `.netrc`,
`.py`, or `.ipynb` files.

### 6. Plot from local store

```bash
# Plot all days
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_local \
    --geojson ea_ghcf_simple.geojson \
    --outdir ./plots

# Plot specific 7-day window
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_local \
    --start-date 2024-12-05 --end-date 2024-12-11 \
    --geojson ea_ghcf_simple.geojson \
    --outdir ./plots
```

### 7. Plot from HuggingFace store

```bash
uv run --python 3.12 plot_imerg_ea.py \
    --hf-repo E4DRR/icechunk-stores \
    --hf-prefix test3_imerg-update-test \
    --start-date 2024-12-05 --end-date 2024-12-11 \
    --geojson ea_ghcf_simple.geojson \
    --outdir ./plots
```

This downloads the store from HF via `snapshot_download`, caches it locally,
opens with Icechunk, and plots the selected date range.

## Icechunk + HuggingFace storage strategy

Direct Icechunk writes to HuggingFace are **not possible** — HF's git-LFS
backend does not support the atomic object operations Icechunk needs. See
[ICECHUNK_HF_STRATEGY.md](ICECHUNK_HF_STRATEGY.md) for full analysis.

The two-phase approach:

| Scenario | Write backend | Cluster | Then upload to HF |
|----------|---------------|---------|-------------------|
| Backfill (2000-2024) | GCS Icechunk | Coiled 10-20 workers | After completion |
| Daily/weekly updates | Local Icechunk | None (sequential) | After each fill |
| Read access | HuggingFace | N/A | N/A |

## Dask/Coiled worker pattern

The fill pipeline follows a **coordinator-worker fork-merge pattern**
(same as `cmorph_east_africa_icechunk.py`):

```
Coordinator (local machine)
  ├── earthaccess.search_data() → granule list
  ├── for each batch:
  │     ├── earthaccess.download() → local NetCDF files
  │     ├── submit _read_imerg_file_ea() to Dask workers
  │     ├── collect numpy arrays from workers
  │     ├── write to Icechunk session
  │     └── commit batch
  └── close cluster
```

The half-hourly pipeline (`imerg_hh_ea_icechunk.py`) uses a worker-side
download pattern where each Coiled worker downloads its own granule via
authenticated HTTPS, reads with netcdf4, subsets, and returns numpy arrays
to the coordinator.

### Resume support

Fill commands support resume. Commit messages encode batch ranges
(`fill batch 7-13: 7/7 OK`). On restart, completed time indices are
detected and skipped automatically.

## Verified test results

### Test 1: Single week (1-week fill + upload)

```
Init:   (7, 350, 320) — 7 days x 350 lat x 320 lon
Fill:   7/7 granules, 0 failures, 12s
Verify: 560,000/560,000 valid values
        Min: 0.0000, Max: 210.7200, Mean: 1.9891 mm/day
Upload: 19 files, 2.5 KB → E4DRR/icechunk-stores/test1_imerg-v7-ea-store
```

### Test 2: Half-hourly with Coiled (1 day, 5 workers)

```
Init:   (48, 350, 320) — 48 half-hours x 350 lat x 320 lon
Fill:   48/48 granules on Coiled cluster (5 workers, n2-standard-4)
Upload: → E4DRR/icechunk-stores/test2_imerg-v7-ea-store
```

### Test 3: Incremental update (2 weeks, fill in 2 batches)

This test validates that the local→HF pattern supports incremental updates.
Store: `test3_imerg-update-test`.

| Step | Action | Result |
|------|--------|--------|
| 1 | Init 2-week template (Dec 1-14) | `(14, 350, 320)`, 7.4s |
| 2 | Fill week 1 (Dec 1-7) | 7/7 → indices 0-6, 2.7s |
| 3 | Verify week 1 | 560k valid values, mean 1.99 mm/day |
| 4 | Upload to HF | 19 files, 2.5 KB |
| 5 | Fill week 2 (Dec 8-14) | 7/7 → indices 7-13, resume skipped 0-6 |
| 6 | Verify both weeks | 4 commits, all 14 days present |
| 7 | Re-upload to HF | 29 files, 9.1 KB (only delta uploaded) |
| 8 | Plot from HF store (Dec 5-11) | 7 PNGs spanning both weeks |

Key findings:
- Incremental fill writes to correct template indices by date mapping
- `huggingface_hub.upload_folder()` only transfers changed/new chunks
- Icechunk commit history preserved across fill cycles
- HF store readable after incremental update

## HuggingFace stores

| Store | Product | Period | Status |
|-------|---------|--------|--------|
| `test1_imerg-v7-ea-store` | IMERG Daily Early | 2024-12-01 to 2024-12-07 | 1-week test |
| `test2_imerg-v7-ea-store` | IMERG HH Final | 2024-12-01 (48 HH) | 1-day test |
| `test3_imerg-update-test` | IMERG Daily Early | 2024-12-01 to 2024-12-14 | Incremental update test |

All at: https://huggingface.co/datasets/E4DRR/icechunk-stores

## Way forward

1. **Full archive backfill (2000-06 to present)**
   - Use GCS-backed Icechunk with Coiled cluster (10-20 workers)
   - Upload completed store to HF
   - See [ICECHUNK_HF_STRATEGY.md](ICECHUNK_HF_STRATEGY.md) for architecture

2. **Operational daily/weekly updates**
   - Sequential `--no-cluster` fill (7 days in ~12s)
   - Upload delta to HF after each fill
   - No Dask cluster needed for small updates

3. **Rechunk for time-series access**
   - Add a `rechunk` subcommand (like CMORPH pipeline)
   - Target pencil chunks: full-time x 5-lat x 5-lon for fast point queries

4. **IMERG Final product**
   - Switch from `GPM_3IMERGDE` (Early) to `GPM_3IMERGDF` (Final) for research
   - Final has ~3.5 month latency but better quality

5. **Integration with thresholds pipeline**
   - Use the Icechunk store as input for GEV return period analysis
   - Replace Planetary Computer source in `imerg_pc_source.py`
