# IMERG Daily Early — East Africa Icechunk Pipeline

Streams GPM IMERG Daily Early (GPM_3IMERGDE v07) precipitation data,
subsets to East Africa, and stores in an Icechunk repository on
HuggingFace ([E4DRR/icechunk-stores/imerg-v7-ea-store](https://huggingface.co/datasets/E4DRR/icechunk-stores)).

## How it works

```
earthaccess.search_data()       # Search NASA GES DISC for IMERG Daily granules
        │
earthaccess.download()          # Download NetCDF4 granules (NOT OPeNDAP)
        │
netcdf4-python read + subset    # Open with nc4.Dataset, subset to EA bbox
        │
Icechunk store write            # Write to versioned Zarr-compatible store
        │
huggingface_hub upload          # Push store to HF dataset repo
```

**Why not OPeNDAP?** GES DISC OPeNDAP servers enforce strict concurrent connection
limits. Batch processing quickly hits rate limits returning HTML error pages.
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
| `plot_imerg_ea.py` | Plot daily precipitation maps with country borders |
| `test_hf_icechunk.py` | Test HF upload and evaluate Dask worker pattern |
| `download_imerg_daily.py` | Simple 7-day download script (standalone) |

## Usage

All scripts use PEP 723 inline metadata — `uv run` handles dependencies automatically.

### 1. Initialize empty template

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py init \
    --start-date 2000-06-01 --end-date 2024-12-31 \
    --local ./imerg_ea_local
```

### 2. Fill with data

**Local/testing (sequential, no cluster):**
```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2024-12-01 --end-date 2024-12-07 \
    --local ./imerg_ea_local --no-cluster
```

**Production (Coiled cluster):**
```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py fill \
    --start-date 2000-06-01 --end-date 2024-12-31 \
    --local ./imerg_ea_local --n-workers 20
```

### 3. Verify

```bash
uv run --python 3.12 imerg_daily_ea_icechunk.py verify \
    --local ./imerg_ea_local
```

### 4. Upload to HuggingFace

```bash
uv run --python 3.12 test_hf_icechunk.py --store ./imerg_ea_local --upload
```

### 5. Plot daily maps

```bash
uv run --python 3.12 plot_imerg_ea.py \
    --store ./imerg_ea_local \
    --geojson ea_ghcf_simple.geojson \
    --outdir ./plots
```

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

### Current limitation

In cluster mode, the coordinator downloads files locally, but Coiled workers
can't access the coordinator's filesystem. **For production runs, the fill
function needs one of these fixes:**

| Option | Approach | Tradeoff |
|--------|----------|----------|
| A | Workers download their own granules via `earthaccess` | Requires auth propagation to workers |
| B | Coordinator stages files on GCS, workers read from GCS | Most robust, adds GCS cost |
| C | Sequential `--no-cluster` mode | Works now, slower for large runs |

**`--no-cluster` mode works correctly today** and is recommended for runs
up to ~1 year of data. For the full 2000-2024 archive, Option A or B should
be implemented.

### Resume support

The fill command supports resume. Commit messages encode batch ranges
(`fill batch 0-29: 30/30 OK`). On restart, it skips completed batches
automatically.

## Verified test run (1 week)

```
Init:   (7, 350, 320) — 7 days × 350 lat × 320 lon
Fill:   7/7 granules, 0 failures, 12s
Verify: 560,000/560,000 valid values
        Min: 0.0000, Max: 210.7200, Mean: 1.9891 mm/day
Upload: 19 files, 2.55 MB → E4DRR/icechunk-stores/imerg-v7-ea-store
```

## Way forward

1. **Full archive backfill (2000-06 to present)**
   - Run init + fill with `--no-cluster` in batches (e.g., 1 year at a time)
   - Or implement worker-side download (Option A) for Coiled parallelism

2. **Operational daily updates**
   - Scheduled fill with rolling 7-day window
   - Resume support handles gaps automatically

3. **Rechunk for time-series access**
   - Add a `rechunk` subcommand (like CMORPH pipeline)
   - Target pencil chunks: full-time × 5-lat × 5-lon for fast point queries

4. **IMERG Final product**
   - Switch from `GPM_3IMERGDE` (Early) to `GPM_3IMERGDF` (Final) for research
   - Final has ~3.5 month latency but better quality

5. **Integration with thresholds pipeline**
   - Use the Icechunk store as input for GEV return period analysis
   - Replace Planetary Computer source in `imerg_pc_source.py`
