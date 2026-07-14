# CMORPH Precipitation — ARCO Pipeline

Cloud-native processing of NOAA CDR CMORPH 30-min precipitation data
(1998-2024) for East Africa.

There are **two** Icechunk stores here; pick by what you need:

| store | what it is | use it when |
|---|---|---|
| `gs://cpc_awc/icechunk/cmorph-s3-nc-v2` | **Global, virtual** (refs into NOAA's S3; no data copied). Full native 0.08° grid, 1998-01-01 → 2024-12-31. | You need any region / the whole globe, or the full native record. |
| `gs://cpc_awc/icechunk/cmorph_ea_*` | **East Africa, materialized + pencil-chunked.** | You need fast repeated point/time-series reads over EA (0.5 s for a 27-yr series). |

The rest of this README covers the EA materialized pipeline. The global virtual
store is summarised next, and fully documented in
[`CMORPH_ICECHUNK_S3_NC_UPLOAD.md`](CMORPH_ICECHUNK_S3_NC_UPLOAD.md).

---

## Global virtual store (`cmorph-s3-nc-v2`) — complete

All **236,688** catalog rows converted (100%), 1998-01-01 → 2024-12-31,
9,862 days / 473,376 half-hour steps. Verified: monotonic, no duplicates,
virtual S3 reads 99.9–100% finite.

CMORPH re-chunked its NetCDFs twice over the record. A zarr array has exactly
**one** chunking and a *virtual* ref must land on the array's chunk grid, so each
chunk-regime lives in its **own group** of the one store:

| group | window(s) | chunk shape |
|---|---|---|
| `/` (root) | 1998-01-01 → 2020-06-30 | `(1, 501, 1506)` |
| `cmorph_832` | 2020-07-01 → 2022-08-31, 2023-06-01 → 2023-07-31 | `(1, 832, 2497)` |
| `cmorph_825` | 2022-09-01 → 2023-05-31, 2023-08-01 → 2024-12-31 | `(1, 825, 2474)` |

The two post-2020 groups **overlap in wall-clock time** (the regimes interleave)
but their day sets are **disjoint** — every day is in exactly one group.

### `read_cmorph_v2.py` — all three eras as ONE series, in one command

The group split is a *storage* constraint, not a data one: `lat`/`lon` are
identical across groups, so on read they concat cleanly along `time` into a
single continuous series. Verified across both era boundaries, including the
interleaved 2023 `cmorph_825`→`cmorph_832` switch.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json

# East Africa, one season, daily totals -> NetCDF   (the intended use)
uv run read_cmorph_v2.py --bbox -16 26 18 56 \
    --start 2024-03-01 --end 2024-05-31 --daily --out ea_mam2024.nc

# a point, one month
uv run read_cmorph_v2.py --point -1.29 36.82 --start 2024-03-01 --end 2024-03-31 --daily

# what would a given request cost? (fetches nothing)
uv run read_cmorph_v2.py --bbox -16 26 18 56 --estimate
```

From Python: `from read_cmorph_v2 import open_cmorph; ds = open_cmorph(bbox=(-16,26,18,56), start=..., end=...)`

`cmorph` is a **rate in mm/hr** on 30-min steps; `--daily` converts to mm/day
totals. Longitudes are stored 0–360 but −180…180 input is accepted.

#### ⚠️ Cost — this store is chunked SPATIALLY, so cost scales with TIME

A chunk spans a large lat/lon tile but only **one** 30-min step (root =
`(1, 501, 1506)`). So:

    S3 fetches ≈ n_timesteps × n_chunks_intersecting_your_region

A point is **not** cheaper per step than a small box — it just wastes more of
each chunk. Measured on this VM (96 fetches → 155 s; 384 → 249 s), i.e.
**~124 s fixed + ~0.33 s per fetch**:

| request | fetches | ETA | |
|---|---:|---|---|
| EA box, 8 days | 384 | ~4 min | ok |
| EA box, one season | 4,416 | ~26 min | ok — the intended use |
| full record at a point | 473,376 | **~43 h** | ❌ |
| full record, EA box | 867,792 | **~78 h** (~1.1 TB out) | ❌ |

**Use this store for: any region over a bounded window** (days → a season).
**For multi-year series at a point/small box, use the materialized,
pencil-chunked EA store** (`cmorph_east_africa_icechunk.py`, chunked
`(473376, 5, 5)`) — it does a full 27-year point series in ~0.5 s. That is
exactly the tradeoff the two stores exist to cover.

`--estimate` costs a request without fetching anything; reads projected over
30 min are **refused** unless you pass `--force`. Note `--daily` collapses 48
steps/day, so a season *scans* 10.7 GB but only *writes* 0.2 GB.

A single continuous *stored* array across all regimes would need a
materialize-rechunk (copies data, no longer virtual) — not done.

### Published on source.coop (public, anonymous)

Mirrored to
**https://source.coop/e4drr-project/observations/s3-noaa-cdr-cmorph-icechunk-vd**
(1,410 objects, 163 MB — it is *virtual*, so the ~1.1 TB of actual data stays in
NOAA's public bucket). Verified: opens anonymously and the virtual refs resolve.

```bash
source .env    # source.coop STS creds (1-hour tokens)
uv run mirror_cmorph_to_source_coop.py               # copy + verify (resumable)
uv run mirror_cmorph_to_source_coop.py --verify-only # just read the published store
```

Reading it needs **no credentials at all**:

```python
import icechunk, xarray as xr

storage = icechunk.s3_storage(
    bucket="e4drr-project",
    prefix="observations/s3-noaa-cdr-cmorph-icechunk-vd",
    endpoint_url="https://data.source.coop",
    region="us-east-1",
    force_path_style=True,     # source.coop is NOT virtual-host addressed
    anonymous=True)
auth = icechunk.containers_credentials(
    {"s3://noaa-cdr-precip-cmorph-pds/": icechunk.s3_anonymous_credentials()})
repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
store = repo.readonly_session("main").store

ds = xr.open_zarr(store, consolidated=False)                     # root era
ds_832 = xr.open_zarr(store, group="cmorph_832", consolidated=False)
```

> ⚠️ **Two gotchas, both of which will bite you with a DNS error:**
> 1. `force_path_style=True` is required — the default virtual-host style
>    resolves `e4drr-project.data.source.coop`, which does not exist.
> 2. **Do not have `AWS_ENDPOINT_URL` set in your environment when reading.**
>    If `.env` is sourced, icechunk applies that endpoint to the *virtual-chunk*
>    S3 client too and sends NOAA reads to
>    `noaa-cdr-precip-cmorph-pds.data.source.coop`. Unset the `AWS_*` vars (both
>    reads are anonymous, so you need none of them). `mirror_cmorph_to_source_coop.py`
>    does this isolation internally.

### Building / extending it

```bash
STORE=gs://cpc_awc/icechunk/cmorph-s3-nc-v2
uv run backfill_cmorph_icechunk.py --store $STORE --end 20200630 --batch-days 60  # root era
bash run_post2020_groups.sh                                                       # post-2020 groups
```

Both are **resumable and idempotent** — they read each group's existing `time`
axis and build only what's missing.

---

## Current Status

| Phase | Status | Detail |
|-------|--------|--------|
| Catalog build | Done | 236K files, 224 MB Parquet |
| Catalog on GCS | Done | `gs://cpc_awc/cmorph_catalog/catalog.parquet` |
| `init` | Done | Template store on GCS (3.5s) |
| `fill` | **Done** | 9,862 days, 473,376 timesteps, 0 failures, ~2.5 hrs |
| `rechunk` | **Done** | Pencil chunks (473376, 5, 5), 8,536 chunks, 7.9 min |

**Pipeline complete.** All four phases finished. The pencil-chunked store
supports instant time-series access (0.5s for full 27-year series at a point).

## Working Pipeline

### 1. `cmorph_parquet_vds_catalog.py` — Build Parquet VDS Catalog (DONE)

Discovers CMORPH NetCDF files on S3, virtualizes them on Coiled workers,
and streams Kerchunk refs into a single Parquet catalog. Each row = one
file with full metadata + Kerchunk JSON.

```bash
micromamba run -n aifs-etl python cmorph_parquet_vds_catalog.py catalog \
    --start-year 1998 --end-year 2024 --n-workers 10
```

Output: `cmorph_vds_catalog/catalog.parquet` (~236K files, ~224 MB)

### 2. `upload_merge_parquet_catalogs.py` — Merge + Upload Catalog (DONE)

Merges partial Parquet catalogs and/or uploads to GCS for Coiled workers.

```bash
# Merge partial catalogs
micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py merge \
    --inputs catalog_1998_2000.parquet catalog_2001_2024.parquet \
    --output catalog.parquet

# Upload to GCS (workers read from here during fill)
micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py upload \
    --input cmorph_vds_catalog/catalog.parquet

# Verify catalog completeness
micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py verify \
    --input cmorph_vds_catalog/catalog.parquet
```

### 3. `cmorph_east_africa_icechunk.py` — EA Subset + Pencil Rechunk

Four-phase pipeline to create a materialized Icechunk store for East
Africa (lat -12 to 23, lon 21 to 53), then rechunk for time-series access.

| Phase | Subcommand | What it does | Status |
|-------|-----------|--------------|--------|
| 1 | `init` | Create empty template store (metadata only, ~5s) | Done |
| 2 | `fill` | Populate via Coiled + obstore S3 reads (9,862 days, 0 failures) | Done |
| 3 | `rechunk` | Rechunk to pencil chunks (473376, 5, 5) via P2P shuffle, 7.9 min | Done |
| 4 | `verify` | Inspect any store (Icechunk or Zarr) | Working |

```bash
# Phase 1: Init (DONE)
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py init \
    --catalog cmorph_vds_catalog/catalog.parquet \
    --gcs-prefix cmorph_ea_subset

# Phase 2: Fill — resumable, auto-detects completed batches
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py fill \
    --catalog cmorph_vds_catalog/catalog.parquet \
    --target-gcs-prefix cmorph_ea_subset \
    --n-workers 20 --commit-batch 50

# Phase 3: Rechunk to pencil chunks (P2P shuffle)
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py rechunk \
    --source-gcs-prefix cmorph_ea_subset \
    --target-path gs://cpc_awc/cmorph_ea_pencil --n-workers 20

# Verify any store
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py verify \
    --gcs-prefix cmorph_ea_subset
```

### How `fill` works

1. Coordinator reads lightweight catalog index (datetime, year, month, day)
2. Groups files into day-batches (24 files = 48 timesteps per day)
3. Submits consecutive batches of `--commit-batch` days to Coiled workers
4. Each worker: reads S3 URLs from GCS Parquet catalog (predicate pushdown,
   ~24 rows per day) → downloads NetCDF via `obstore.get()` → opens with
   `netcdf4` engine → `isel(lat, lon)` for EA subset → returns numpy array
5. Coordinator writes all results in the batch to Icechunk with
   `to_zarr(region=)`, commits only when the full batch succeeds
6. Moves to the next contiguous batch

### Resume and batch sizing

Fill is **resumable** — you can stop and restart at any time, and it picks
up exactly where it left off. This works because:

1. Each committed batch has a structured message: `"fill days 1035-1084: 50/50 OK"`
2. On startup, the script reads the Icechunk commit history (version control)
3. It finds the highest `d_end` from all `"fill days X-Y"` commits
4. Resumes from `d_end + 1`, skipping all completed batches

**How it looks in practice:**

```
# First run — process days 0-1034 with 5 workers, then stop (Ctrl+C or kill)
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py fill \
    --catalog cmorph_vds_catalog/catalog.parquet \
    --target-gcs-prefix cmorph_ea_subset --n-workers 5 --commit-batch 5

# Icechunk commit history now has:
#   fill days 0-4: 5/5 OK
#   fill days 5-9: 5/5 OK
#   ...
#   fill days 1030-1034: 5/5 OK

# Second run — resume with 20 workers and larger batches
micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py fill \
    --catalog cmorph_vds_catalog/catalog.parquet \
    --target-gcs-prefix cmorph_ea_subset --n-workers 20 --commit-batch 50

# Output:
#   Resuming from day 1035 (days 0-1034 done)
#   Remaining: 8827 days
#   Batch: days 1035-1084 (50 days, 0/8827 done so far)
#   ...continues from where it left off...
```

The resume is safe because only **fully succeeded** batches are committed.
If a batch has any failure, it is NOT committed — the entire batch will be
retried on the next run. This guarantees every committed range has complete
data with no gaps in the time series.

You can freely change `--n-workers` and `--commit-batch` between runs.
The only thing that persists is the Icechunk commit history in the GCS store.

**Batch size** (`--commit-batch`) controls the trade-off:

| Batch size | Commit frequency | Idle worker time | Failure blast radius |
|-----------|-----------------|-----------------|---------------------|
| 5 | Every ~10s | High (~40%) | Low — retry 5 days |
| **50** | **Every ~30s** | **Low (~15%)** | **Medium — retry 50 days** |
| 200 | Every ~2min | Minimal (~5%) | High — retry 200 days |

With `--commit-batch 50 --n-workers 20`:
- Each batch: 50 days submitted, ~3 days per worker
- Workers process in parallel (~5s per day)
- Coordinator writes 50 results + commits (~5s)
- ~30s per batch, ~100 batches/hour, ~1.5 hours for full dataset

### How `rechunk` works

Rechunks the materialized EA Icechunk store from spatial-first chunks
`(48, 120, 110)` to pencil chunks `(473376, 5, 5)` for fast time-series access.

1. Opens the source Icechunk store using `service_account_key` (JSON string,
   not file path) so GCS credentials survive pickle serialization to Dask workers
2. Configures Dask P2P rechunking (`array.rechunk.method: p2p`, fusion disabled)
3. Launches a **fixed-size** Coiled cluster (P2P cannot handle adaptive scaling)
4. Rechunks with `ds.chunk({"time": -1, "lat": 5, "lon": 5})` — Dask creates a
   P2P shuffle network where workers exchange data directly (constant memory)
5. Writes to a new plain Zarr store on GCS with pencil chunk encoding

**Key fix**: `service_account_file=` (local path) causes a Rust panic on Dask
workers because the file doesn't exist on remote VMs.  `service_account_key=`
(JSON string) embeds credentials in the storage config, surviving pickle.

**Performance**: 372 GB dataset rechunked in **7.9 minutes** with 20 workers.

**Result**: Full 27-year time series at any point loads in **0.5 seconds**
(single 45 MB chunk) vs minutes with the original spatial chunks.

| Access Pattern | Spatial chunks (48, 120, 110) | Pencil chunks (473376, 5, 5) |
|---------------|------------------------------|------------------------------|
| Spatial snapshot (1 timestep) | **Fast** — 1 chunk (2.4 MB) | Slow — 8,536 chunks |
| Time series (27yr, 1 point) | Slow — ~10K chunks | **Fast** — 1 chunk (45 MB, 0.5s) |

### `read_ea_icechunk.py` — Read the EA store as xarray Dataset

Opens the materialized EA Icechunk store and demonstrates access patterns:

```bash
# Inspect metadata + fill progress
micromamba run -n aifs-etl python read_ea_icechunk.py

# Load a spatial snapshot (precipitation map at one time)
micromamba run -n aifs-etl python read_ea_icechunk.py --snapshot 2005-06-15T12:00

# Load a time series at a point (e.g. Nairobi: -1.29, 36.82)
micromamba run -n aifs-etl python read_ea_icechunk.py --timeseries --lat -1.29 --lon 36.82

# Use a local store
micromamba run -n aifs-etl python read_ea_icechunk.py --local /path/to/store
```

### `test_fill_local.py` — Local validation

Tests the exact worker pipeline locally (no Coiled needed):

```bash
# Test default day (1998-01-17)
micromamba run -n aifs-etl python test_fill_local.py

# Test a specific day
micromamba run -n aifs-etl python test_fill_local.py --year 2000 --month 6 --day 15
```

Validates: obstore S3 download → tempfile → netcdf4 → EA subset → numpy
concat → shape/dtype/data checks.

## Past Attempts (in `not_working/`)

Several approaches were tried before arriving at the current pipeline:

1. **Direct VirtualiZarr concatenation** (`cmorph_concatenated_icechunk.py`) —
   OOM at scale: coordinator accumulated all virtual datasets in memory.

2. **Batch processing on branches** (`cmorph_multi_year_processor.py`,
   `cmorph_s3_to_gcs_icechunk_parallel.py`) — Wrote batches to separate
   Icechunk branches, but merging branches with virtual refs failed.

3. **Two-phase concat** (`cmorph_two_phase_concat.py`) — Month-by-month
   `append_dim` approach. Worked for small tests but OOM on full dataset
   because `to_icechunk(append_dim)` loads the entire manifest each call.

4. **Subprocess-isolated Icechunk writer** (`icechunk_month_worker.py` +
   icechunk subcommand in catalog script) — Ran each month in a subprocess
   to avoid OOM. Successfully wrote a 3-year virtual store, but the virtual
   refs approach cannot spatially subset without materializing full global
   chunks from S3.

5. **fsspec reference filesystem** — Used Kerchunk refs to open files via
   `fsspec("reference", ...)`. Failed on Coiled workers with
   `"Reference-FS's target filesystem must have same value of asynchronous"`.
   Replaced by direct `obstore.get()` → tempfile → netcdf4 approach.

**Key lesson**: Virtual refs point to full global NetCDF chunks — spatial
subsetting requires materializing real data. The current approach (Parquet
catalog + direct S3 reads via obstore + materialized Icechunk store) avoids
all the OOM, virtual-ref, and fsspec async limitations.

## Cost Analysis: Cloud-Native Dataset Generation at Scale

### The problem

NOAA's CMORPH archive on S3 contains **236,688 NetCDF files** spanning 1998-2024
at 30-minute, 8 km global resolution.  The raw global archive is roughly
**~400 GB on S3** (each file ~1.7 MB, global grid 1649 lat × 4948 lon).
A fully materialized global Zarr/Icechunk store at the same resolution would
be **~15 TB** (473K timesteps × 8.16M grid cells × 4 bytes).

For East Africa analysis, we only need 2.6% of the spatial domain (481 × 439
out of 1649 × 4948 grid cells) but the full 27-year temporal range.  The
traditional approach — download all files, convert, subset locally — would
require downloading hundreds of GB and hours of single-machine processing.

### What we built instead

A three-phase cloud-native pipeline that generates a **372 GB analysis-ready
Icechunk store** for East Africa from the raw S3 archive, plus a **pencil-chunked
Zarr store** for time-series access, for approximately **$6 in total compute cost**.

### Cost breakdown

| Phase | What it does | Workers | Duration | Cost |
|-------|-------------|---------|----------|------|
| 1. Parquet catalog | VirtualiZarr on 236K files → Kerchunk refs in Parquet | 20 | ~1.5 hrs | ~$2.50 |
| 2. Fill (materialize) | Read S3 → spatial subset → write to Icechunk | 5-20 | ~2.5 hrs | ~$3.00 |
| 3. Rechunk (P2P) | Pencil chunks (473376, 5, 5) for time-series access | 20 | 8 min | ~$0.30 |
| **Total** | **236K files → 372 GB EA store + pencil rechunk** | | **~4.5 hrs** | **~$5.80** |

Workers are Coiled-managed GCP VMs at ~$0.03/worker-hour (spot pricing in
us-east1).  Fill uses `n2-standard-4`, rechunk uses `n2-highmem-4`.

### Why this is economically viable

**1. No data egress costs**

Workers run in `us-east1` (same region as NOAA's S3 bucket).  S3 → EC2/GCE
transfers within the same region are free.  We read ~400 GB from S3 during
the fill phase without paying a cent in egress.

**2. Parquet catalog as a lightweight index (224 MB vs 400 GB)**

The Parquet VDS catalog captures the full metadata and Kerchunk references
for all 236K files in a single 224 MB file.  This enables:

- **Predicate pushdown**: Workers read only their day's rows (~24 rows, ~2 MB)
  from the 224 MB catalog, not the entire thing
- **Reusable index**: The catalog is built once and reused for any number of
  downstream operations (fill, verify, future re-processing)
- **No coordinator memory pressure**: The catalog streams to disk via PyArrow,
  never accumulating all refs in memory

**3. Selective materialization — only read what you need**

Each worker downloads only the spatial subset it needs:

```
Global CMORPH file:  1649 lat × 4948 lon × 2 time = 1.7 MB
EA subset per file:   481 lat ×  439 lon × 2 time = 1.7 MB (reads full file, subsets in memory)
```

While we download the full file (~1.7 MB), we only retain and write the EA
portion.  The alternative — storing a virtual Icechunk with pointers to S3
— fails because virtual refs point to full global chunks and cannot be
spatially subsetted without materialization.

**4. Coiled spot instances keep costs minimal**

Coiled automatically provisions GCP Spot VMs, which cost 60-80% less than
on-demand pricing.  A 20-worker cluster of `n2-standard-4` VMs costs roughly
$0.60/hour.  The entire pipeline uses ~10 worker-hours total.

**5. Icechunk versioning eliminates reprocessing**

The Icechunk store has full Git-like version control.  If new CMORPH data
becomes available (e.g., 2025 onward), we can resume the fill from the last
committed day — no need to reprocess the existing 9,862 days.  Each commit
is atomic and the store is always in a consistent state.

### Comparison with alternatives

| Approach | Time | Cost | Limitations |
|----------|------|------|-------------|
| Download all + local convert | Days | $50+ (egress + storage) | Single machine OOM, no versioning |
| Virtual Icechunk (no materialization) | 2 hrs | $2.50 | Cannot spatially subset virtual refs |
| Rechunker library | Hours | $5-10 | Requires intermediate storage, no resume |
| **This pipeline** | **5 hrs** | **~$7** | **Resumable, versioned, cloud-native** |

### The economics at scale

This approach scales linearly with the spatial domain:

| Region | Lat × Lon | Store size | Est. cost |
|--------|-----------|-----------|-----------|
| East Africa (current) | 481 × 439 | 400 GB | ~$7 |
| All of Africa | 960 × 960 | ~1.7 TB | ~$25 |
| Global | 1649 × 4948 | ~15 TB | ~$150 |
| Global + rechunk | 1649 × 4948 | ~15 TB | ~$200 |

For any region, the Parquet catalog (Phase 1) is built once and reused.
Only the fill and rechunk phases scale with the output size.

### Key insight

**The cost of generating analysis-ready, cloud-optimized datasets is now
so low that there is no reason to work with raw NetCDF files directly.**

For $7 and 5 hours of wall time, we transformed 236,688 individual NetCDF
files on S3 into a single, versioned, chunk-optimized Icechunk store that
supports:

- Instant spatial snapshots (load one timestep of precipitation map)
- Fast time-series access at any point (after pencil rechunking)
- Git-like version history for every write
- Resume from interruption at any point
- Append new data without reprocessing

This makes VirtualiZarr/Kerchunk + Parquet catalogs + Icechunk the
economically viable path to analysis-ready cloud-optimized datasets at
any scale.

### GCS storage costs

| Configuration | Size | Monthly | Daily | Yearly |
|--------------|------|---------|-------|--------|
| Both stores (EA + pencil) + catalog | 745 GB | $14.90 | $0.50 | $179 |
| Pencil store only (delete EA source) | 373 GB | $7.45 | $0.25 | $89 |

GCS Standard storage in us-east1: $0.020/GB/month.

The EA Icechunk source store (`cmorph_ea_subset`) can be deleted once the
pencil store is validated, since the pencil store contains the same data
and the source can be regenerated from the Parquet catalog + S3 if needed.
The EA source is only needed for spatial-snapshot workloads (one timestep,
full map).

## Next: Return Period Analysis (Extreme Value)

### Why pencil chunks are perfect for this

The return period analysis requires the **full time series at each pixel** —
exactly what pencil chunks `(473376, 5, 5)` provide.  Each pencil chunk
contains all 27 years of 30-minute data for a 5×5 spatial block:

```
One pencil chunk = 473,376 timesteps × 5 lat × 5 lon × 4 bytes = 45 MB
Reading it: 0.5 seconds from GCS
Contains: 27 years of precipitation at 25 pixels
```

The 8,536 pencil chunks are **independent units of work** — each can be
processed without reading any other chunk.  This makes the analysis
embarrassingly parallel with zero cross-chunk communication.

### Pipeline design

Based on `not_working/phase0_extreme_value_test.py` (Phase 0 test with
153 days of pseudo-years), adapted for the full 27-year pencil-chunked
store with real annual maxima.

**Stage 1: Rolling accumulation + annual maxima (Coiled workers)**

Each worker reads one pencil chunk and computes:

```python
# For each duration (30min, 1hr, 3hr, 6hr, 12hr, 24hr, 48hr, 72hr, 7day):
#   1. Rolling cumulative sum along time axis (numpy cumsum trick)
#   2. Split into 27 calendar years
#   3. Extract annual maximum for each pixel
#
# Input:  (473376, 5, 5) float32 = 45 MB
# Output: (9 durations × 27 years × 5 × 5) float32 = 121 KB
```

Memory per worker: ~45 MB data + ~400 MB rolling buffers = ~500 MB peak.
Well within n2-standard-4 (16 GB).

**Stage 2: Distribution fitting (local, no cluster needed)**

For each pixel × duration, fit a distribution to the 27 annual maxima:

```python
# Methods: Normal, Gumbel, GEV (Generalized Extreme Value)
# Input:  27 annual maxima per pixel per duration
# Output: return period precipitation depths for T = 2, 5, 10, 20, 50, 100 yr
```

Output dataset: `(9 durations × 6 return periods × 481 lat × 439 lon)` =
~46 MB.  This fits in memory easily and runs in seconds.

**Stage 3: Write to Icechunk**

Store the return period precipitation map as a versioned Icechunk dataset
on GCS.  Tiny output (~46 MB) — single commit.

### Estimated cost

| Stage | Workers | Time | Cost |
|-------|---------|------|------|
| Annual maxima (8,536 chunks) | 20 | ~2 hrs | ~$1.20 |
| Distribution fit (local) | 0 | ~30s | $0 |
| **Total** | | **~2 hrs** | **~$1.20** |

### Key differences from Phase 0 test

| Aspect | Phase 0 (old) | Full analysis (new) |
|--------|--------------|-------------------|
| Data source | Old Icechunk store (153 days) | Pencil Zarr (27 years) |
| Annual maxima | Pseudo-years (30-day blocks) | Real calendar years |
| Spatial extent | 20×20 pixel test | Full EA (481×439) |
| Read pattern | Workers open Icechunk with temp creds | Workers read Zarr via gcsfs |
| Chunk alignment | Unaligned — read across chunks | **Perfect** — one chunk = one task |

### Implementation plan

```bash
# 1. Run full EA return period analysis
micromamba run -n aifs-etl python cmorph_return_periods.py compute \
    --source gs://cpc_awc/cmorph_ea_pencil \
    --output-gcs-prefix cmorph_ea_return_periods \
    --n-workers 20

# 2. Verify results
micromamba run -n aifs-etl python cmorph_return_periods.py verify \
    --gcs-prefix cmorph_ea_return_periods

# 3. Read return period map
micromamba run -n aifs-etl python cmorph_return_periods.py plot \
    --gcs-prefix cmorph_ea_return_periods \
    --duration 24hr --return-period 100
```

## Dask Operations Notes

See `DASK_OPERATIONS_NOTES.md` for detailed analysis of:

- Why workers sit idle during sequential-batch fill (barrier pattern)
- P2P rechunking for Phase 3 (constant memory shuffle)
- Fill vs rechunk: different Dask patterns

## Environment

All scripts require the `aifs-etl` micromamba environment:

```bash
micromamba run -n aifs-etl python <script.py>
```

## GCS Stores

| Store | Description |
|-------|-------------|
| `gs://cpc_awc/cmorph_catalog/catalog.parquet` | Parquet catalog on GCS (for workers) |
| `gs://cpc_awc/cmorph_ea_subset` | Materialized EA Icechunk, chunks (48, 120, 110), 372 GB |
| `gs://cpc_awc/cmorph_ea_pencil` | Pencil-chunked EA Zarr, chunks (473376, 5, 5), 372 GB |
| `gs://cpc_awc/cmorph_ea_return_periods` | Return period analysis output (planned) |
