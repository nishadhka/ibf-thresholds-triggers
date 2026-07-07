# CMORPH → Icechunk (`cmorph-s3-nc`): upload record + retry status

**Store:** `gs://cpc_awc/icechunk/cmorph-s3-nc`
**Source (par):** `gs://cpc_awc/cmorph_catalog/catalog.parquet` (236,688 rows, one
per 30-min NetCDF, `kerchunk_refs` pre-computed) → virtual refs into
`s3://noaa-cdr-precip-cmorph-pds/` (anonymous).
**Builder / runner:** `build_cmorph_icechunk.py` / `backfill_cmorph_icechunk.py`
(GEFS-style, no Coiled; per-day → per-60-day-batch commits; single sequential
writer; manifest split along `time`; #884 OOM fix intact).
**SA:** `coiled-data-e4drr_202505.json`.

---

## What is in the store now (verified)

- **Coverage:** `1998-01-01 00:00 → 2020-05-31 23:30`, **7,887 days**,
  **378,576 half-hour timesteps**, monotonic & unique.
- **Variable:** `cmorph (time, lat=1649, lon=4948)`, int16, **native global
  0.08° grid**, chunk shape **`(1, 501, 1506)`**.
- **Read-back checked:** virtual S3 materialize 99.7–99.9% finite, 0–~100 mm
  (physical precip).

## Run history

| run | mode | result |
|---|---|---|
| initial | per-day commit | 626 days (1998-01 → 1999-09), then halted — per-day commit rewrote the growing 60-day manifest shard 60× → slowdown (9 s → 20 s/day) |
| batch | per-60-day commit (`--batch-days 60`) | 8.2 h, **121/154 batches OK**, **33 failed** → store reached 2020-05-31 with two gaps (below) |

Batching fixed the speed (flat ~149 s / 60-day batch ≈ **2.5 s/day**).

## The two gaps, and why "just retry" doesn't fill them

### 1. Structural — CMORPH re-chunked the grid from 2020-07 (28 batches)
CMORPH kept the same 1649×4948 grid but changed the **NetCDF chunk shape**:

| window | chunk shape |
|---|---|
| 1998-01 → **2020-06** | `(1, 501, 1506)` ← the store |
| 2020-07 → 2022-08 | `(1, 832, 2497)` |
| 2022-09 → 2023-05 | `(1, 825, 2474)` |
| 2023-06 → 2023-07 | `(1, 832, 2497)` |
| 2023-08 → 2024-12 | `(1, 825, 2474)` |

A Zarr array has **one** chunking, and a *virtual* ref must land on the array's
chunk grid, so files chunked `(832,2497)`/`(825,2474)` **cannot** be referenced
into the `(501,1506)` array (`ValueError: inconsistent chunk shapes … requires
ZEP003`). This is not retryable into this store. The robust builder now
**skips** these to `cmorph_skipped_<store>.json` instead of hard-failing.

**To include 2020-07 → present, one of:**
- **Separate group per chunk-regime in the same store** (e.g. `cmorph_832`,
  `cmorph_825`) — all data virtual, one store, but the time series is split by
  regime (consumer opens the relevant group). *Recommended.*
- **Separate store per regime** (`cmorph-s3-nc-832`, `…-825`).
- **Materialize-rechunk** the post-2020 files to `(501,1506)` — one continuous
  array, but copies data (no longer virtual).

### 2. Transient — 2004-04-26 → 2005-02-19 (5 batches, ~300 days)
These are `(501,1506)` (compatible) but died on a **transient S3 error during
`reconstruct`** that killed the whole 60-day batch (re-running the batch to a
fresh store now succeeds). The robust builder makes reconstruct **per-file
try/except**, so a transient error skips one file, not 60 days.

**But** the store already appended 2005-02-20 → 2020-05-31 *after* this window,
and Icechunk appends only at the **end** — a **mid-series gap cannot be
back-filled**. Filling 2004-2005 requires **rebuilding** the store (from scratch
or from 2004) in chronological order.

## Recommended path forward

1. **Rebuild `cmorph-s3-nc` clean** with the robust builder → continuous
   `1998-01 → 2020-06` `(501,1506)` store, transient errors self-heal per-file
   (~6–7 h, batch mode). This closes the 2004-2005 gap.
2. **Add the post-2020 regimes** as separate groups (`cmorph_832`, `cmorph_825`)
   in the same store (or separate stores) — covers 2020-07 → present.
3. Re-run is idempotent/resumable; a consumer reads `cmorph` (pre-2020-07) and
   the regime groups (post) and concatenates with a rechunk if a single series
   is needed.

## Commands

```bash
export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json

# clean rebuild of the (501,1506) era (fresh prefix keeps the current one intact)
uv run backfill_cmorph_icechunk.py --store gs://cpc_awc/icechunk/cmorph-s3-nc-v2 \
    --end 20200630 --batch-days 60

# post-2020 regime (once a per-group builder mode is added)
#   groups cmorph_832 (2020-07..2022-08, 2023-06..2023-07) and
#          cmorph_825 (2022-09..2023-05, 2023-08..present)
```

Skipped files are logged to `cmorph_skipped_<store>.json`.
