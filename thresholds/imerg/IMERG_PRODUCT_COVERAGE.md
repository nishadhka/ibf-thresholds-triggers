# IMERG Product Coverage Analysis

**Store**: `gs://cpc_awc/ea_imerg_ic_store`
**Date**: 2026-03-19
**Shape**: `(451827, 400, 345)` — 9,413 days × 48 half-hourly × 400 lat × 345 lon

## IMERG product hierarchy

NASA GPM IMERG releases the same precipitation data at three quality tiers,
each with a different latency:

| Product | Short name | Latency | Quality | Use case |
|---------|-----------|---------|---------|----------|
| **Final** | `GPM_3IMERGHH` | ~3.5 months | Research-grade, gauge-calibrated | Climate research, long-term analysis |
| **Late** | `GPM_3IMERGHHL` | ~14 hours | Intermediate | Operational monitoring |
| **Early** | `GPM_3IMERGHHE` | ~4 hours | Near-real-time, lower quality | Early warning, nowcasting |

All three products cover the same spatial grid and time slots. The pipeline
fills the store with the highest-quality product available and replaces
lower-quality data as better products are released.

## How the commit batch mechanism works

The `fill` command in `imerg_hh_gcs_icechunk.py` groups days into batches
and creates **one Icechunk commit per batch**. The batch size is controlled
by `--commit-batch` (default: 10 days).

### Fill loop (simplified)

```python
COMMIT_BATCH = args.commit_batch  # default 10 days per commit

for batch_start in range(0, len(remaining), COMMIT_BATCH):
    batch = remaining[batch_start : batch_start + COMMIT_BATCH]
    batch_idx_min = batch[0]["t_start_idx"]     # first time index
    batch_idx_max = batch[-1]["t_start_idx"]    # last time index

    # Open a fresh writable session for each batch
    session = target_repo.writable_session("main")

    for day in batch:
        # Download day from THREDDS (via Coiled worker or locally)
        result = worker_read_thredds_day(day)

        # Write into the pre-allocated time slots
        ds_write = xr.Dataset({"precipitation": (("time","lat","lon"), result["data"])})
        ds_write.to_zarr(
            session.store,
            region={"time": slice(t_start, t_start + 48)},  # 48 HH steps
        )

    # One commit per batch — tagged with product and index range
    session.commit(
        f"fill batch {batch_idx_min}-{batch_idx_max} "
        f"[{product_short_name}]: {ok}/{total} OK"
    )
```

### Commit message format

Each commit records three things:

```
fill batch 0-1440 [GPM_3IMERGHH]: 30/30 OK
         │    │         │            │
         │    │         │            └── success count / batch size
         │    │         └── product short name (Final/Late/Early)
         │    └── time index range (each day = 48 steps)
         └── first time index of batch
```

Examples:
```
fill batch 0-1440 [GPM_3IMERGHH]: 30/30 OK        ← 30 days Final
fill batch 446784-447120 [GPM_3IMERGHHL]: 7/7 OK   ← 7 days Late
fill batch 447168-447216 [GPM_3IMERGHHE]: 1/1 OK   ← 1 day Early
```

**Key design choices:**

- **Not monthly** — commits are by batch size (default 10 days), not calendar
  month. A 30-day `--commit-batch 30` run produces one commit per ~month, but
  the default `--commit-batch 10` produces ~3 commits per month.
- **Partial commits** — if 27/30 days in a batch succeed and 3 fail (THREDDS
  errors), the 27 successful days are still committed. The commit message
  shows `27/30 OK`. Failed days can be retried on the next run.
- **Index range, not dates** — the commit stores time dimension indices
  (e.g., `0-1440`), not calendar dates. Each day occupies 48 indices
  (half-hourly). Index 0 = first HH of 2000-06-01, index 48 = first HH
  of 2000-06-02, etc.
- **Old commits** — early backfill runs before multi-product support did not
  include the `[GPM_3IMERG*]` tag. These are assumed Final (quality=3).

### Quality-aware resume

When re-running `fill`, the pipeline walks the commit history backwards
(newest → oldest), stops at the `"initialize ..."` commit, and builds a
map of `{time_index → max_quality_level}`:

```python
for commit in repo.ancestry(branch="main"):
    if commit.message.startswith("initialize "):
        break  # stop — everything before init is stale
    if commit.message.startswith("fill batch "):
        # Parse product tag → quality level
        # Parse index range → mark those indices as filled
        filled_quality[idx] = max(filled_quality.get(idx, 0), quality)
```

Then for each day to fill, it checks:

- `existing_quality >= current_quality` → **skip** (already same or better)
- `existing_quality < current_quality` → **overwrite** (upgrade quality)

Quality levels:
- `Final (quality=3)` — never overwritten
- `Late (quality=2)` — overwritten by Final only
- `Early (quality=1)` — overwritten by Late or Final
- `Empty (quality=0)` — filled by any product

This allows progressive upgrades: fill with Early for near-real-time access,
then replace with Late, then with Final as NASA releases each product.

## How the pipeline writes data (init vs fill)

Understanding the two-step write process is key to interpreting coverage:

1. **`init`** — creates the template store with all zeros for every time slot
   and commits `"initialize IMERG HH EA template"`. At this point the store
   has the correct shape but **no real data** — every pixel is zero.

2. **`fill`** — downloads actual precipitation from NASA THREDDS, writes it
   into the pre-allocated time slots, and commits with a tagged message:
   `"fill batch 0-1440 [GPM_3IMERGHH]: 30/30 OK"`.

Only days that have a `fill batch` commit contain real downloaded data.
Days without one still hold the init zeros — they were never filled.

## How the gaps were identified

### Approach 1: Data presence scan (`check_monthly_coverage.py`)

Loads one half-hourly timestep per day and checks `any(sample != 0)`:

```bash
uv run --python 3.12 check_monthly_coverage.py \
    --sa-file coiled-data-e4drr_202503.json --list-missing
```

Result: 9,411 / 9,413 days reported as "filled." **This is misleading.**
It cannot distinguish between init zeros (template, no real data) and
genuinely filled days — both can appear as non-zero due to floating-point
noise, or a filled day could be all-zero if precipitation was truly zero
everywhere. This check is a rough first pass, not authoritative.

### Approach 2: Commit history analysis (`check_product_coverage.py`) — authoritative

Parses every Icechunk commit message to determine exactly which days were
filled and by which product. This is the **ground truth** because:

- The `init` commit is clearly separated — the parser stops when it hits
  `"initialize ..."` and does not count init zeros as filled
- Each `fill batch` commit records the exact time-index range and the
  product tag (`[GPM_3IMERGHH]`, `[GPM_3IMERGHHL]`, or `[GPM_3IMERGHHE]`)
- Days with no `fill batch` commit covering their time index → truly Empty

```bash
uv run --python 3.12 check_product_coverage.py \
    --sa-file coiled-data-e4drr_202503.json

# Export per-day CSV
uv run --python 3.12 check_product_coverage.py \
    --sa-file coiled-data-e4drr_202503.json --csv product_coverage.csv
```

### Why the two checks disagree

| Check | Method | Empty days reported |
|-------|--------|-------------------|
| `check_monthly_coverage.py` | Non-zero pixel scan | 2 |
| `check_product_coverage.py` | Commit history parse | 198 |

The pixel scan over-counts "filled" by ~196 days because init zeros can
have floating-point noise or NaN values that pass the `any(sample != 0)`
test. The commit-based check correctly identifies these 198 days as never
having been filled by any `fill` command — they still hold template data
from the `init` step.

**Use `check_product_coverage.py` as the authoritative source** for both
coverage and product quality.

## Coverage before gap-fill (2026-03-19)

| Product | Days | % |
|---------|------|---|
| Final | 9,120 | 96.9% |
| Late | 95 | 1.0% |
| Early | 0 | 0.0% |
| Empty | 198 | 2.1% |

### Gaps identified

| Period | Days | Status | Cause |
|--------|------|--------|-------|
| 2006-04-01 → 2006-04-30 | 30 | Empty | THREDDS server failure during backfill — Final data exists on NASA servers but was not ingested |
| 2007-12-22 → 2008-01-20 | 30 | Empty | Same — batch failure during backfill, not retried |
| 2025-07-20 → 2025-11-30 | 134 | Empty | Final data not yet released by NASA for this period; Late backfill not yet run |
| 2025-12-02 → 2026-03-05 | 95 | Late | Final not yet released (~3.5 month lag); filled with Late product |
| 2026-03-06 → 2026-03-10 | 3–5 | Empty | Template extends beyond last ingestion |

## Gap-fill executed (2026-03-19)

All three gaps were filled by daisy-chaining `fill` commands:

```bash
# Gap 1: April 2006 — Final (30 days)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2006-04-01 --end-date 2006-04-30 \
    --product final --no-cluster --commit-batch 30 \
&& \
# Gap 2: Dec 2007 – Jan 2008 — Final (30 days)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2007-12-22 --end-date 2008-01-20 \
    --product final --no-cluster --commit-batch 30 \
&& \
# Gap 3: Jul–Nov 2025 — Late (135 days, Final not yet available)
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2025-07-20 --end-date 2025-12-01 \
    --product late --no-cluster --commit-batch 30
```

### Results

| Gap | Days | Product | Time | Result |
|-----|------|---------|------|--------|
| 2006-04 | 30/30 | Final | 10 min | 0 failed |
| 2007-12-22 → 2008-01-20 | 30/30 | Final | 6 min | 0 failed |
| 2025-07-20 → 2025-12-01 | 135/135 | Late | 48 min | 0 failed |
| **Total** | **195/195** | | **64 min** | **0 failed** |

## Coverage after gap-fill (2026-03-19)

| Product | Days | % |
|---------|------|---|
| Final | 9,150 | 97.2% |
| Late | 230 | 2.4% |
| Early | 0 | 0.0% |
| Empty | 3 | 0.0% |

### Product transitions (updated)

```
2000-06-01  →  Final     (backfill start, continuous)
2025-07-20  →  Late      (Final not yet released by NASA)
2026-03-07  →  Empty     (3 days, not yet ingested)
```

### Remaining work

**3 empty days** at the tail (2026-03-07 → 2026-03-10):

```bash
# Extend template and fill to present
uv run --python 3.12 imerg_hh_gcs_icechunk.py extend --end-date 2026-03-19
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2026-03-06 --end-date 2026-03-19 \
    --product late --no-cluster --commit-batch 7
```

**Replace Late with Final** (when NASA releases Final data, ~3.5 months
after observation). Re-run with `--product final` — the quality-aware
resume will overwrite Late data:

```bash
uv run --python 3.12 imerg_hh_gcs_icechunk.py fill \
    --start-date 2025-07-20 --end-date 2026-03-19 \
    --product final --no-cluster --commit-batch 30
```

## Upload to source.coop

The IMERG store is ~12 GB / 12,000+ objects. `gcs_to_source_coop_transfer.py`
handles this by downloading and uploading in ~2 GB batches, cleaning local
staging after each batch so only ~2 GB of disk is needed at any time.

### How batched transfer works

```
Batch 1: chunks_part1 (2 GB)          Batch 2: chunks_part2 (2 GB)
  GCS ──download──► local staging       GCS ──download──► local staging
  local ──upload──► source.coop S3      local ──upload──► source.coop S3
  local ──clean──► free disk            local ──clean──► free disk
  ...repeat for all 10 batches...
```

- Objects are grouped by top-level subdir (chunks, manifests, snapshots, etc.)
- Large groups (e.g. `chunks/` at ~12 GB) are split into ~2 GB sub-batches
- Small groups (manifests, refs, snapshots, transactions) fit in one batch each
- After each batch uploads, local files are deleted before the next download
- Peak local disk usage: **~2 GB** instead of 12 GB

### Credentials

source.coop uses S3-compatible access with **temporary credentials** (STS
session tokens). These typically expire after 1 hour. The script reads from
`.env` or environment:

```bash
# In .env or exported
export AWS_ACCESS_KEY_ID=ASIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
export AWS_DEFAULT_REGION=us-west-2
```

Or using `SOURCE_COOP_*` prefixed names (checked first):

```bash
export SOURCE_COOP_ACCESS_KEY_ID=...
export SOURCE_COOP_SECRET_ACCESS_KEY=...
export SOURCE_COOP_SESSION_TOKEN=...
```

**Credential lifetime**: at ~2 GB per batch and ~12 min upload time per batch,
the full 12 GB transfer takes ~60–70 min. Ensure credentials are valid for
at least 1.5 hours. If credentials expire mid-transfer, refresh `.env` and
re-run — the script skips already-uploaded files (size-match check).

### Commands

```bash
# Dry run — show batches
uv run gcs_to_source_coop_transfer.py --dry-run

# Full transfer (download → upload → clean per batch, ~2 GB disk)
uv run gcs_to_source_coop_transfer.py

# Keep local files after upload (needs 12 GB disk)
uv run gcs_to_source_coop_transfer.py --no-clean

# Upload existing staging dir only (skip GCS download)
uv run gcs_to_source_coop_transfer.py --skip-download

# Verify uploaded store from S3 (anonymous read)
uv run gcs_to_source_coop_transfer.py --verify
```
