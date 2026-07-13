# CMORPH → Icechunk: upload record (`cmorph-s3-nc-v2` — COMPLETE)

**Store:** `gs://cpc_awc/icechunk/cmorph-s3-nc-v2` ← **use this one**
**Superseded:** `gs://cpc_awc/icechunk/cmorph-s3-nc` (v1 — has a 2004-2005 gap, no
post-2020; safe to retire).
**Source (par):** `gs://cpc_awc/cmorph_catalog/catalog.parquet` (236,688 rows, one
per 30-min NetCDF, `kerchunk_refs` pre-computed) → virtual refs into
`s3://noaa-cdr-precip-cmorph-pds/` (anonymous). Public mirror, byte-identical
(md5 `2eb444f793a157817278d4d25e4195ae`):
`huggingface.co/datasets/E4DRR/virtualizarr-stores/…/cmorph-aws-s3-1998-2024.parquet`.
**Builder / runner:** `build_cmorph_icechunk.py` / `backfill_cmorph_icechunk.py`
(GEFS-style, no Coiled; per-60-day-batch commits; single sequential writer;
manifest split along `time`).
**SA:** `coiled-data-e4drr@e4drr-crafd.iam.gserviceaccount.com`
(`coiled-data-e4drr_202505.json`, gitignored).

---

## Status: 236,688 / 236,688 par files converted (100%)

CMORPH kept the same 1649×4948 grid throughout but changed the **NetCDF chunk
shape** twice. A Zarr array has **one** chunking and a *virtual* ref must land on
the array's chunk grid, so each chunk-regime gets its **own group** in the one
store. Consumers open the group covering their period.

| group | window(s) | chunk shape | par files | days | steps |
|---|---|---|---|---|---|
| `/` (root) | 1998-01-01 → 2020-06-30 | `(1, 501, 1506)` | 197,208 | 8,217 | 394,416 |
| `cmorph_832` | 2020-07-01 → 2022-08-31, 2023-06-01 → 2023-07-31 | `(1, 832, 2497)` | 20,472 | 853 | 40,944 |
| `cmorph_825` | 2022-09-01 → 2023-05-31, 2023-08-01 → 2024-12-31 | `(1, 825, 2474)` | 19,008 | 792 | 38,016 |
| **total** | **1998-01-01 → 2024-12-31** | | **236,688** | **9,862** | **473,376** |

Note the two post-2020 groups' time ranges **overlap in wall-clock** (the regimes
interleave), but their day sets are **disjoint** — every day appears in exactly
one group.

**Verified on all three groups:** monotonic, zero duplicates, native chunking
preserved, virtual read from S3 materializes 99.9–100% finite with physical
precip values (max 50–77 mm). Root is continuous with **zero missing 30-min
slots**.

## Reading it

```python
import icechunk, xarray as xr, json
storage = icechunk.gcs_storage(bucket="cpc_awc", prefix="icechunk/cmorph-s3-nc-v2",
    config={"service_account_key": json.dumps(json.load(open("coiled-data-e4drr_202505.json")))})
auth = icechunk.containers_credentials(
    {"s3://noaa-cdr-precip-cmorph-pds/": icechunk.s3_anonymous_credentials()})
repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
store = repo.readonly_session("main").store

ds_pre  = xr.open_zarr(store, consolidated=False)                      # 1998 → 2020-06
ds_832  = xr.open_zarr(store, group="cmorph_832", consolidated=False)  # see table
ds_825  = xr.open_zarr(store, group="cmorph_825", consolidated=False)
```
A single continuous series across regimes needs a **materialize-rechunk** (copies
data, no longer virtual) — not done here.

## Build history / the three bugs that had to be fixed

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | per-day commits slowed 9 s → 20 s/day | each commit rewrote the growing 60-day manifest shard up to 60× | commit **per 60-day batch** = one shard written once (`--batch-days 60`); flat ~120–150 s/batch ≈ 2.5 s/day |
| 2 | 33/154 batches failed; store ended with a **2004-2005 mid-series gap** | (a) post-2020 chunk-regime files hard-failed the batch; (b) a transient S3 error during `reconstruct` killed all 60 days. Icechunk appends only at the **end**, so a mid-series gap can't be back-filled | robust builder: **chunk-shape filter** + **per-file try/except** (offenders → `cmorph_skipped_<store>.json`); then a **clean rebuild** to `-v2` in chronological order, which closed the gap |
| 3 | every append into a **named group** died: `KeyError: '//cmorph_832/lon'` | virtualizarr 2.7.0: on append, coords lacking the append dim (lat/lon) still take the write path and their chunks are inlined; `key_prefix = f"{group.name}/{arr_name}"` and a named group's `.name` carries a leading `/` → icechunk's `store.set` resolves `//group/lon`. Root group immune (`.name == "/"`), so the 501,1506 era never hit it | `_patch_virtualizarr_group_append()` in the builder strips the leading slash on the inlined-chunk write (the virtual-ref path already tolerates it) |

**Not an OOM.** The manifest split (`MANIFEST_SPLIT_TIME = 2880`, set on
`RepositoryConfig` at `Repository.create` and persisted) means an append touches
only the **last shard** — this is what fixed the original #884 manifest OOM.
Appends are bounded and cheap; the 60-day batching keeps only 1,440 files in
memory at a time.

## Commands (reproduce / extend)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=coiled-data-e4drr_202505.json
STORE=gs://cpc_awc/icechunk/cmorph-s3-nc-v2

# root (501,1506) era -- resumable & idempotent (skips days already in the group)
uv run backfill_cmorph_icechunk.py --store $STORE --end 20200630 --batch-days 60

# post-2020 regime groups (all four windows, sequential single writer)
bash run_post2020_groups.sh
```

Both are **resumable**: `dates_done()` reads the group's existing `time` axis and
skips it, so a re-run only builds what's missing. To extend past 2024-12-31,
regenerate the par catalog for the new files, then re-run the matching group
window (check the new files' chunk shape first — a third regime would need a new
group).
