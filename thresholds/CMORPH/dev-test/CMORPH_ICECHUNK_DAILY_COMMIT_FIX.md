# CMORPH → Icechunk: the daily-commit fix for the #884 OOM

This corrects the three failed direct-to-Icechunk strategies in
`cmorph_s3_to_gcs_icechunk_parallel.py`, documented in
[`ICECHUNK_PARALLEL_ERRORS_ANALYSIS.md`](ICECHUNK_PARALLEL_ERRORS_ANALYSIS.md)
and discussed upstream in
[VirtualiZarr #884](https://github.com/zarr-developers/VirtualiZarr/discussions/884).
The corrected implementation is
[`cmorph_daily_commit_icechunk.py`](cmorph_daily_commit_icechunk.py).
The pattern was first proven on the ECMWF GIK pipeline in
`grib-index-kerchunk/ecmwf/icechunk-par/` (468 MB flat coordinator RSS for
360K virtual refs/commit; dask read + materialization verified bit-exact).

## Why each old mode failed, and what replaces it

| old mode | failure | corrected by |
|---|---|---|
| **collect** — workers return kerchunk dicts, coordinator holds all, one big commit | ~82 KB × 236,688 files ≈ 20 GB → OOM-killed (exit 137) at ~28K files | coordinator only ever holds **one day** (24 dicts ≈ 2 MB); write, commit, free, next day |
| **direct** — every worker commits per-file to `main` | 70–80% `ConflictError` (optimistic concurrency), GCS 429s on the branch ref | **single writer**: workers never touch the repo; one sequential commit per day → zero conflicts, zero ref-mutation rate-limit pressure |
| **batch** — per-worker branches to merge later | merge complexity, 429s, stale-state on cluster restarts | no branches at all; resume comes from the store itself (last committed timestep) |
| append growth (`append_dim` re-reads O(n) metadata; single manifest rewritten per commit) | memory/IO blow-up by ~87K timesteps | **manifest splitting along time** (2,880 timesteps = 60 days per shard): each commit rewrites only the last shard — O(1) appends |
| STS 1-hour token expiry mid-run | partial state, restarts | the build needs **no cloud credentials at all**: S3 reads are anonymous, the store is **temp local**; credentials are only needed for the final publish sync |
| non-standard grids (825×2474, 832×2497) | "cannot concatenate arrays with inconsistent chunk shapes" mid-commit | per-file shape validation before concat; offenders logged to `cmorph_skipped_files.json`, day still commits |

## The corrected flow

```
for each day (chronological):
    workers (Coiled or local threads): virtualize the day's 24 files
        → 24 kerchunk dicts (~2 MB total)     # workers never open the repo
    coordinator: reconstruct VDSs → xr.concat (48 half-hour steps)
        → to_icechunk(local_store, append_dim="time") → commit → free
publish once at the end:
    gsutil -m rsync -r <local-store> gs://<bucket>/<prefix>
    # Icechunk's local layout is byte-identical to its object-store layout;
    # readers only need GET/LIST (verified pattern on source.coop)
```

## Measured (2020-01-01…03, local backend, 2-core VM)

- ~7 s per day-commit (24 files virtualized + appended + committed)
- coordinator peak RSS **249–281 MB, flat** across days (vs 20 GB projected
  for collect mode) — per-day memory is independent of archive length
- resume verified: rerun skips completed days, appends only new ones
- stored `cmorph` int16 **bit-exact** vs direct S3 netCDF reads (decoded
  values differ only by float32-vs-float64 `scale_factor` arithmetic, ~1e-6)

Full-archive projection: 236,688 files = 9,862 daily commits. At 5–7 s/day
sequential that is ~14–19 h on one small VM with zero cluster cost for the
commit side; Coiled workers (`--backend coiled`) parallelize the
virtualization within each day and cut the per-day wall time to the commit
itself. 7.6 M chunk refs land in ~165 manifest shards of ~46K refs each.

## Two implementation gotchas (carried in the script)

1. Kerchunk serializes `_FillValue` for the float64 `lat`/`lon` coords as a
   float, which `xarray`'s zarr backend refuses on reconstruction — the
   script strips those attrs (`sanitize_refs`).
2. `time` must be in `loadable_variables` when reconstructing, so `xr.concat`
   and `append_dim="time"` order on real timestamps.
