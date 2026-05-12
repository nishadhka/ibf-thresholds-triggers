# GDO / RFE2 / CHIRPS SPI — East Africa Icechunk Stores

Pipelines for downloading drought and precipitation datasets from
Copernicus GDO and NOAA CPC, subsetting to East Africa, and writing
to GCS-backed Icechunk stores.

All four datasets share:
- East Africa bounding box: lat [-14.5, 25.5], lon [19.5, 54.0]
- GCS bucket: `cpc_awc`
- Resume-aware ingestion (re-run to pick up failed files)
- PEP 723 inline metadata — `uv run --python 3.12` handles dependencies

## Datasets

| Store | Variable | Source | Resolution | Period | GCS Path |
|-------|----------|--------|------------|--------|----------|
| CHIRPS SPI | spi1, spi3, spi6, spi9, spi12, spi24, spi48 | Copernicus GDO | 0.05 deg, monthly | 1991-2026 | `gs://cpc_awc/chirps_spi_ic_store` |
| fAPAR Anomaly | fpanv | Copernicus GDO (VIIRS) | 0.083 deg, dekadal | 2012-2026 | `gs://cpc_awc/gdo_fpar_ic_store` |
| fAPAR Anomaly | fapan | Copernicus GDO (MODIS) | 0.083 deg, dekadal | 2001-2022 | `gs://cpc_awc/gdo_fpar_modis_ic_store` |
| Soil Moisture Anomaly | smang | Copernicus GDO | 0.05 deg, dekadal | 1995-2026 | `gs://cpc_awc/gdo_sma_ic_store` |
| RFE2 Rainfall | rfe2 | NOAA CPC/FEWS | 0.1 deg, daily | 2001-2026 | `gs://cpc_awc/rfe2_ic_store` |

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- GCS service account file (`coiled-data-e4drr_202505.json`)

---

## `chirps_spi_icechunk.py` — CHIRPS SPI Multi-Index

Downloads all 7 SPI indices (1, 3, 6, 9, 12, 24, 48 months) from
Copernicus GDO, subsets to EA, and writes as 7 variables in a single
Icechunk store. SPI1/SPI3 are dekadal but resampled to monthly
(day==1 timesteps) so all 7 variables share one time axis.

**Source**: `https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/GDO_CHIRPS_Standardized_Precipitation_Index_{SPI_NAME}/ver3-0-0/`

**Variables**: spi1, spi3, spi6, spi9, spi12, spi24, spi48

**Store stats** (from logs):
- 36 years ingested (1991-2026), 422 monthly time steps
- Ingestion time: ~163 min

```bash
# Full pipeline: ingest all years + verify
uv run --python 3.12 chirps_spi_icechunk.py run \
    --service-account coiled-data-e4drr_202505.json

# Ingest only (skip verify)
uv run --python 3.12 chirps_spi_icechunk.py ingest \
    --service-account coiled-data-e4drr_202505.json

# Verify existing store
uv run --python 3.12 chirps_spi_icechunk.py verify \
    --service-account coiled-data-e4drr_202505.json

# Use local storage instead of GCS
uv run --python 3.12 chirps_spi_icechunk.py run --local ./chirps_spi_local
```

**Known issues**:
- SPI1/SPI3 are dekadal (36 steps/year), SPI6-48 are monthly (12 steps/year).
  To share a single time axis, SPI1/SPI3 are filtered to monthly (day==1) only.
- Near-duplicate lat values in some files — consecutive pairs differ by ~1e-14.
  Fixed by detecting via `np.unique(np.round(lat, 4))` and taking every 2nd point.

---

## `gdo_fpar_icechunk.py` — fAPAR Anomalies (VIIRS)

Downloads GDO fAPAR anomaly NC files (one per year), subsets to EA,
writes to GCS Icechunk store. Processes one file at a time: download,
subset, write, cleanup.

**Source**: `https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/GDO_Fraction_of_Absorbed_Photosynthetically_Active_Radiation_Anomalies_fAPAR_VIIRS/ver3-0-0/`

**Variable**: fpanv (fAPAR anomaly from VIIRS)

**Store stats**:
- 15 files (2012-2026), 507 dekadal time steps
- Grid: 480 lat x 414 lon (0.083 deg)

```bash
# Full pipeline
uv run --python 3.12 gdo_fpar_icechunk.py run \
    --service-account coiled-data-e4drr_202505.json

# Ingest only
uv run --python 3.12 gdo_fpar_icechunk.py ingest \
    --service-account coiled-data-e4drr_202505.json

# Verify
uv run --python 3.12 gdo_fpar_icechunk.py verify \
    --service-account coiled-data-e4drr_202505.json
```

**Known issues**:
- **Double lat**: Files for years 2020, 2022, 2024, 2025 have near-duplicate
  lat values (consecutive pairs differ by ~1e-14, doubling the lat dimension
  from ~480 to ~960). Detected by comparing `len(lat)` vs
  `len(np.unique(np.round(lat, 4)))` and fixed by taking every 2nd lat point
  (`ds.isel(lat=slice(None, None, 2))`).
- **Inconsistent lat ordering**: Some files have ascending lat (-56 to 90),
  others descending (90 to -56). The pipeline detects direction and ensures
  consistent descending order across all files.
- Some files have an extra `band` dimension — squeezed out with
  `squeeze("band", drop=True)`.

---

## `gdo_fpar_modis_icechunk.py` — fAPAR Anomalies (MODIS)

Same pipeline shape as the VIIRS script, against the older MODIS-based
GDO fAPAR product (`ver1-3-1`, 2001–2022, var `fapan`). The MODIS series
ends 2022-11-11; for 2023+ use the VIIRS store.

**Source**: `https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/GDO_Fraction_of_Absorbed_Photosynthetically_Active_Radiation_Anomalies_fAPAR_MODIS/ver1-3-1/`

**Variable**: fapan (fAPAR anomaly from MODIS)

**Store stats**:
- 22 files (2001-2022), last file truncated at 20221111
- Grid: 480 lat x 414 lon (0.083 deg)
- GCS prefix: `gdo_fpar_modis_ic_store`
- source.coop prefix: `e4drr-project/observations/gdo_fpar_modis_icechunk`

```bash
# Full pipeline (GCS)
uv run --python 3.12 gdo_fpar_modis_icechunk.py run \
    --service-account coiled-data-e4drr_202505.json

# Direct ingest to source.coop
uv run --python 3.12 gdo_fpar_modis_icechunk.py run --source-coop
```

Inherits the same `--source-coop`, `--local`, and resume-detection
behaviour as the VIIRS script. The double-lat / inconsistent-lat-ordering
fixes apply identically.

---

## `gdo_sma_icechunk.py` — Soil Moisture Index Anomaly

Downloads GDO SMA ZIP files (one per year, containing dekadal GeoTIFFs),
extracts, subsets to EA, writes to GCS Icechunk store.

**Source**: `https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/GDO_Soil_Moisture_Index_Anomaly/ver3-0-1/`

**Variable**: smang (Soil Moisture Index Anomaly)

**Store stats**:
- 32 files (1995-2026), 1,122 dekadal time steps
- Grid: 800 lat x 690 lon (0.05 deg)
- Ingestion time: ~69 min (first run), split across 2 runs due to SIGTERM

```bash
# Full pipeline
uv run --python 3.12 gdo_sma_icechunk.py run \
    --service-account coiled-data-e4drr_202505.json

# Ingest only
uv run --python 3.12 gdo_sma_icechunk.py ingest \
    --service-account coiled-data-e4drr_202505.json

# Verify
uv run --python 3.12 gdo_sma_icechunk.py verify \
    --service-account coiled-data-e4drr_202505.json
```

**Known issues**:
- **Double lat** (same as fAPAR): Some GeoTIFFs have near-duplicate y
  coordinates. Fixed with the same deduplication approach.
- **GeoTIFF uses (y, x) instead of (lat, lon)**: The pipeline renames
  `y -> lat`, `x -> lon` after subsetting for consistency with other stores.
- **SIGTERM during long runs**: The first SMA run was killed after ingesting
  14/32 files. The pipeline's resume detection correctly skipped the 14
  already-committed files on the second run.
- **GDAL/PROJ warnings**: `proj.db` version mismatch warnings appear in logs
  but are harmless and don't affect data.

---

## `rfe2_icechunk.py` — CPC RFE2 Daily Rainfall

Downloads daily RFE2 rainfall estimate ZIPs from CPC FTP (each containing
one GeoTIFF), subsets to EA, and writes to GCS Icechunk in monthly batches.

**Source**: `https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/geotiff/`

**Variable**: rfe2 (Rainfall Estimate, mm/day)

**Store stats**:
- 9,193 daily files (2001-01-01 to 2026-03-07), 303 monthly commits
- Grid: 400 lat x 345 lon (0.1 deg)
- Ingestion time: ~46 min

```bash
# Full pipeline
uv run --python 3.12 rfe2_icechunk.py run \
    --service-account coiled-data-e4drr_202505.json

# Ingest only
uv run --python 3.12 rfe2_icechunk.py ingest \
    --service-account coiled-data-e4drr_202505.json

# Verify
uv run --python 3.12 rfe2_icechunk.py verify \
    --service-account coiled-data-e4drr_202505.json
```

**Known issues**:
- **Inconsistent lon grid in file 20240624**: This file includes lon=54.0
  (the EA boundary) while all other files stop at 53.9, producing 346 lon
  instead of 345. Fixed by using a slightly exclusive upper bound:
  `x=slice(EA_LON_MIN, EA_LON_MAX - 0.01)`.
- **Floating-point coordinate differences between files**: `xr.concat()` with
  the default `join='outer'` doubles dimensions (400 lat becomes 800) because
  y/x coordinates differ by ~1e-14 between files. Fixed with
  `xr.concat(arrays, dim="time", join="override")`.
- **Monthly batching**: ~9,200 daily files would create too many Icechunk
  commits if done individually. Batching by month keeps commits to ~303 and
  makes resume detection practical.
- **Double lat** (same as fAPAR/SMA): Some GeoTIFFs have near-duplicate y
  values. Detected and deduplicated the same way.

---

## Common patterns across all pipelines

### Resume detection

All pipelines track committed files/years/months via Icechunk commit messages
(`"ingest {identifier}"`). Re-running a pipeline safely skips already-committed
data and only processes remaining files.

### Double lat issue

Multiple Copernicus GDO datasets have a recurring issue where certain years'
files contain near-duplicate latitude values — pairs of lat points that differ
by ~1e-14 (e.g., 10.0000000000000 and 10.0000000000001). This doubles the
lat dimension. All pipelines detect this by comparing `len(lat)` vs
`len(np.unique(np.round(lat, 4)))` and fix it by taking every 2nd point.

Affected: fAPAR (years 2020, 2022, 2024, 2025), SMA (various), RFE2 (rare).

### File processing pattern

All pipelines follow the same sequential pattern to minimize disk usage:
1. Download file to temp location
2. Open, subset to EA bounding box
3. Append to Icechunk store and commit
4. Delete temp file

This means only one file is on disk at a time, avoiding the need for
large staging directories.

### GCS storage

All stores use the same GCS bucket (`cpc_awc`) with different prefixes.
Authentication is via a service account JSON file passed with
`--service-account`. All pipelines also support `--local` for local
filesystem storage during testing.

---

## Store access (reading)

```python
import icechunk
import xarray as xr

# Example: read CHIRPS SPI store
storage = icechunk.gcs_storage(
    bucket="cpc_awc",
    prefix="chirps_spi_ic_store",
    service_account_file="coiled-data-e4drr_202505.json",
)
repo = icechunk.Repository.open(
    storage, config=icechunk.RepositoryConfig.default()
)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
# Variables: spi1, spi3, spi6, spi9, spi12, spi24, spi48
# Time: 1991-01-01 to 2026-02-01 (422 monthly steps)
```

Replace prefix with `gdo_fpar_ic_store`, `gdo_sma_ic_store`, or
`rfe2_ic_store` for the other datasets.

---

## source.coop (public S3 access)

All four stores are published to [source.coop](https://source.coop) under
`s3://us-west-2.opendata.source.coop/e4drr-project/observations/`.

| Store | S3 Prefix |
|-------|-----------|
| CHIRPS SPI | `e4drr-project/observations/chirps_spi_icechunk` |
| fAPAR Anomaly (VIIRS) | `e4drr-project/observations/gdo_fpar_icechunk` |
| fAPAR Anomaly (MODIS) | `e4drr-project/observations/gdo_fpar_modis_icechunk` |
| Soil Moisture Anomaly | `e4drr-project/observations/gdo_sma_icechunk` |
| RFE2 Rainfall | `e4drr-project/observations/rfe2_icechunk` |
| IMERG HH Precip | `e4drr-project/observations/imerg_hh_icechunk` |

### Pencil-chunked stores (time-series optimized)

Pencil-chunked versions of the pancake stores, optimized for fast
time-series queries at a single location:

| Store | S3 Prefix | Chunks |
|-------|-----------|--------|
| CHIRPS SPI (pencil) | `e4drr-project/observations/chirps_spi_pencil_zarr` | (all, 5, 5) |
| fAPAR Anomaly (pencil) | `e4drr-project/observations/gdo_fpar_pencil_zarr` | (all, 5, 5) |
| Soil Moisture Anomaly (pencil) | `e4drr-project/observations/gdo_sma_pencil_zarr` | (all, 5, 5) |
| RFE2 Rainfall (pencil) | `e4drr-project/observations/rfe2_pencil_zarr` | (all, 5, 5) |
| IMERG HH Precip (pencil) | `e4drr-project/observations/imerg_hh_pencil_zarr` | (all, 5, 5) |

Create pencil stores on source.coop using `rechunk_to_pencil.py --coiled`:

```bash
# Dry run — show source/target shapes
uv run rechunk_to_pencil.py all --coiled --dry-run

# Rechunk a single dataset
uv run rechunk_to_pencil.py gdo_fpar --coiled

# Rechunk all datasets
uv run rechunk_to_pencil.py all --coiled

# Verify pencil store (anonymous read)
uv run rechunk_to_pencil.py gdo_fpar --coiled --verify
```

### Read access (no credentials needed)

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/observations/chirps_spi_icechunk",
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

Use `gcs_to_source_coop_transfer.py` to sync stores from GCS to S3:

```bash
# Set credentials in .env or environment
export SOURCE_COOP_ACCESS_KEY_ID=...
export SOURCE_COOP_SECRET_ACCESS_KEY=...
export SOURCE_COOP_SESSION_TOKEN=...   # optional, for temporary credentials

# Dry run — list files and target paths
uv run gcs_to_source_coop_transfer.py rfe2 --dry-run

# Transfer a single dataset
uv run gcs_to_source_coop_transfer.py rfe2

# Transfer all 4 datasets
uv run gcs_to_source_coop_transfer.py all

# Resume upload only (skip GCS download)
uv run gcs_to_source_coop_transfer.py rfe2 --skip-download

# Verify uploaded store
uv run gcs_to_source_coop_transfer.py rfe2 --verify
```

### Incremental ingestion to source.coop

All four pipeline scripts support `--source-coop` to write directly to
source.coop instead of GCS:

```bash
uv run --python 3.12 rfe2_icechunk.py ingest --source-coop
```

Requires `SOURCE_COOP_ACCESS_KEY_ID`, `SOURCE_COOP_SECRET_ACCESS_KEY`,
and optionally `SOURCE_COOP_SESSION_TOKEN` in `.env` or environment.

---

## METAR observations — East Africa

Station-level surface observations to complement the gridded drought stores.
The same EA bbox (`lat [-14.5, 25.5], lon [19.5, 54.0]`) is used across all
METAR scripts so station data co-locates cleanly with the gridded products.

| Script | Purpose | Source |
|--------|---------|--------|
| `metar_africa_coverage.py` | Station coverage map + histogram for one WB2 partition (defaults to 2020-01) | `gs://weatherbench2/datasets/metar/metar-timeNominal-by-month/` |
| `metar_ea_backfill_iowa.py` | Per-station historical backfill (2024+) | Iowa Mesonet ASOS (`mesonet.agron.iastate.edu`) |
| `metar_ea_live_24h.py` | Last-24h (max 72h) live feed | NOAA AWC (`aviationweather.gov/api/data/metar`) |
| `metar_ea_mam2026_coverage.py` | One-off MAM 2026 coverage analysis from local parquets | Local `data/metar_ea_iowa/` |

The WeatherBench2 mirror covers 2001-07 → 2023-12 (23 years, 270 monthly
partitions). For 2024-onwards the Iowa Mesonet ASOS API supplies the same
ICAO stations parsed into a WB2-compatible schema, so
`pd.concat([wb2_2001_2023, iowa_2024_now])` works directly.

### Backfill schema (`metar_ea_backfill_iowa.py`)

Output partitioned as `data/metar_ea_iowa/year=YYYY/month=MM/YYYY-MM.parquet`,
matching the WB2 layout. 20 columns, all units normalised to °C / m·s⁻¹ /
hPa / m:

`stationName, locationName, latitude, longitude, timeObs, timeNominal,
reportType, temperature, dewpoint, relativeHumidity, windDir, windSpeed,
windGust, altimeter, seaLevelPressure, visibility, precip1Hour, skyCover1,
weatherCodes, rawMetar`

The station list is seeded from the WB2 `year=2023/month=12` partition
(filtered to the EA bbox — 82 stations) so the active set reflects the
latest available roster, not the 2020 sample.

### MAM 2026 ingest

```bash
# 82 EA stations × 71 days (2026-03-01 → 2026-05-11) via Iowa Mesonet
uv run --python 3.12 metar_ea_backfill_iowa.py \
    --start 2026-03-01 --end 2026-05-12

# Coverage map + histogram for the MAM 2026 window
uv run --python 3.12 metar_ea_mam2026_coverage.py
```

**Run stats** (run on 2026-05-12):
- 82 stations seeded → 72 returned data (10 inactive in this window)
- 102,808 observations · 1.4 MB Mar + 1.3 MB Apr + 504 KB May parquet
- ~17 min wall time (paced at 12s between station fetches to stay polite
  with Iowa Mesonet; occasional 429s retried automatically)
- Coverage: median 83% of the 1,728 expected hourly slots, ~70 distinct
  stations reporting per day

Top reporters (≥98% coverage): FMCZ (Mayotte), OTBD (Doha), HKMO (Mombasa),
HRYR (Kigali), HKEL (Eldoret), and the Saudi cluster (OERK Riyadh, OEJN
Jeddah, OEAH Al Ahsa, all at 100%). Notably sparse / offline in MAM 2026:
HDAM Djibouti (7 obs), FZRF Kalemie (35), HUKS Kasese (326).

The `metar_ea_mam2026_coverage.py` script writes
`metar_ea_mam2026_stations.png` (cartopy station map coloured by
coverage %) and `metar_ea_mam2026_coverage.png` (histogram + daily
volume). Both PNGs and the CSV exports of the parquets are intentionally
kept out of git — regenerate by re-running the script or the small
parquet-to-csv loop:

```bash
uv run --python 3.12 --with pandas --with pyarrow - <<'PY'
import pandas as pd
from pathlib import Path
for p in sorted(Path("data/metar_ea_iowa").glob("year=*/month=*/*.parquet")):
    pd.read_parquet(p).to_csv(p.with_suffix(".csv"), index=False)
PY
```

### Live feed

```bash
# Last 24h, write parquet to data/metar_ea_live/
uv run --python 3.12 metar_ea_live_24h.py

# Smoke test (6h, print to stdout)
uv run --python 3.12 metar_ea_live_24h.py --hours 6 --print
```

Same units/schema as the backfill (with the addition of `elevation` and
`timeReport` columns from AWC), so live + backfill + WB2 concatenate
cleanly for any analysis crossing the 2024 boundary.
