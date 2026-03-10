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
