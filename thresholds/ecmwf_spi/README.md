# ECMWF ERA5-Drought SPI — East Africa Pipeline

End-to-end workflow for downloading, storing, and analyzing the ECMWF ERA5-Drought
Standardized Precipitation Index (SPI) dataset over East Africa.

**Source**: [CDS derived-drought-historical-monthly](https://cds.climate.copernicus.eu/datasets/derived-drought-historical-monthly)

## Dataset Overview

| Property | Value |
|---|---|
| Variables | SPI1, SPI3, SPI6, SPI12, SPI24, SPI36, SPI48 |
| Spatial extent | Lat: -14.25° to 25.25°, Lon: 19.75° to 53.75° |
| Resolution | 0.25° × 0.25° (159 × 137 grid) |
| Temporal range | 1940-01 to 2025-11 (monthly) |
| Reference period | 1991–2020 |
| File count | 7,094 NetCDF files (7 SPI periods × ~1,000 months each) |
| Total download | ~1.3 GB compressed (7 zip files), ~1.3 GB uncompressed |
| Per-file size | ~25 KB each (1 timestep × 159 lat × 137 lon, float64) |

## Step 1: Download

The `download.txt` file contains 7 pre-generated URLs from the CDS API request.
Each zip file contains one SPI accumulation period's full time series.

```bash
wget -i download.txt -P . --content-disposition
```

## Step 2: Unzip

Extract all NetCDF files into a `data/` directory:

```bash
mkdir -p data/
for f in *.zip; do unzip -o "$f" -d data/; done
```

This produces 7,094 files with naming pattern:
```
SPI{N}_gamma_global_era5_moda_ref1991to2020_{YYYYMM}.area-subset.{bounds}.nc
```

Each file contains one monthly timestep at 0.25° resolution.

## Step 3: Icechunk Store

Ingest all 7 SPI periods into a single version-controlled Icechunk store,
then rechunk to pencil format (full time × 5 lat × 5 lon) for fast
per-pixel time-series access.

### Local Icechunk workflow

```bash
# Step 3a: Create empty template (writes metadata only, no data)
uv run ecmwf_spi_icechunk.py init --data-dir data/
#   → creates ecmwf_spi_ea_store/ (local Icechunk repo)

# Step 3b: Fill with data from 7,094 local NetCDF files
uv run ecmwf_spi_icechunk.py fill --data-dir data/
#   → reads each SPI period, writes in batches of 120 months (10 years)
#   → commits after each batch for resume safety

# Step 3c: Rechunk to pencil chunks for time-series access
uv run ecmwf_spi_icechunk.py rechunk \
    --source-path ecmwf_spi_ea_store \
    --target-path ecmwf_spi_ea_pencil
#   → pencil chunks: (n_time, 5, 5) ≈ full time series per 5×5 lat/lon tile
#   → enables O(1) reads for any pixel's full history

# Step 3d: Verify
uv run ecmwf_spi_icechunk.py verify --store-path ecmwf_spi_ea_pencil
```

### GCS upload (gs://cpc_awc/ecmwf_spi/)

To store the Icechunk/Zarr data on GCS (same bucket as CMORPH data but
under the `ecmwf_spi/` prefix), use `--gcs-bucket` and `--gcs-prefix`:

```bash
# Init + fill directly to GCS
uv run ecmwf_spi_icechunk.py init --data-dir data/ \
    --gcs-bucket cpc_awc --gcs-prefix ecmwf_spi_ea \
    --service-account coiled-data-e4drr_202505.json

uv run ecmwf_spi_icechunk.py fill --data-dir data/ \
    --gcs-bucket cpc_awc --gcs-prefix ecmwf_spi_ea \
    --service-account coiled-data-e4drr_202505.json

# Rechunk to GCS pencil store
uv run ecmwf_spi_icechunk.py rechunk \
    --source-path ecmwf_spi_ea_store \
    --target-path gs://cpc_awc/ecmwf_spi_ea_pencil \
    --service-account coiled-data-e4drr_202505.json
```

## Step 4: Return Period Analysis

Compute drought return period thresholds for every pixel using normal
distribution fitting:

```bash
# Compute thresholds (reads directly from local NetCDF files)
uv run ecmwf_spi_return_periods.py compute \
    --data-dir data/ \
    --output ecmwf_spi_return_periods.nc

# Verify output
uv run ecmwf_spi_return_periods.py verify \
    --input ecmwf_spi_return_periods.nc

# Plot distribution for a specific location
uv run ecmwf_spi_return_periods.py plot \
    --input ecmwf_spi_return_periods.nc \
    --data-dir data/ \
    --spi-period SPI3 \
    --lat 9.0 --lon 40.0 \
    --highlight-date 2015-09-01

# Generate threshold maps for all SPI periods
uv run ecmwf_spi_return_periods.py plot-map \
    --input ecmwf_spi_return_periods.nc
```

## Methodology: SPI Return Periods

### Why Normal Distribution (not Gumbel)?

SPI is fundamentally different from raw precipitation:

- **Precipitation** is non-negative, right-skewed → use Gumbel (extreme value)
  for annual maxima return periods
- **SPI** is already standardized to approximate N(0,1) by construction
  (gamma-fitted then probability-integral-transformed) → the normal
  distribution is the natural choice

### Three Threshold Types

The analysis produces three types of drought return period thresholds:

1. **Standard Normal** — Theoretical thresholds from N(0,1):
   - Direct application of `ppf(1/T)` where T is the return period
   - Same for all pixels (by SPI definition)
   - Serves as a reference baseline

2. **Fitted Normal** — Per-pixel N(μ, σ) fit:
   - Accounts for residual non-normality in the gamma-fitted SPI
   - `threshold = μ + σ × ppf(1/T)`
   - More accurate for pixels where SPI deviates from perfect normality

3. **Empirical Percentile** — Non-parametric:
   - `threshold = percentile(series, 100/T)`
   - No distributional assumption
   - More robust but noisier with limited data

### Return Period Categories

| Return Period | Label | Standard N(0,1) Threshold | Probability |
|---|---|---|---|
| 3-year | Mild | -0.431 | 33.3% |
| 5-year | Moderate | -0.842 | 20.0% |
| 10-year | Severe | -1.282 | 10.0% |
| 20-year | Extreme | -1.645 | 5.0% |
| 50-year | Exceptional | -2.054 | 2.0% |

### Interpretation

A 10-year return period drought threshold of SPI = -1.28 means:
- In any given month, there is a 10% chance of SPI falling below -1.28
- On average, this severity occurs once every 10 years
- The event shown in the screenshot (2015-09, SPI3 = -3.065 at 9°N, 40°E)
  exceeded the 50-year threshold, indicating an exceptional drought

## Output File Structure

The return period output is stored both locally and on GCS:

- **Local**: `ecmwf_spi_return_periods.nc` (7.6 MB)
- **GCS**: `gs://cpc_awc/ecmwf_spi/ecmwf_spi_return_periods.nc`

`ecmwf_spi_return_periods.nc` contains:

| Variable | Dimensions | Description |
|---|---|---|
| `fitted_threshold` | (spi_period, return_period, lat, lon) | Per-pixel fitted normal thresholds |
| `empirical_threshold` | (spi_period, return_period, lat, lon) | Per-pixel empirical percentile thresholds |
| `standard_threshold` | (return_period,) | Standard N(0,1) thresholds |
| `fit_mu` | (spi_period, lat, lon) | Fitted normal mean |
| `fit_sigma` | (spi_period, lat, lon) | Fitted normal std dev |
| `n_valid_months` | (spi_period, lat, lon) | Count of non-NaN months |

## GCS Storage Layout

All outputs are stored under `gs://cpc_awc/` alongside the existing CMORPH data:

```
gs://cpc_awc/
├── cmorph_ea_pencil/            # CMORPH pencil-chunked Zarr
├── cmorph_ea_return_periods/    # CMORPH return period output
├── ecmwf_spi/                   # ← ECMWF SPI outputs
│   └── ecmwf_spi_return_periods.nc   (7.6 MB, return period thresholds)
├── ecmwf_spi_ea/                # Icechunk store (after GCS init+fill)
└── ecmwf_spi_ea_pencil/         # Pencil-chunked Zarr (after GCS rechunk)
```

Service account: `coiled-data-e4drr_202505.json` (project: `e4drr-crafd`)

## Dependencies

All scripts use PEP 723 inline metadata — run with `uv run` and dependencies
are resolved automatically. No manual `pip install` needed.

Key packages: `xarray`, `netcdf4`, `numpy`, `scipy`, `matplotlib`, `icechunk`,
`zarr`, `dask`, `coiled`.

## References

- Keune, J., Di Giuseppe, F., Barnard, C., et al. (2025): ERA5–Drought: Global
  drought indices based on ECMWF reanalysis, *Scientific Data* 12, 616.
  https://doi.org/10.1038/s41597-025-04896-y
- WMO (2012): Standardized Precipitation Index User Guide. WMO-No. 1090.
  https://library.wmo.int/doc_num.php?explnum_id=7768
