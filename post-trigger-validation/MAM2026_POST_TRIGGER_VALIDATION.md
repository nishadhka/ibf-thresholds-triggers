# MAM 2026 post-trigger validation (Karamoja)

Compares the SEAS51 SPI-3 trigger decisions for the January (readiness) and February (activation) 2026
initializations with observed MAM 2026 SPI-3 from two observation stores. SPI-3 is labelled by the last month of
its window, so MAM 2026 is the value at `2026-05-01`.

## Files

| File | What it does |
|---|---|
| `10-era5-spi-store-update.py` | Brings the ERA5-Drought SPI store on source.coop up to date: `download` fetches months from the CDS, `append` adds them to the store. Only needed when the store is behind the target month. |
| `11-spi3-mam2026-validation.py` | Reads CHIRPS v3 daily rainfall (dynamical.org) and ERA5-Drought SPI-3 (source.coop), averages SPI-3 over the nine districts, and scores the Jan and Feb decisions for each source. |
| `trigger_evaluation.csv` | Earlier CHIRPS-based evaluation (seven districts, February decision only). Superseded by `out/chirps_trigger_evaluation_mam2026.csv`. |

The routine is self-contained in this folder:

```bash
uv run 10-era5-spi-store-update.py download --start 2026-10 --end 2026-10   # only if ERA5 is behind
uv run 10-era5-spi-store-update.py append
uv run 11-spi3-mam2026-validation.py
```

`10-...` needs a CDS key (`.cdsapirc` in this folder, git-ignored, or `~/.cdsapirc`) and a `source-coop login`.
`11-...` needs nothing. Run with `uv run 11-spi3-mam2026-validation.py` (`--obs chirps`, `--obs era5`, or the default `both`). The stores
are public, so no credentials are needed. A full run takes about 30 s. The CHIRPS monthly totals for the
Karamoja box are cached in `data/`, and the results go to `out/`. Both folders are git-ignored.

Outputs: `out/{chirps,era5}_district_spi3_monthly.csv`, `out/{chirps,era5}_trigger_evaluation_mam2026.csv`,
`out/chirps_vs_era5_mam2026.csv`.

## Observation stores

| Source | Store | Content | Covers |
|---|---|---|---|
| CHIRPS | [dynamical.org `ucsb-chc-chirps-analysis-final`](https://dynamical.org/catalog/ucsb-chc-chirps-analysis-final/) | CHIRPS v3 final daily, 0.05 deg, global land; the Icechunk URL comes from its STAC collection | 1981-01-01 to 2026-09-30, updated monthly |
| ERA5 | source.coop `e4drr-project/observations/era5_ecmwf_icechunk` | ERA5-Drought SPI1 to SPI48, 0.25 deg, 1991-2020 reference | 1940-01 to 2026-09 |

CHIRPS v3 is the product `process_chirps_data()` calibrates the trigger on (`ea_chirps_v3_monthly_20250312.zarr`),
so it replaces the e4drr CHIRPS v2.0 store (`observations/chirps_daily_icechunk`) for this validation.

Updating the e4drr stores on source.coop (2026-10-29). The full builders, which create the stores from scratch,
are on the `xarray-method` branch; `10-era5-spi-store-update.py` copies only the ERA5 update path:

- Write access uses the `source-coop` CLI (`~/.local/bin/source-coop`, v0.3.0). Run `source-coop login --duration 12h`
  once; after that, `source-coop creds` refreshes the credentials. They work only through `https://data.source.coop`
  (bucket `e4drr-project`, path-style), not the raw `us-west-2.opendata.source.coop` bucket. Both builders take
  `--source-coop-cli` for this.
- CHIRPS: `chirps_daily_icechunk.py ingest --source-coop-cli --threads 16` resumes from the last `ingest YYYYMM`
  commit, reading the DE Africa daily GeoTIFFs (available to 2026-09). The proxy sometimes returns an empty error
  body. Commits now retry with a fresh session, and the run stops rather than skip a month, because a skipped
  month leaves a gap in the time axis. The first run on 2026-10-29 lost 2007-04 and 2013-10 that way, so `main`
  was reset to `ingest 200703` (snapshot `82ZMZAG6XD667M281FX0`; the gapped tip was `7E341XNW35H8WYBSZACG`)
  and re-ingested. The ingest was then stopped, because this validation now uses CHIRPS v3 from dynamical.org.
  The v2.0 store is gap-free from 1981-01-01 to 2015-01-31; re-running the command continues at 2015-02.
- ERA5: `download --start 2026-02 --end 2026-09` (CDS `derived-drought-historical-monthly`, intermediate
  dataset), then `append`, with `ecmwf_spi_icechunk.py --source-coop-cli` or `10-era5-spi-store-update.py`. The 2026-02 to 2026-09 files are ERA5T (preliminary).
  The pencil (`era5_ecmwf_pencil`) and return-period stores were not rebuilt and still end at 2026-01.

## Method

- CHIRPS v3: daily rate (kg m-2 s-1) × 86400 summed to monthly mm (months with a missing day are dropped),
  conservatively regridded to 0.25 deg on the grid `process_chirps_data()` builds (centres from
  `np.arange(min, max, 0.25)`), then SPI-3 with its xclim call: gamma, APP, `floc=0`, 3-month window, calibration
  1991-01-01 to 2018-01-01 (`--cal-end` changes it). The regrid uses area-overlap weights on the regular grids
  rather than xesmf, which needs ESMF from conda. `--res 0` skips it.
- ERA5: the `SPI3` variable as ECMWF publishes it (gamma, 1991-2020 reference), not recomputed.
- District value is the mean of the 0.25 deg cells whose centre falls inside the district; the cell nearest the
  centroid is used if none does.
- Observed drought: moderate if SPI-3 < -0.68, severe if SPI-3 < -0.84.
- Forecast trigger: district probability above the trigger (Jan 15.2% / 11.1%, Feb 16.2% / 15.2%, moderate / severe).
- Outcomes: Hit, Miss, False alarm, Correct no-action.

## Result: ERA5-Drought SPI-3

No district was in moderate or severe drought (SPI-3 from 0.25 in Napak to 3.70 in Moroto).

| District | SPI-3 MAM 2026 |
|---|---|
| Abim | 2.587 |
| Napak | 0.253 |
| Nabilatuk | 0.580 |
| Kotido | 0.273 |
| Moroto | 3.700 |
| Nakapiripirit | 0.899 |
| Kaabong | 2.207 |
| Karenga | 1.096 |
| Amudat | 0.729 |

| Initialization | Forecast | Outcome |
|---|---|---|
| January 2026 (readiness) | Triggered in all 9 districts, both scenarios | 9 false alarms per scenario |
| February 2026 (activation) | No trigger in any district | 9 correct no-action per scenario |

## Result: CHIRPS v3 SPI-3

No district was in moderate or severe drought (SPI-3 from -0.45 in Napak to 1.22 in Amudat).

| Initialization | Forecast | Outcome |
|---|---|---|
| January 2026 (readiness) | Triggered in all 9 districts, both scenarios | 9 false alarms per scenario |
| February 2026 (activation) | No trigger in any district | 9 correct no-action per scenario |

## CHIRPS v3 and ERA5 side by side

| District | CHIRPS v3 SPI-3 | ERA5-Drought SPI-3 |
|---|---|---|
| Abim | 0.184 | 2.587 |
| Napak | -0.451 | 0.253 |
| Nabilatuk | 0.396 | 0.580 |
| Kotido | 0.125 | 0.273 |
| Moroto | 0.633 | 3.700 |
| Nakapiripirit | 1.050 | 0.899 |
| Kaabong | 1.037 | 2.207 |
| Karenga | 1.024 | 1.096 |
| Amudat | 1.224 | 0.729 |

Both sources agree in every district and both scenarios: no drought. The February no-activation decision was
correct; the January readiness signal was a false alarm everywhere. The sources differ most in Abim and Moroto,
where ECMWF's SPI is far wetter. Napak is closest to the moderate threshold in CHIRPS (-0.45 against -0.68).

Sensitivity: without the 0.25 deg regrid (`--res 0`), CHIRPS district values move by 0.1 to 0.2 (Napak -0.57) and
no outcome changes. Box-mean MAM 2026 rainfall agrees within 1 to 3% between 0.05 and 0.25 deg.

## Caveats

- ERA5 is a model reanalysis, not an observation, and 2026 is ERA5T. It is a sensitivity check; CHIRPS v3 stays
  the primary reference.
- The older `trigger_evaluation.csv` had Abim at -0.821 (a moderate miss). CHIRPS v3 final gives 0.184, so that
  miss does not hold. Which CHIRPS product and version the older file used is not recorded.
- ECMWF's SPI uses a 1991-2020 reference and its own fit, so its values differ in size from an xclim SPI on the
  same rainfall. An earlier xclim SPI-3 on ERA5 monthly precipitation (1991-2018 calibration) gave -0.25 to 1.24
  for MAM 2026, with the same no-drought outcome. Dry-season extremes differ more (Abim, Jan 2026: -2.05 xclim,
  -4.24 ECMWF).
- Forecast probabilities are copied from Tables 2 and 3 of the Del 3 report, not recomputed.
