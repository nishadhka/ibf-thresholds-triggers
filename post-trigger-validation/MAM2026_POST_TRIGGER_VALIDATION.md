# MAM 2026 post-trigger validation (Karamoja)

Compares the SEAS51 SPI-3 trigger decisions for the January (readiness) and February (activation) 2026
initializations with observed MAM 2026 SPI-3. SPI-3 is labelled by the last month of its window, so MAM 2026 is
the value at `2026-05-01`.

## Files

| File | What it does |
|---|---|
| `11-era5-spi3-mam2026-validation.py` | Downloads ERA5 monthly precipitation from the CDS, computes SPI-3, averages it over the nine districts, and scores the Jan and Feb decisions. |
| `trigger_evaluation.csv` | Earlier CHIRPS-based evaluation (seven districts, February decision only). |

Run with `micromamba run -n zarrv3 python 11-era5-spi3-mam2026-validation.py`. It reads `~/.cdsapirc`, writes the
download to `data/` and the results to `out/` (both git-ignored).

## Method

- ERA5 `total_precipitation` monthly means, 0.25 deg, 1991-01 to 2026-09. Rate (m/day) is converted to mm/month.
- SPI-3 uses the same xclim call as `process_chirps_data()` in `../01-run-process-spi.py`: gamma, APP, `floc=0`,
  3-month window, calibration 1991-01-01 to 2018-01-01 (`--cal-end` changes it).
- District value is the mean of the ERA5 pixels whose centre falls inside the district; the nearest pixel to the
  centroid is used if none does. The 0.25 deg grid is coarse for the smaller districts.
- Observed drought: moderate if SPI-3 < -0.68, severe if SPI-3 < -0.84.
- Forecast trigger: district probability above the trigger (Jan 15.2% / 11.1%, Feb 16.2% / 15.2%, moderate / severe).
- Outcomes: Hit, Miss, False alarm, Correct no-action.

## Result (ERA5, MAM 2026 SPI-3)

No district was in moderate or severe drought (SPI-3 from -0.25 to 1.24).

| Initialization | Forecast | Outcome |
|---|---|---|
| January 2026 (readiness) | Triggered in all 9 districts, both scenarios | 9 false alarms per scenario |
| February 2026 (activation) | No trigger in any district | 9 correct no-action per scenario |

The February no-activation decision was correct against ERA5. The January readiness signal was a false alarm in
every district. Abim's CHIRPS value (-0.821, a moderate miss in `trigger_evaluation.csv`) is -0.245 in ERA5, so the
two observation products disagree there.

## Caveats

- ERA5 is a model reanalysis, not an observation. It is a sensitivity check; CHIRPS stays the primary reference.
- SPI-3 here was not identical to the published ERA5-Drought SPI store (`e4drr-project/observations/era5_ecmwf_pencil`,
  1991-2020 reference). Wet-season values agree in sign and rough size; dry-season extremes differ (Abim, Jan 2026:
  -2.05 here, -4.24 in the store). The store ends at 2026-01, so it cannot be used for MAM.
- The CHIRPS routine (`process_chirps_data`, `--mode chirps`) needs Google Cloud credentials for
  `gs://seas51/ea_chirps_v3_monthly_20250312.zarr`, or a local file via `--chirps-file`. That zarr is dated
  2025-03-12 and probably stops before MAM 2026; CHIRPS data to 2026-05 is still to be obtained.
- Forecast probabilities are copied from Tables 2 and 3 of the Del 3 report, not recomputed.
