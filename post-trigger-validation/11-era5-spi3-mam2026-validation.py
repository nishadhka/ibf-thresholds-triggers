"""
Post-trigger validation of the MAM 2026 Karamoja drought trigger against ERA5 SPI-3.

ERA5 monthly total precipitation is downloaded from the CDS, turned into SPI-3 with the same xclim call
as process_chirps_data() in ../01-run-process-spi.py (gamma, APP, floc=0, 3-month window), averaged over the
nine Karamoja districts, and compared with the January (readiness) and February (activation) 2026 SEAS51
trigger decisions.

SPI-3 is labelled by the last month of the window, so MAM 2026 is the value at 2026-05-01.

Usage:
    python 11-era5-spi3-mam2026-validation.py
    python 11-era5-spi3-mam2026-validation.py --cal-start 1991-01-01 --cal-end 2020-12-31
"""
import argparse
import os
import pathlib

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from xclim.indices import standardized_precipitation_index

HERE = pathlib.Path(__file__).resolve().parent
DISTRICTS = HERE.parent / "karamoja_9_districts.geojson"
DATA = HERE / "data"
OUT = HERE / "out"

AREA = [6.0, 32.5, 0.5, 36.0]          # N, W, S, E around Karamoja
TARGET = "2026-05-01"                  # SPI-3 for March-April-May 2026
MODERATE, SEVERE = -0.68, -0.84        # SPI-3 < threshold, as in the AAP

# SEAS51 district probabilities (%) from the MAM 2026 report, Tables 2 and 3.
# (moderate, severe) per district. Triggers: Jan 15.2 / 11.1, Feb 16.2 / 15.2.
FORECAST = {
    "Jan": {"trigger": (15.2, 11.1), "prob": {
        "Abim": (38.8, 35.3), "Napak": (38.9, 31.6), "Nabilatuk": (39.9, 31.5), "Kotido": (40.2, 35.2),
        "Moroto": (37.8, 32.2), "Nakapiripirit": (39.0, 33.9), "Kaabong": (36.6, 30.9),
        "Karenga": (36.5, 29.4), "Amudat": (39.2, 35.1)}},
    "Feb": {"trigger": (16.2, 15.2), "prob": {
        "Abim": (11.1, 8.9), "Napak": (11.0, 8.4), "Nabilatuk": (10.3, 7.7), "Kotido": (11.2, 8.5),
        "Moroto": (10.3, 7.2), "Nakapiripirit": (10.0, 7.0), "Kaabong": (8.9, 7.4),
        "Karenga": (8.8, 7.8), "Amudat": (10.4, 5.5)}},
}


def download_era5(first_year, last_year, path):
    """Monthly mean total precipitation (m/day) on 0.25 deg; skipped if the file already exists."""
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True)
    cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "total_precipitation",
        "year": [str(y) for y in range(first_year, last_year + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": AREA,
        "data_format": "netcdf",
    }, str(path))


def open_precip(path):
    ds = xr.open_dataset(path)
    names = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
    ds = ds.rename({k: v for k, v in names.items() if k in ds.variables or k in ds.dims})
    if "expver" in ds.dims:   # ERA5 / ERA5T are stacked on expver in the newest months
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    tp = ds["tp"].drop_vars([v for v in ("expver", "number") if v in ds.coords])
    tp = tp.assign_coords(time=tp.time.dt.floor("D"))   # CDS stamps the 1st of the month at 06:00
    # mean rate in m/day -> mm per month
    mm = tp * 1000 * tp.time.dt.days_in_month
    mm.attrs["units"] = "mm/month"
    return mm.dropna("time", how="all")


def spi3(precip, cal_start, cal_end):
    return standardized_precipitation_index(
        precip, freq="MS", window=3, dist="gamma", method="APP",
        cal_start=cal_start, cal_end=cal_end, fitkwargs={"floc": 0}).compute()


def district_mean(da, gdf):
    """Mean of the pixels whose centre is in each district; the pixel nearest the centroid if none is."""
    mask = regionmask.mask_geopandas(gdf, da.lon, da.lat)
    rows = {}
    for i, row in gdf.iterrows():
        sel = da.where(mask == i)
        if int(sel.isel(time=0).notnull().sum()) == 0:
            c = row.geometry.centroid
            sel = da.sel(lat=c.y, lon=c.x, method="nearest")
        rows[row["admin2Name"]] = sel.mean([d for d in sel.dims if d != "time"])
    return xr.concat(rows.values(), dim=pd.Index(list(rows), name="district"))


def outcome(forecast_trigger, observed):
    return {(True, True): "Hit", (True, False): "False alarm",
            (False, True): "Miss", (False, False): "Correct no-action"}[(forecast_trigger, observed)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-start", default="1991-01-01")
    ap.add_argument("--cal-end", default="2018-01-01", help="2018-01-01 matches the CHIRPS routine in 01-run-process-spi.py")
    ap.add_argument("--first-year", type=int, default=1991)
    ap.add_argument("--last-year", type=int, default=2026)
    args = ap.parse_args()

    raw = DATA / f"era5_tp_monthly_{args.first_year}_{args.last_year}.nc"
    download_era5(args.first_year, args.last_year, raw)
    precip = open_precip(raw)
    print("ERA5 precipitation:", str(precip.time.values[0])[:7], "to", str(precip.time.values[-1])[:7])
    if np.datetime64(TARGET) not in precip.time.values:
        raise SystemExit(f"{TARGET} is not in the ERA5 download yet")

    gdf = gpd.read_file(DISTRICTS)
    obs = district_mean(spi3(precip, args.cal_start, args.cal_end), gdf)
    OUT.mkdir(exist_ok=True)
    obs.to_dataframe(name="era5_spi3").reset_index().to_csv(OUT / "era5_district_spi3_monthly.csv", index=False)

    rows = []
    for d in gdf["admin2Name"]:
        v = float(obs.sel(district=d, time=TARGET))
        mod, sev = v < MODERATE, v < SEVERE
        row = {"district": d, "era5_spi3_mam2026": round(v, 3),
               "obs_moderate_drought": "Yes" if mod else "No", "obs_severe_drought": "Yes" if sev else "No"}
        for init, f in FORECAST.items():
            pm, ps = f["prob"][d]
            tm, ts = pm > f["trigger"][0], ps > f["trigger"][1]
            row[f"{init}_fc_moderate_trigger"] = "Yes" if tm else "No"
            row[f"{init}_fc_severe_trigger"] = "Yes" if ts else "No"
            row[f"{init}_moderate_outcome"] = outcome(tm, mod)
            row[f"{init}_severe_outcome"] = outcome(ts, sev)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "era5_trigger_evaluation_mam2026.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
