# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=2",
#     "pystac",
#     "xarray",
#     "zarr>=3",
#     "geopandas",
#     "regionmask",
#     "xclim",
#     "netcdf4",
# ]
# ///
"""
Post-trigger validation of the MAM 2026 Karamoja drought trigger against observed SPI-3, read from public
Icechunk stores (no credentials):

  chirps  dynamical.org ucsb-chc-chirps-analysis-final: CHIRPS v3 final daily precip, 0.05 deg. Summed to months,
          conservatively regridded to 0.25 deg and turned into SPI-3 as process_chirps_data() in
          ../01-run-process-spi.py does with CHIRPS v3 monthly (gamma, APP, floc=0, 3-month window,
          calibration 1991-01-01 to 2018-01-01).
  era5    source.coop e4drr-project/observations/era5_ecmwf_icechunk: ERA5-Drought SPI3 from ECMWF, 0.25 deg,
          1991-2020 reference. Used as published, not recomputed.

Each SPI-3 field is averaged over the nine Karamoja districts and compared with the January (readiness) and
February (activation) 2026 SEAS51 trigger decisions. SPI-3 is labelled by the last month of the window, so
MAM 2026 is the value at 2026-05-01.

Usage:
    uv run 11-spi3-mam2026-validation.py                  # both sources and a side-by-side comparison
    uv run 11-spi3-mam2026-validation.py --obs chirps
    uv run 11-spi3-mam2026-validation.py --obs era5
"""
import argparse
import pathlib

import geopandas as gpd
import icechunk
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from xclim.indices import standardized_precipitation_index

HERE = pathlib.Path(__file__).resolve().parent
DISTRICTS = HERE.parent / "karamoja_9_districts.geojson"
DATA = HERE / "data"
OUT = HERE / "out"

ERA5_STORE = "e4drr-project/observations/era5_ecmwf_icechunk"
CHIRPS_STAC = ("https://stac.dynamical.org/catalog.json", "ucsb-chc-chirps-analysis-final")
BBOX = {"lat": (0.5, 6.0), "lon": (32.5, 36.0)}   # around Karamoja
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


def open_era5():
    storage = icechunk.s3_storage(bucket="us-west-2.opendata.source.coop", prefix=ERA5_STORE,
                                  region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(storage)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def open_chirps():
    """CHIRPS v3 final daily from dynamical.org; the store URL comes from its STAC collection."""
    import pystac

    catalog, collection = CHIRPS_STAC
    href = pystac.Catalog.from_file(catalog).get_child(collection).assets["icechunk-https"].href
    repo = icechunk.Repository.open(icechunk.http_storage(href))
    ds = xr.open_zarr(repo.readonly_session("main").store, chunks=None)
    return ds.rename({"latitude": "lat", "longitude": "lon"})


def subset(ds, pad=0.0):
    """Karamoja box (widened by pad), whichever way lat and lon are ordered in the store."""
    sel = {}
    for dim, (lo, hi) in BBOX.items():
        lo, hi = lo - pad, hi + pad
        sel[dim] = slice(hi, lo) if ds[dim][0] > ds[dim][-1] else slice(lo, hi)
    return ds.sel(sel)


def overlap_weights(src, dst, res):
    """(len(dst), len(src)) fraction of each destination cell covered by each source cell."""
    s_res = abs(float(src[1] - src[0]))
    s_lo, s_hi = src - s_res / 2, src + s_res / 2
    d_lo, d_hi = dst - res / 2, dst + res / 2
    return np.clip(np.minimum(d_hi[:, None], s_hi) - np.maximum(d_lo[:, None], s_lo), 0, None)


def regrid_conservative(da, res):
    """Area-weighted regrid onto the grid process_chirps_data() builds with xesmf "conservative":
    cell centres np.arange(lat_min, lat_max, res) and np.arange(lon_min, lon_max, res)."""
    lat = np.arange(*BBOX["lat"], res)
    lon = np.arange(*BBOX["lon"], res)
    wlat = overlap_weights(da.lat.values, lat, res) * np.cos(np.deg2rad(da.lat.values))
    wlon = overlap_weights(da.lon.values, lon, res)
    valid = np.isfinite(da.values)
    num = np.einsum("ki,tij,lj->tkl", wlat, np.where(valid, da.values, 0.0), wlon)
    den = np.einsum("ki,tij,lj->tkl", wlat, valid.astype(float), wlon)
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return xr.DataArray(out, dims=("time", "lat", "lon"),
                        coords={"time": da.time.values, "lat": lat, "lon": lon}, attrs=da.attrs)


def chirps_monthly(first_year, last_month):
    """CHIRPS v3 monthly totals (mm) over the Karamoja box, cached in data/; months with a missing day are dropped."""
    cache = DATA / f"chirps_v3_karamoja_monthly_{first_year}_{last_month[:7]}.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    print(f"Reading CHIRPS v3 daily {first_year}-01 to {last_month[:7]} from dynamical.org")
    end = pd.Timestamp(last_month) + pd.offsets.MonthEnd(0)
    # pad by one 0.25 deg cell so the regrid has full source cells at the edges
    daily = subset(open_chirps()["precipitation_surface"], pad=0.25).sel(time=slice(f"{first_year}-01-01", end))
    daily = (daily * 86400).load()   # kg m-2 s-1 -> mm/day
    total = daily.resample(time="MS").sum(min_count=1)
    days = daily.time.resample(time="MS").count()
    monthly = total.where(days == days.time.dt.days_in_month, drop=True).drop_attrs()
    monthly.attrs ={"units": "mm/month", "source": "CHIRPS v3 final daily via dynamical.org"}
    DATA.mkdir(exist_ok=True)
    monthly.to_netcdf(cache)
    return monthly


def chirps_spi3(cal_start, cal_end, first_year, res):
    precip = chirps_monthly(first_year, TARGET)
    precip = regrid_conservative(precip, res) if res else subset(precip)
    precip.attrs["units"] = "mm/month"
    return standardized_precipitation_index(
        precip, freq="MS", window=3, dist="gamma", method="APP",
        cal_start=cal_start, cal_end=cal_end, fitkwargs={"floc": 0}).compute()


def era5_spi3(first_year):
    return subset(open_era5()["SPI3"]).sel(time=slice(f"{first_year}-01-01", None)).load()


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


def evaluate(obs, name, districts):
    rows = []
    for d in districts:
        v = float(obs.sel(district=d, time=TARGET))
        mod, sev = v < MODERATE, v < SEVERE
        row = {"district": d, f"{name}_spi3_mam2026": round(v, 3),
               "obs_moderate_drought": "Yes" if mod else "No", "obs_severe_drought": "Yes" if sev else "No"}
        for init, f in FORECAST.items():
            pm, ps = f["prob"][d]
            tm, ts = pm > f["trigger"][0], ps > f["trigger"][1]
            row[f"{init}_fc_moderate_trigger"] = "Yes" if tm else "No"
            row[f"{init}_fc_severe_trigger"] = "Yes" if ts else "No"
            row[f"{init}_moderate_outcome"] = outcome(tm, mod)
            row[f"{init}_severe_outcome"] = outcome(ts, sev)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["chirps", "era5", "both"], default="both")
    ap.add_argument("--cal-start", default="1991-01-01", help="CHIRPS SPI calibration start")
    ap.add_argument("--cal-end", default="2018-01-01", help="2018-01-01 matches the CHIRPS routine in 01-run-process-spi.py")
    ap.add_argument("--first-year", type=int, default=1991)
    ap.add_argument("--res", type=float, default=0.25,
                    help="CHIRPS regrid resolution before SPI, as in process_chirps_data(); 0 keeps 0.05 deg")
    args = ap.parse_args()

    gdf = gpd.read_file(DISTRICTS)
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    names = ["chirps", "era5"] if args.obs == "both" else [args.obs]
    results = {}
    for name in names:
        spi = chirps_spi3(args.cal_start, args.cal_end, args.first_year, args.res) if name == "chirps" else era5_spi3(args.first_year)
        print(f"{name} SPI-3: {str(spi.time.values[0])[:7]} to {str(spi.time.values[-1])[:7]}")
        if np.datetime64(TARGET) not in spi.time.values:
            print(f"  {TARGET} is not in the {name} store yet, skipped")
            continue
        obs = district_mean(spi, gdf)
        obs.to_dataframe(name=f"{name}_spi3").reset_index().to_csv(OUT / f"{name}_district_spi3_monthly.csv", index=False)
        res = evaluate(obs, name, gdf["admin2Name"])
        res.to_csv(OUT / f"{name}_trigger_evaluation_mam2026.csv", index=False)
        print(res.to_string(index=False), "\n")
        results[name] = res

    if len(results) == 2:
        c, e = results["chirps"], results["era5"]
        cmp = pd.DataFrame({"district": c["district"],
                            "chirps_spi3": c["chirps_spi3_mam2026"], "era5_spi3": e["era5_spi3_mam2026"]})
        for sc in ("moderate", "severe"):
            cmp[f"chirps_{sc}"] = c[f"obs_{sc}_drought"]
            cmp[f"era5_{sc}"] = e[f"obs_{sc}_drought"]
            cmp[f"{sc}_agree"] = np.where(cmp[f"chirps_{sc}"] == cmp[f"era5_{sc}"], "Yes", "No")
        cmp.to_csv(OUT / "chirps_vs_era5_mam2026.csv", index=False)
        print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
