#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.1.0",
#     "numpy>=1.26.0",
#     "pyarrow>=15.0.0",
#     "gcsfs>=2024.2.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Backfill METAR observations over East Africa from 2024-onwards via Iowa
Mesonet ASOS, writing per-month parquets that mirror the WeatherBench2
schema and partitioning.

Station list is seeded from the WeatherBench2 December 2023 partition
(latest active set), filtered to the EA bounding box used by the GDO
ingest scripts (lat -14.5..25.5, lon 19.5..54.0).

Output layout:
  data/metar_ea_iowa/year=YYYY/month=MM/YYYY-MM.parquet

Usage:
  uv run --python 3.12 metar_ea_backfill_iowa.py \
      --start 2024-01-01 --end 2026-05-08
"""

import argparse
import io
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# East Africa bbox (matches GDO ingest scripts)
LAT_MIN, LAT_MAX = -14.5, 25.5
LON_MIN, LON_MAX = 19.5, 54.0

# WeatherBench2 partition used to seed the station list (latest active set)
WB2_SEED_URL = (
    "gs://weatherbench2/datasets/metar/metar-timeNominal-by-month/"
    "year=2023/month=12/2023-12.parquet"
)

# Iowa Mesonet ASOS request endpoint
IOWA_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

OUT_ROOT = Path("data/metar_ea_iowa")

# Iowa "data=all" returns this canonical set; we keep only the columns that
# overlap WB2 and convert units so the two archives concatenate cleanly.
IOWA_COLS = "tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,skyc1,wxcodes,metar"


def seed_stations():
    """Return DataFrame[stationName, latitude, longitude, locationName] within EA bbox."""
    print(f"Seeding station list from {WB2_SEED_URL}")
    df = pd.read_parquet(WB2_SEED_URL)
    in_box = (
        (df.latitude >= LAT_MIN)
        & (df.latitude <= LAT_MAX)
        & (df.longitude >= LON_MIN)
        & (df.longitude <= LON_MAX)
    )
    grp = (
        df[in_box]
        .groupby("stationName")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            locationName=("locationName", "first"),
        )
        .reset_index()
    )
    print(f"  {len(grp)} EA stations seeded")
    return grp


def fetch_station(station, start, end, attempts=5):
    """One station, all data, return parsed DataFrame (possibly empty)."""
    params = {
        "station": station,
        "data": IOWA_COLS,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year,   "month2": end.month,   "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "yes",
        "missing": "M",
        "trace": "T",
        "direct": "yes",
        "report_type": "3,4",  # METAR + SPECI
    }
    last_err = None
    for attempt in range(attempts):
        try:
            r = requests.get(IOWA_URL, params=params, timeout=180)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    {station}: 429 rate-limit, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.text
            if not text or text.startswith("ERROR") or len(text.splitlines()) < 2:
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            return df
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    {station}: error '{e}', retry in {wait}s")
            time.sleep(wait)
    print(f"    {station}: giving up after {attempts} attempts (last: {last_err})")
    return pd.DataFrame()


def f_to_c(s):
    return (pd.to_numeric(s, errors="coerce") - 32.0) * (5.0 / 9.0)


def kt_to_ms(s):
    return pd.to_numeric(s, errors="coerce") * 0.514444


def inhg_to_hpa(s):
    return pd.to_numeric(s, errors="coerce") * 33.8639


def mi_to_m(s):
    return pd.to_numeric(s, errors="coerce") * 1609.344


def normalise(df, meta):
    """Convert Iowa columns to WB2-equivalent units and column names."""
    if df.empty:
        return df
    out = pd.DataFrame()
    out["stationName"] = df["station"]
    out["locationName"] = meta["locationName"]
    # Iowa CSV emits the lat/lon columns when latlon=yes; fall back to seed.
    out["latitude"] = pd.to_numeric(df.get("lat", meta["latitude"]),
                                    errors="coerce").fillna(meta["latitude"])
    out["longitude"] = pd.to_numeric(df.get("lon", meta["longitude"]),
                                     errors="coerce").fillna(meta["longitude"])
    out["timeObs"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    # Nominal hour (rounded down) matches WB2's monthly partitioning key
    out["timeNominal"] = out["timeObs"].dt.floor("h")
    out["reportType"] = "METAR"
    out["temperature"] = f_to_c(df.get("tmpf"))
    out["dewpoint"] = f_to_c(df.get("dwpf"))
    out["relativeHumidity"] = pd.to_numeric(df.get("relh"), errors="coerce")
    out["windDir"] = pd.to_numeric(df.get("drct"), errors="coerce")
    out["windSpeed"] = kt_to_ms(df.get("sknt"))
    out["windGust"] = kt_to_ms(df.get("gust"))
    out["altimeter"] = inhg_to_hpa(df.get("alti"))
    out["seaLevelPressure"] = pd.to_numeric(df.get("mslp"), errors="coerce")
    out["visibility"] = mi_to_m(df.get("vsby"))
    out["precip1Hour"] = pd.to_numeric(df.get("p01i"), errors="coerce")
    out["skyCover1"] = df.get("skyc1")
    out["weatherCodes"] = df.get("wxcodes")
    out["rawMetar"] = df.get("metar")
    out = out.dropna(subset=["timeObs"]).reset_index(drop=True)
    return out


def write_monthly(df_all):
    """Partition merged frame into year=YYYY/month=MM/YYYY-MM.parquet files."""
    if df_all.empty:
        print("No rows to write.")
        return 0, 0
    df_all["_year"] = df_all["timeObs"].dt.year
    df_all["_month"] = df_all["timeObs"].dt.month
    n_files = 0
    n_rows = 0
    for (yr, mo), part in df_all.groupby(["_year", "_month"]):
        out_dir = OUT_ROOT / f"year={yr}" / f"month={mo}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{yr:04d}-{mo:02d}.parquet"
        part = part.drop(columns=["_year", "_month"]).sort_values(
            ["timeObs", "stationName"]
        )
        part.to_parquet(out_path, index=False)
        n_files += 1
        n_rows += len(part)
        print(f"  wrote {out_path} ({len(part):,} rows)")
    return n_files, n_rows


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    ap = argparse.ArgumentParser(description="Backfill EA METAR from Iowa Mesonet ASOS")
    ap.add_argument("--start", type=parse_date, required=True,
                    help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=parse_date, required=True,
                    help="End date YYYY-MM-DD (exclusive of the next day)")
    ap.add_argument("--sleep", type=float, default=12.0,
                    help="Polite sleep between station fetches (seconds)")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="Limit to these ICAO IDs (default: all EA stations)")
    args = ap.parse_args()

    stations = seed_stations()
    if args.stations:
        stations = stations[stations.stationName.isin(args.stations)].reset_index(drop=True)
        print(f"  filtered to {len(stations)} requested stations")
    if stations.empty:
        sys.exit("No stations to fetch.")

    overall_start = time.time()
    frames = []
    n_with_data = 0
    n_failed = 0
    for i, row in stations.iterrows():
        label = f"[{i+1}/{len(stations)}]"
        station = row["stationName"]
        print(f"{label} {station}  {row['locationName']}")
        df = fetch_station(station, args.start, args.end)
        if df.empty:
            print(f"    no data")
            n_failed += 1
        else:
            norm = normalise(df, row)
            if not norm.empty:
                frames.append(norm)
                n_with_data += 1
                print(f"    {len(norm):,} rows "
                      f"({norm['timeObs'].min()} → {norm['timeObs'].max()})")
            else:
                n_failed += 1
        time.sleep(args.sleep)

    if not frames:
        sys.exit("No observations fetched.")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\nWriting per-month parquets to {OUT_ROOT}/")
    n_files, n_rows = write_monthly(df_all)

    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"  Stations with data: {n_with_data}/{len(stations)}")
    print(f"  Failed/empty:       {n_failed}")
    print(f"  Total rows:         {n_rows:,}")
    print(f"  Monthly parquets:   {n_files}")
    print(f"  Wall time:          {elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
