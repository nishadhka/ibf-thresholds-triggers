#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.1.0",
#     "numpy>=1.26.0",
#     "pyarrow>=15.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Last-24h METAR live feed for East Africa via NOAA Aviation Weather Center.

One bbox-scoped HTTP request returns every METAR posted in the last N hours
(max 72) across the East Africa bounding box used by the GDO ingest scripts.
Output is normalised to the same schema and units as
metar_ea_backfill_iowa.py so the two archives concatenate cleanly.

Output:
  data/metar_ea_live/YYYY-MM-DDTHHMMSSZ.parquet   (one file per run)

Usage:
  uv run --python 3.12 metar_ea_live_24h.py             # last 24h
  uv run --python 3.12 metar_ea_live_24h.py --hours 6 --print
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# East Africa bbox (matches GDO ingest scripts)
LAT_MIN, LAT_MAX = -14.5, 25.5
LON_MIN, LON_MAX = 19.5, 54.0

# NOAA AWC METAR API — JSON, bbox-scoped, no auth
AWC_URL = "https://aviationweather.gov/api/data/metar"

OUT_ROOT = Path("data/metar_ea_live")


def fetch(hours):
    params = {
        # AWC expects bbox as "minLat,minLon,maxLat,maxLon"
        "bbox": f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}",
        "format": "json",
        "hours": hours,
        "taf": "false",
    }
    r = requests.get(AWC_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response: {type(data).__name__}")
    return data


def normalise(records):
    """Return DataFrame matching the backfill schema (units: °C, m/s, hPa, m)."""
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame(records)

    def col(name, default=np.nan):
        return raw[name] if name in raw.columns else pd.Series([default] * len(raw))

    out = pd.DataFrame()
    out["stationName"] = col("icaoId")
    out["locationName"] = col("name")
    out["latitude"] = pd.to_numeric(col("lat"), errors="coerce")
    out["longitude"] = pd.to_numeric(col("lon"), errors="coerce")
    out["elevation"] = pd.to_numeric(col("elev"), errors="coerce")

    # AWC times are epoch seconds (obsTime / reportTime)
    def to_dt(s):
        return pd.to_datetime(pd.to_numeric(s, errors="coerce"),
                              unit="s", utc=True, errors="coerce")
    out["timeObs"] = to_dt(col("obsTime"))
    out["timeNominal"] = out["timeObs"].dt.floor("h")
    out["timeReport"] = to_dt(col("reportTime"))

    out["reportType"] = col("metarType", "METAR")
    out["temperature"] = pd.to_numeric(col("temp"), errors="coerce")
    out["dewpoint"] = pd.to_numeric(col("dewp"), errors="coerce")
    # Relative humidity isn't returned by AWC; compute from T/Td (Magnus).
    t = out["temperature"]
    td = out["dewpoint"]
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    out["relativeHumidity"] = (100 * e / es).clip(0, 100)

    out["windDir"] = pd.to_numeric(col("wdir"), errors="coerce")
    # AWC wspd / wgst are in knots → convert to m/s
    out["windSpeed"] = pd.to_numeric(col("wspd"), errors="coerce") * 0.514444
    out["windGust"] = pd.to_numeric(col("wgst"), errors="coerce") * 0.514444
    out["altimeter"] = pd.to_numeric(col("altim"), errors="coerce")
    out["seaLevelPressure"] = pd.to_numeric(col("slp"), errors="coerce")
    # AWC visib is statute miles ("10+" → 10) → convert to metres
    vis_raw = col("visib").astype(str).str.replace("+", "", regex=False)
    out["visibility"] = pd.to_numeric(vis_raw, errors="coerce") * 1609.344
    out["precip1Hour"] = pd.to_numeric(col("precip"), errors="coerce")
    out["weatherCodes"] = col("wxString")
    out["rawMetar"] = col("rawOb")

    out = out.dropna(subset=["timeObs"]).reset_index(drop=True)
    # Defensive bbox filter (AWC occasionally returns a station just outside)
    in_box = (
        (out["latitude"] >= LAT_MIN) & (out["latitude"] <= LAT_MAX)
        & (out["longitude"] >= LON_MIN) & (out["longitude"] <= LON_MAX)
    )
    return out[in_box].reset_index(drop=True)


def write_run(df, run_time):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = run_time.strftime("%Y-%m-%dT%H%M%SZ")
    out_path = OUT_ROOT / f"{stamp}.parquet"
    df.sort_values(["timeObs", "stationName"]).to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Last 24h EA METAR via NOAA AWC")
    ap.add_argument("--hours", type=int, default=24,
                    help="Hours of history to request (max 72)")
    ap.add_argument("--print", dest="echo", action="store_true",
                    help="Print rows to stdout instead of writing parquet")
    args = ap.parse_args()

    hours = min(max(args.hours, 1), 72)
    run_time = datetime.now(timezone.utc)
    print(f"Fetching last {hours}h of METAR over EA "
          f"bbox=({LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}) at {run_time.isoformat()}")
    records = fetch(hours)
    print(f"  {len(records):,} raw records returned")
    df = normalise(records)
    if df.empty:
        sys.exit("No observations after normalisation/bbox filter.")
    print(f"  {len(df):,} obs from {df.stationName.nunique()} stations, "
          f"{df.timeObs.min()} → {df.timeObs.max()}")

    if args.echo:
        cols = ["stationName", "locationName", "latitude", "longitude",
                "elevation", "timeObs", "timeNominal", "timeReport",
                "reportType", "temperature", "dewpoint", "relativeHumidity",
                "windSpeed", "windGust", "altimeter"]
        print(df[cols].to_string(index=False))
        return

    out_path = write_run(df, run_time)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
