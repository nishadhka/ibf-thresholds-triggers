# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icechunk>=2",
#     "xarray",
#     "zarr>=3",
#     "netcdf4",
#     "cdsapi",
# ]
# ///
"""
Bring the ERA5-Drought SPI store on source.coop up to date, so 11-spi3-mam2026-validation.py can read the
months it needs.

Store: e4drr-project/observations/era5_ecmwf_icechunk (SPI1 to SPI48, 0.25 deg, 1991-2020 reference).
The download and append steps are taken from thresholds/ecmwf_spi/ecmwf_spi_icechunk.py on the xarray-method
branch, which builds the store from scratch (init, fill, rechunk). Only the update path is kept here.

  download  CDS derived-drought-historical-monthly, every SPI period, the given months, unzipped to data/era5_spi/.
            Reads .cdsapirc in this folder if present, else ~/.cdsapirc.
  append    adds the months after the store's last month, all SPI periods in one commit. Writes through
            https://data.source.coop with credentials from the source-coop CLI; run
            `source-coop login --duration 12h` once first.

Usage:
    uv run 10-era5-spi-store-update.py download --start 2026-10 --end 2026-10
    uv run 10-era5-spi-store-update.py append --dry-run
    uv run 10-era5-spi-store-update.py append
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time
import zipfile
from datetime import datetime

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "era5_spi"

SPI_PERIODS = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]
CDS_DATASET = "derived-drought-historical-monthly"
CDS_AREA = [25, 20, -15, 53]           # N, W, S, E: the store grid

SOURCE_COOP_PROXY = "https://data.source.coop"
SOURCE_COOP_BUCKET = "e4drr-project"
SOURCE_COOP_PREFIX = "observations/era5_ecmwf_icechunk"
COMMIT_RETRIES = 5


def source_coop_cli_credentials():
    """Credentials from `source-coop creds`, refreshed from the cached login."""
    cli = os.getenv("SOURCE_COOP_CLI", os.path.expanduser("~/.local/bin/source-coop"))
    out = subprocess.run([cli, "creds", "--format", "credential-process"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"no source.coop login: {out.stderr.strip()}\n"
                         "run: source-coop login --duration 12h")
    c = json.loads(out.stdout)
    return icechunk.S3StaticCredentials(
        access_key_id=c["AccessKeyId"],
        secret_access_key=c["SecretAccessKey"],
        session_token=c["SessionToken"],
        expires_after=datetime.fromisoformat(c["Expiration"].replace("Z", "+00:00")),
    )


def open_repo():
    storage = icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET, prefix=SOURCE_COOP_PREFIX, region="us-west-2",
        endpoint_url=SOURCE_COOP_PROXY, force_path_style=True,
        get_credentials=source_coop_cli_credentials)
    return icechunk.Repository.open(storage)


def download(args):
    import cdsapi

    rc = HERE / ".cdsapirc"
    if rc.exists():
        os.environ.setdefault("CDSAPI_RC", str(rc))
    months = pd.date_range(args.start, args.end, freq="MS")
    DATA.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for spi in SPI_PERIODS:
        for year, ym in months.groupby(months.year).items():
            target = DATA / f"{spi}_{year}_{ym[0].month:02d}-{ym[-1].month:02d}.zip"
            if not target.exists():
                print(f"CDS {spi} {year} months {[t.month for t in ym]}")
                client.retrieve(CDS_DATASET, {
                    "variable": ["standardised_precipitation_index"],
                    "accumulation_period": [spi.removeprefix("SPI")],
                    "version": "1_0",
                    "product_type": ["reanalysis"],
                    "dataset_type": args.dataset_type,
                    "year": [str(year)],
                    "month": [f"{t.month:02d}" for t in ym],
                    "area": CDS_AREA,
                }, str(target))
            with zipfile.ZipFile(target) as z:
                z.extractall(DATA)
    print(f"Downloaded to {DATA}")


def catalog_files():
    """{spi: {Timestamp: path}} for files named SPI<n>_..._<YYYYMM>....nc (era5 and era5t)."""
    pattern = re.compile(r"^(SPI\d+)_.*?_(\d{4})(\d{2})(?:\D.*)?\.nc$")
    catalog = {}
    for f in sorted(DATA.glob("SPI*.nc")):
        m = pattern.match(f.name)
        if m and m.group(1) in SPI_PERIODS:
            t = pd.Timestamp(year=int(m.group(2)), month=int(m.group(3)), day=1)
            catalog.setdefault(m.group(1), {})[t] = f
    return catalog


def append(args):
    repo = open_repo()
    ds_store = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    last = pd.Timestamp(ds_store.time.values[-1])
    print(f"Store ends {last:%Y-%m}")

    catalog = catalog_files()
    new_times = sorted({t for files in catalog.values() for t in files if t > last})
    if not new_times:
        print(f"Nothing newer than the store in {DATA}")
        return
    expected = list(pd.date_range(last + pd.offsets.MonthBegin(1), new_times[-1], freq="MS"))
    if expected != new_times:
        raise SystemExit(f"new months are not contiguous after {last:%Y-%m}: {[f'{t:%Y-%m}' for t in new_times]}")

    data_vars = {}
    for spi in ds_store.data_vars:
        files = catalog.get(spi, {})
        missing = [f"{t:%Y-%m}" for t in new_times if t not in files]
        if missing:
            raise SystemExit(f"{spi} has no file for {missing}")
        layers = []
        for t in new_times:
            with xr.open_dataset(files[t]) as ds:
                da = ds[[v for v in ds.data_vars if v.startswith("SPI")][0]].squeeze(drop=True)
                # NaN where the file's grid differs from the store's, so check the count below
                da = da.reindex(lat=ds_store.lat.values, lon=ds_store.lon.values, method="nearest", tolerance=1e-3)
                layers.append(da.values.astype(np.float32))
        data_vars[spi] = (("time", "lat", "lon"), np.stack(layers))
        print(f"  {spi}: {len(new_times)} months, {int(np.isfinite(data_vars[spi][1]).sum())} valid values")

    ds_new = xr.Dataset(data_vars, coords={"time": np.array(new_times, dtype="datetime64[ns]")})
    message = f"append {new_times[0]:%Y-%m}..{new_times[-1]:%Y-%m}"
    if args.dry_run:
        print(f"DRY RUN: would commit '{message}'")
        return

    # The source.coop proxy sometimes returns an empty error body; retry with a fresh session.
    # If a lost response hid a commit that landed, the tip already carries the message.
    for attempt in range(1, COMMIT_RETRIES + 1):
        if next(iter(repo.ancestry(branch="main"))).message == message:
            break
        try:
            session = repo.writable_session("main")
            ds_new.to_zarr(session.store, append_dim="time", consolidated=False)
            print(f"Committed {message}: {session.commit(message)}")
            break
        except Exception as e:
            if attempt == COMMIT_RETRIES:
                raise
            print(f"  attempt {attempt} failed ({e}); retrying")
            time.sleep(10 * attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Fetch months from the CDS")
    p.add_argument("--start", required=True, help="First month, YYYY-MM")
    p.add_argument("--end", required=True, help="Last month, YYYY-MM")
    p.add_argument("--dataset-type", default="intermediate_dataset",
                   choices=["intermediate_dataset", "consolidated_dataset"],
                   help="Recent months are only in the intermediate (ERA5T) dataset")
    p = sub.add_parser("append", help="Append downloaded months to the source.coop store")
    p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"download": download, "append": append}[args.command](args)


if __name__ == "__main__":
    main()
