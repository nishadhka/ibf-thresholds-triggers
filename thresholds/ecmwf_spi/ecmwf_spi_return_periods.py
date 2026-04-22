#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netcdf4",
#     "scipy",
#     "matplotlib",
#     "pandas",
#     "icechunk>=0.1",
#     "zarr>=3",
#     "s3fs",
#     "python-dotenv",
#     "cartopy",
#     "shapely",
# ]
# ///
"""
ECMWF SPI — Drought Return Period Analysis (Normal Distribution)
=================================================================

Computes drought return period thresholds for every pixel using the
ECMWF ERA5-Drought SPI dataset.  Unlike precipitation return periods
(which use Gumbel for annual maxima), SPI is already a standardized
index designed to follow a normal distribution, so we use:

  1. **Standard Normal** thresholds (theoretical):
     SPI is defined to be N(0,1) by construction. Return period T
     maps to the left-tail quantile:
       SPI_threshold = Φ⁻¹(1/T)
     where Φ⁻¹ is the inverse CDF (ppf) of the standard normal.

  2. **Empirical (fitted)** thresholds:
     Fit N(μ, σ) to each pixel's actual SPI series and compute
     quantiles from the fitted distribution.  This accounts for any
     residual non-normality in the gamma-fitted SPI.

Return period categories (drought = negative SPI tail):
  - 3-year RP  (Mild):        p = 1/3  → SPI ≈ -0.43
  - 5-year RP  (Moderate):    p = 1/5  → SPI ≈ -0.84
  - 10-year RP (Severe):      p = 1/10 → SPI ≈ -1.28
  - 20-year RP (Extreme):     p = 1/20 → SPI ≈ -1.64
  - 50-year RP (Exceptional): p = 1/50 → SPI ≈ -2.05

Subcommands:

  compute   — Compute return period thresholds for all pixels
  verify    — Validate output NetCDF
  plot      — Generate SPI distribution + threshold plots (as in screenshot)

Usage:
    # Compute thresholds from local NetCDF files
    uv run ecmwf_spi_return_periods.py compute \
        --data-dir data/ --output ecmwf_spi_return_periods.nc

    # Verify output
    uv run ecmwf_spi_return_periods.py verify \
        --input ecmwf_spi_return_periods.nc

    # Plot SPI distribution for a specific location
    uv run ecmwf_spi_return_periods.py plot \
        --input ecmwf_spi_return_periods.nc --lat 9.0 --lon 40.0

Author: AI Assistant
Date: 2026-02-17
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ecmwf_spi_return_periods.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

# SPI accumulation periods
SPI_PERIODS = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]

# source.coop S3 settings
S3_BUCKET = "us-west-2.opendata.source.coop"
S3_REGION = "us-west-2"
PENCIL_S3_PREFIX = "e4drr-project/observations/era5_ecmwf_pencil"
RP_ICECHUNK_S3_PREFIX = "e4drr-project/observations/era5_ecmwf_rp_icechunk"

# Drought return periods (years) — left tail of the distribution
RETURN_PERIODS = [3, 5, 10, 20, 50]

# Labels for return period severity categories
RP_LABELS = {
    3: "Mild",
    5: "Moderate",
    10: "Severe",
    20: "Extreme",
    50: "Exceptional",
}

# Colors for severity bands in plots
RP_COLORS = {
    3: "#FFD700",      # gold/yellow — mild
    5: "#FF8C00",      # dark orange — moderate
    10: "#FF4500",     # orange-red — severe
    20: "#DC143C",     # crimson — extreme
    50: "#8B0000",     # dark red — exceptional
}


# ─── Computation ────────────────────────────────────────────────────────────


def discover_files(data_dir: str):
    """Discover SPI NetCDF files grouped by period.

    Returns dict: {spi_period: [filepath, ...]} sorted by time.
    """
    import pandas as pd

    data_path = Path(data_dir)
    catalog = {}

    for spi in SPI_PERIODS:
        pattern = f"{spi}_gamma_global_era5_moda_ref1991to2020_*.nc"
        files = sorted(data_path.glob(pattern))
        if files:
            catalog[spi] = [str(f) for f in files]
            logger.info(f"  {spi}: {len(files)} files")

    return catalog


def compute_return_periods(args):
    """Compute drought return period thresholds for all pixels.

    For each SPI period and each pixel:
      1. Load the full time series
      2. Fit a normal distribution N(μ, σ)
      3. Compute SPI thresholds for each return period using:
         - Standard normal: Φ⁻¹(1/T)
         - Fitted normal: μ + σ × Φ⁻¹(1/T)
      4. Compute percentile-based thresholds (empirical)
    """
    import xarray as xr
    from scipy import stats

    logger.info("=" * 70)
    logger.info("COMPUTE: SPI Drought Return Period Analysis")
    logger.info("=" * 70)
    overall_start = time.time()

    # Discover files
    catalog = discover_files(args.data_dir)
    if not catalog:
        logger.error("No SPI files found!")
        return

    # Get grid from first file
    first_spi = list(catalog.keys())[0]
    ds_sample = xr.open_dataset(catalog[first_spi][0])
    lat = ds_sample["lat"].values
    lon = ds_sample["lon"].values
    n_lat = len(lat)
    n_lon = len(lon)
    ds_sample.close()
    logger.info(f"  Grid: {n_lat} lat × {n_lon} lon")

    n_spi = len(catalog)
    n_rp = len(RETURN_PERIODS)

    # Standard normal thresholds (same for all pixels, by definition)
    standard_thresholds = np.array(
        [stats.norm.ppf(1.0 / T) for T in RETURN_PERIODS], dtype=np.float32,
    )
    logger.info(f"  Standard normal thresholds: {dict(zip(RETURN_PERIODS, standard_thresholds))}")

    # Output arrays
    spi_names_out = []
    fitted_thresholds = np.full((n_spi, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
    empirical_thresholds = np.full((n_spi, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
    fit_mu = np.full((n_spi, n_lat, n_lon), np.nan, dtype=np.float32)
    fit_sigma = np.full((n_spi, n_lat, n_lon), np.nan, dtype=np.float32)
    n_valid_months = np.full((n_spi, n_lat, n_lon), 0, dtype=np.int32)

    for s_idx, (spi, files) in enumerate(catalog.items()):
        spi_names_out.append(spi)
        logger.info(f"\n--- Processing {spi} ({len(files)} files) ---")
        t0 = time.time()

        # Load all files for this SPI period and concatenate
        datasets = []
        for f in files:
            ds = xr.open_dataset(f)
            var_name = [v for v in ds.data_vars if v.startswith("SPI")][0]
            datasets.append(ds[[var_name]])

        ds_all = xr.concat(datasets, dim="time").sortby("time")
        for d in datasets:
            d.close()

        var_name = [v for v in ds_all.data_vars][0]
        data = ds_all[var_name].values.astype(np.float32)  # (n_time, n_lat, n_lon)
        n_time = data.shape[0]
        logger.info(f"  Loaded: {n_time} months, shape {data.shape}")

        # Process each pixel
        for i in range(n_lat):
            for j in range(n_lon):
                ts = data[:, i, j]
                valid = ts[~np.isnan(ts)]
                n_valid = len(valid)
                n_valid_months[s_idx, i, j] = n_valid

                if n_valid < 30:
                    continue

                # Fit normal distribution
                mu = float(np.mean(valid))
                sigma = float(np.std(valid, ddof=1))
                fit_mu[s_idx, i, j] = mu
                fit_sigma[s_idx, i, j] = sigma

                if sigma < 1e-6:
                    continue

                # Fitted thresholds: μ + σ × Φ⁻¹(1/T)
                for r_idx, T in enumerate(RETURN_PERIODS):
                    z = stats.norm.ppf(1.0 / T)
                    fitted_thresholds[s_idx, r_idx, i, j] = mu + sigma * z

                    # Empirical: percentile-based
                    pct = 100.0 / T
                    empirical_thresholds[s_idx, r_idx, i, j] = np.percentile(valid, pct)

        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s")
        ocean_frac = (n_valid_months[s_idx] < 30).mean()
        logger.info(f"  Ocean/missing fraction: {ocean_frac*100:.1f}%")

        del ds_all, data

    # ── Build output Dataset ──
    logger.info("\nBuilding output Dataset...")

    ds_out = xr.Dataset(
        {
            "fitted_threshold": (
                ["spi_period", "return_period", "lat", "lon"],
                fitted_thresholds,
                {
                    "long_name": "SPI drought threshold (fitted normal)",
                    "units": "dimensionless",
                    "description": (
                        "SPI value below which drought of given return period occurs. "
                        "Computed as mu + sigma * ppf(1/T) from fitted N(mu, sigma)."
                    ),
                },
            ),
            "empirical_threshold": (
                ["spi_period", "return_period", "lat", "lon"],
                empirical_thresholds,
                {
                    "long_name": "SPI drought threshold (empirical percentile)",
                    "units": "dimensionless",
                    "description": (
                        "SPI value at the (1/T)×100 percentile of the observed series."
                    ),
                },
            ),
            "standard_threshold": (
                ["return_period"],
                standard_thresholds,
                {
                    "long_name": "SPI drought threshold (standard normal)",
                    "units": "dimensionless",
                    "description": "ppf(1/T) from N(0,1). Same for all pixels by SPI definition.",
                },
            ),
            "fit_mu": (
                ["spi_period", "lat", "lon"],
                fit_mu,
                {"long_name": "Fitted normal mean (mu)", "units": "dimensionless"},
            ),
            "fit_sigma": (
                ["spi_period", "lat", "lon"],
                fit_sigma,
                {"long_name": "Fitted normal std dev (sigma)", "units": "dimensionless"},
            ),
            "n_valid_months": (
                ["spi_period", "lat", "lon"],
                n_valid_months,
                {"long_name": "Number of valid (non-NaN) monthly values"},
            ),
        },
        coords={
            "spi_period": (
                "spi_period", spi_names_out,
                {"long_name": "SPI accumulation period"},
            ),
            "return_period": (
                "return_period", RETURN_PERIODS,
                {"units": "years", "long_name": "Return period"},
            ),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF ERA5-Drought SPI — Drought Return Period Thresholds",
            "source": "ECMWF CDS derived-drought-historical-monthly",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": f"Created {datetime.now().isoformat()}",
            "distribution": "Normal (standard and fitted)",
            "methodology": (
                "Three threshold types: (1) Standard normal ppf(1/T), "
                "(2) Fitted N(mu,sigma) per pixel, (3) Empirical percentile. "
                "Drought = left tail (negative SPI values)."
            ),
            "return_period_labels": str(RP_LABELS),
            "Conventions": "CF-1.8",
        },
    )

    output_path = args.output
    logger.info(f"Writing to {output_path}...")
    ds_out.to_netcdf(output_path)
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("COMPUTE COMPLETE")
    logger.info(f"  Output: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"  Dimensions: {dict(ds_out.sizes)}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 70)


# ─── Verify ─────────────────────────────────────────────────────────────────


def run_verify(args):
    """Validate the output NetCDF for physical reasonableness."""
    import xarray as xr
    from scipy import stats

    logger.info("=" * 70)
    logger.info("VERIFY: Checking SPI return period output")
    logger.info("=" * 70)

    ds = xr.open_dataset(args.input)
    logger.info(f"Dataset:\n{ds}")

    fitted = ds["fitted_threshold"]
    empirical = ds["empirical_threshold"]
    standard = ds["standard_threshold"]

    logger.info(f"\n--- Standard Normal Thresholds ---")
    for r_idx, rp in enumerate(ds.return_period.values):
        val = float(standard.values[r_idx])
        label = RP_LABELS.get(int(rp), "")
        logger.info(f"  {int(rp)}-yr ({label}): SPI = {val:.3f}")

    # NaN fraction
    logger.info(f"\n--- NaN Fraction (fitted thresholds) ---")
    for s_idx, spi in enumerate(ds.spi_period.values):
        nan_frac = float(np.isnan(fitted.values[s_idx]).mean())
        logger.info(f"  {spi}: {nan_frac*100:.1f}% NaN")

    # Monotonicity: more extreme RP → more negative threshold
    logger.info(f"\n--- Monotonicity Check (RP ordering) ---")
    for s_idx, spi in enumerate(ds.spi_period.values):
        layer = fitted.values[s_idx]  # (n_rp, n_lat, n_lon)
        violations = 0
        valid_pixels = 0
        for r_idx in range(1, layer.shape[0]):
            mask = ~np.isnan(layer[r_idx]) & ~np.isnan(layer[r_idx - 1])
            valid_pixels += int(mask.sum())
            # Higher RP should give MORE negative SPI
            violations += int((layer[r_idx][mask] > layer[r_idx - 1][mask]).sum())
        status = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
        logger.info(f"  {spi}: {status}")

    # Addis Ababa / Ethiopian spot check
    logger.info(f"\n--- Addis Ababa Spot Check (lat≈9.0, lon≈38.75) ---")
    try:
        # Find nearest indices manually (lat may be descending)
        lat_idx = int(np.argmin(np.abs(ds.lat.values - 9.0)))
        lon_idx = int(np.argmin(np.abs(ds.lon.values - 38.75)))
        logger.info(f"  Nearest pixel: lat={float(ds.lat.values[lat_idx]):.2f}, lon={float(ds.lon.values[lon_idx]):.2f}")
        for s_idx, spi in enumerate(ds.spi_period.values):
            vals = [f"{float(fitted.values[s_idx, r, lat_idx, lon_idx]):.3f}" for r in range(len(RETURN_PERIODS))]
            pairs = ", ".join(f"{rp}yr={v}" for rp, v in zip(RETURN_PERIODS, vals))
            logger.info(f"  {spi}: {pairs}")
    except Exception as e:
        logger.warning(f"  Spot check failed: {e}")

    # Comparison: fitted vs empirical
    logger.info(f"\n--- Fitted vs Empirical Comparison ---")
    for s_idx, spi in enumerate(ds.spi_period.values):
        diff = fitted.values[s_idx] - empirical.values[s_idx]
        mask = ~np.isnan(diff)
        if mask.any():
            logger.info(
                f"  {spi}: mean_diff={float(np.nanmean(diff)):.4f}, "
                f"max_abs_diff={float(np.nanmax(np.abs(diff[mask]))):.4f}"
            )

    # Fit quality: mu should be ~0, sigma should be ~1
    logger.info(f"\n--- Fit Quality (expected: mu≈0, sigma≈1) ---")
    for s_idx, spi in enumerate(ds.spi_period.values):
        mu = ds["fit_mu"].values[s_idx]
        sigma = ds["fit_sigma"].values[s_idx]
        logger.info(
            f"  {spi}: mu median={float(np.nanmedian(mu)):.4f}, "
            f"sigma median={float(np.nanmedian(sigma)):.4f}"
        )

    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 70)


# ─── Plot ───────────────────────────────────────────────────────────────────


def run_plot(args):
    """Generate SPI distribution plot with return period thresholds.

    Produces a histogram + normal PDF overlay with colored severity bands
    and dashed threshold lines, matching the screenshot style:
    - Blue histogram bars for the SPI distribution
    - Red/orange/yellow bars colored by severity category
    - Standard normal PDF overlay (dashed black)
    - Vertical dashed lines for each return period threshold
    - Green line for a specific date if provided
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    from scipy import stats

    logger.info("Generating SPI distribution plot...")

    ds = xr.open_dataset(args.input)

    # Select SPI period
    spi_period = args.spi_period
    if spi_period not in ds.spi_period.values:
        available = list(ds.spi_period.values)
        logger.error(f"SPI period '{spi_period}' not found. Available: {available}")
        return

    # Get pixel time series from local files
    data_dir = Path(args.data_dir)
    pattern = f"{spi_period}_gamma_global_era5_moda_ref1991to2020_*.nc"
    files = sorted(data_dir.glob(pattern))

    if not files:
        logger.error(f"No files found for {spi_period} in {data_dir}")
        return

    datasets = []
    for f in files:
        d = xr.open_dataset(f)
        var_name = [v for v in d.data_vars if v.startswith("SPI")][0]
        datasets.append(d[[var_name]])

    ds_spi = xr.concat(datasets, dim="time").sortby("time")
    for d in datasets:
        d.close()

    var_name = [v for v in ds_spi.data_vars][0]

    # Extract pixel
    pixel = ds_spi[var_name].sel(lat=args.lat, lon=args.lon, method="nearest")
    actual_lat = float(pixel.lat.values)
    actual_lon = float(pixel.lon.values)
    pixel = pixel.load()
    times = pixel.time.values
    vals = pixel.values.astype(np.float64)
    valid_mask = ~np.isnan(vals)
    valid_vals = vals[valid_mask]

    logger.info(f"  Location: {actual_lat:.2f}°N, {actual_lon:.2f}°E")
    logger.info(f"  {len(valid_vals)} valid months out of {len(vals)}")

    # Get thresholds (manual index lookup — lat may be descending)
    lat_idx = int(np.argmin(np.abs(ds.lat.values - actual_lat)))
    lon_idx = int(np.argmin(np.abs(ds.lon.values - actual_lon)))
    spi_idx = list(ds.spi_period.values).index(spi_period)
    thresholds = {}
    for r_idx, rp in enumerate(RETURN_PERIODS):
        thresholds[rp] = float(ds["fitted_threshold"].values[spi_idx, r_idx, lat_idx, lon_idx])

    # Fit normal
    mu = float(np.mean(valid_vals))
    sigma = float(np.std(valid_vals, ddof=1))

    # ── Create figure ──
    fig, ax = plt.subplots(figsize=(14, 8))

    # Histogram bins
    bin_edges = np.arange(
        max(-4, np.floor(valid_vals.min()) - 0.5),
        min(4, np.ceil(valid_vals.max()) + 0.5) + 0.1,
        0.1,
    )
    counts, edges = np.histogram(valid_vals, bins=bin_edges, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Color bars by severity category
    # Most extreme threshold first
    sorted_rps = sorted(RETURN_PERIODS, reverse=True)
    bar_colors = np.full(len(bin_centers), "#6495ED", dtype=object)  # default: cornflower blue

    for rp in sorted_rps:
        thresh = thresholds[rp]
        mask = bin_centers <= thresh
        bar_colors[mask] = RP_COLORS[rp]

    # Positive tail: green for above +1
    positive_mask = bin_centers >= 1.0
    bar_colors[positive_mask] = "#32CD32"

    ax.bar(bin_centers, counts, width=bin_width, color=bar_colors,
           edgecolor="none", alpha=0.85, zorder=2)

    # Standard normal PDF overlay
    x_pdf = np.linspace(-4, 4, 500)
    y_pdf = stats.norm.pdf(x_pdf, 0, 1)
    ax.plot(x_pdf, y_pdf, "k--", linewidth=1.5, label="Standard Normal", zorder=5)

    # Threshold lines
    for rp in RETURN_PERIODS:
        thresh = thresholds[rp]
        label_text = f"{RP_LABELS[rp]} ({rp}YRP): {thresh:.3f}"
        color = RP_COLORS[rp]
        linestyle = "--"
        ax.axvline(x=thresh, color=color, linestyle=linestyle, linewidth=1.5,
                   label=label_text, zorder=4)
        # Label on plot
        ax.text(thresh, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.4,
                f"{rp}YRP", rotation=90, va="top", ha="right",
                fontsize=9, fontweight="bold", color=color)

    # Highlight a specific date if provided
    if args.highlight_date:
        import pandas as pd
        target_date = pd.Timestamp(args.highlight_date)
        # Find nearest time
        time_idx = np.argmin(np.abs(times.astype("datetime64[D]") -
                                     np.datetime64(target_date, "D")))
        highlight_val = vals[time_idx]
        if not np.isnan(highlight_val):
            date_str = str(times[time_idx])[:7]  # YYYY-MM
            ax.axvline(x=highlight_val, color="#32CD32", linewidth=2.5,
                       label=f"{date_str}: {highlight_val:.3f}", zorder=6)

    # Formatting
    ax.set_xlabel(f"{spi_period} Values", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{spi_period} Distribution with Return Period Thresholds\n"
        f"{actual_lat:.2f}°N, {actual_lon:.2f}°E\n"
        f"(Using Pre-calculated Thresholds)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-4, 4)
    ax.grid(axis="y", alpha=0.3)

    # Save
    output = args.plot_output or f"spi_{spi_period.lower()}_distribution_{actual_lat:.0f}N_{actual_lon:.0f}E.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output}")

    ds.close()
    ds_spi.close()


# ─── Plot map ───────────────────────────────────────────────────────────────


def run_plot_map(args):
    """Generate return period threshold maps for all SPI periods."""
    import matplotlib.pyplot as plt
    import xarray as xr

    logger.info("Generating return period threshold maps...")

    ds = xr.open_dataset(args.input)
    fitted = ds["fitted_threshold"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    for s_idx, spi in enumerate(ds.spi_period.values):
        fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(20, 5))
        fig.suptitle(f"{spi} Drought Thresholds (Fitted Normal)", fontsize=14, y=1.02)

        for r_idx, (ax, rp) in enumerate(zip(axes, RETURN_PERIODS)):
            data = fitted.isel(spi_period=s_idx, return_period=r_idx).values
            im = ax.pcolormesh(
                lon, lat, data,
                cmap="RdYlBu", vmin=-3, vmax=0,
                shading="auto",
            )
            ax.set_title(f"{rp}-yr ({RP_LABELS[rp]})", fontsize=10)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.08)

        out = f"spi_{spi.lower()}_threshold_maps.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {out}")

    ds.close()


# ─── Cartopy map plots from RP Icechunk store ────────────────────────────────


def run_plot_map_cartopy(args):
    """Plot return period threshold maps with cartopy + GeoJSON country overlay.

    Reads the RP Icechunk store from source.coop (anonymous) and produces one
    figure per SPI period (7 figures), each with 5 subplots — one per return
    period.  Country boundaries are drawn from a local GeoJSON file.

    Usage:
        uv run ecmwf_spi_return_periods.py plot-map-cartopy \\
            --geojson ea_ghcf_simple.geojson \\
            --output-dir maps/

        # Use local RP store instead of source.coop
        uv run ecmwf_spi_return_periods.py plot-map-cartopy \\
            --store-path /local/era5_ecmwf_rp_icechunk \\
            --geojson ea_ghcf_simple.geojson
    """
    import json

    import cartopy.crs as ccrs
    import icechunk
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import xarray as xr
    from shapely.geometry import shape

    logger.info("Generating cartopy RP threshold maps...")

    # ── Open RP Icechunk store ──
    store_path = args.store_path
    if store_path is None:
        logger.info(f"  Reading s3://{S3_BUCKET}/{RP_ICECHUNK_S3_PREFIX} (anonymous)")
        storage = icechunk.s3_storage(
            bucket=S3_BUCKET,
            prefix=RP_ICECHUNK_S3_PREFIX,
            region=S3_REGION,
            anonymous=True,
        )
    else:
        logger.info(f"  Reading local store: {store_path}")
        storage = icechunk.local_filesystem_storage(path=store_path)

    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    lat = ds["lat"].values
    lon = ds["lon"].values
    spi_periods = list(ds["spi_period"].values)
    return_periods = list(ds["return_period"].values)

    # ── Load GeoJSON country boundaries ──
    with open(args.geojson) as f:
        gj = json.load(f)
    country_geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    country_codes = [feat["properties"].get("GID_0", "") for feat in gj["features"]]
    logger.info(f"  Countries: {country_codes}")

    proj = ccrs.PlateCarree()
    extent = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── One figure per SPI period, 5 subplots (one per RP) ──
    for s_idx, spi in enumerate(spi_periods):
        fig, axes = plt.subplots(
            1, len(return_periods),
            figsize=(22, 5),
            subplot_kw={"projection": proj},
        )
        fig.suptitle(
            f"{spi} — Drought Return Period Thresholds (Fitted Normal)\n"
            f"East Africa | ECMWF ERA5-Drought | ref 1991–2020",
            fontsize=13, y=1.02,
        )

        # Shared colormap: all fitted thresholds are ≤ 0
        # RdYlBu_r: blue=near 0 (milder) → red=very negative (more severe)
        cmap = "RdYlBu_r"
        vmin, vmax = -3.0, 0.0

        ims = []
        for r_idx, (ax, rp) in enumerate(zip(axes, return_periods)):
            data = ds["fitted_threshold"].isel(
                spi_period=s_idx, return_period=r_idx,
            ).values

            im = ax.pcolormesh(
                lon, lat, data,
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=proj, shading="auto",
            )
            ims.append(im)

            # Country boundaries from GeoJSON
            ax.add_geometries(
                country_geoms, proj,
                facecolor="none", edgecolor="black", linewidth=0.6,
            )

            # Gridlines
            gl = ax.gridlines(
                draw_labels=(r_idx == 0),
                linewidth=0.3, color="gray", alpha=0.5,
                xlocs=mticker.MultipleLocator(10),
                ylocs=mticker.MultipleLocator(10),
            )
            gl.top_labels = False
            gl.right_labels = False
            if r_idx == 0:
                gl.left_labels = True
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}

            ax.set_extent(extent, crs=proj)
            ax.set_title(
                f"{int(rp)}-yr  ({RP_LABELS[int(rp)]})\n"
                f"std={float(ds['standard_threshold'].values[r_idx]):.2f}",
                fontsize=9,
            )

            # Per-subplot colorbar at bottom
            cbar = fig.colorbar(
                im, ax=ax, orientation="horizontal",
                pad=0.04, shrink=0.9, aspect=20,
            )
            cbar.set_label("SPI threshold", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        out_path = output_dir / f"era5_spi_{spi.lower()}_rp_thresholds_ea.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {out_path}")

    ds.close()
    logger.info(f"Done — {len(spi_periods)} figures saved to {output_dir}/")


# ─── Compute from pencil zarr store ─────────────────────────────────────────


def compute_from_store(args):
    """Compute return period thresholds by reading from the pencil zarr on source.coop.

    Reads SPI time series from the pencil-chunked zarr (full-time × 5×5 lat/lon),
    runs per-pixel normal fit + empirical percentiles, and writes the output as a
    pan-chunk Icechunk store to source.coop S3.

    Pan chunk layout for the RP store: (1, n_rp, n_lat, n_lon) — one SPI-period
    slab at full spatial extent, matching a flat per-layer access pattern.

    Usage:
        # Read from source.coop, write RP icechunk to source.coop
        uv run ecmwf_spi_return_periods.py compute-store

        # Override source/target paths
        uv run ecmwf_spi_return_periods.py compute-store \\
            --source-path /local/era5_ecmwf_pencil \\
            --store-path /local/era5_ecmwf_rp_icechunk
    """
    import icechunk
    import xarray as xr
    from scipy import stats

    logger.info("=" * 70)
    logger.info("COMPUTE-STORE: SPI Return Periods from pencil zarr → Icechunk")
    logger.info("=" * 70)
    overall_start = time.time()

    # ── Open source pencil zarr ──
    source_path = args.source_path
    if source_path is None:
        # Default: read anonymously from source.coop
        import s3fs
        logger.info(f"  Source: s3://{S3_BUCKET}/{PENCIL_S3_PREFIX} (anonymous)")
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": S3_REGION})
        store_map = s3fs.S3Map(root=f"{S3_BUCKET}/{PENCIL_S3_PREFIX}", s3=fs)
        ds = xr.open_zarr(store_map, consolidated=True)
    else:
        logger.info(f"  Source: {source_path} (local)")
        ds = xr.open_zarr(source_path, consolidated=True)

    lat = ds["lat"].values
    lon = ds["lon"].values
    n_lat, n_lon = len(lat), len(lon)
    spi_vars = [v for v in SPI_PERIODS if v in ds.data_vars]
    n_spi = len(spi_vars)
    n_rp = len(RETURN_PERIODS)
    logger.info(f"  Grid: {n_lat} lat × {n_lon} lon")
    logger.info(f"  SPI variables found: {spi_vars}")
    logger.info(f"  Return periods: {RETURN_PERIODS}")

    # Standard normal thresholds (pixel-independent)
    standard_thresholds = np.array(
        [stats.norm.ppf(1.0 / T) for T in RETURN_PERIODS], dtype=np.float32,
    )
    logger.info(f"  Standard thresholds: {dict(zip(RETURN_PERIODS, standard_thresholds.round(3)))}")

    # Output arrays
    fitted_thresholds   = np.full((n_spi, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
    empirical_thresholds= np.full((n_spi, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
    fit_mu              = np.full((n_spi, n_lat, n_lon), np.nan, dtype=np.float32)
    fit_sigma           = np.full((n_spi, n_lat, n_lon), np.nan, dtype=np.float32)
    n_valid_months      = np.full((n_spi, n_lat, n_lon), 0, dtype=np.int32)

    for s_idx, spi in enumerate(spi_vars):
        logger.info(f"\n--- Processing {spi} ---")
        t0 = time.time()

        # Load full time series for this SPI (pencil chunks = fast per-pixel access)
        data = ds[spi].values.astype(np.float32)  # (n_time, n_lat, n_lon)
        logger.info(f"  Loaded: shape {data.shape}")

        for i in range(n_lat):
            for j in range(n_lon):
                ts = data[:, i, j]
                valid = ts[~np.isnan(ts)]
                n_valid = len(valid)
                n_valid_months[s_idx, i, j] = n_valid

                if n_valid < 30:
                    continue

                mu = float(np.mean(valid))
                sigma = float(np.std(valid, ddof=1))
                fit_mu[s_idx, i, j] = mu
                fit_sigma[s_idx, i, j] = sigma

                if sigma < 1e-6:
                    continue

                for r_idx, T in enumerate(RETURN_PERIODS):
                    z = stats.norm.ppf(1.0 / T)
                    fitted_thresholds[s_idx, r_idx, i, j] = mu + sigma * z
                    empirical_thresholds[s_idx, r_idx, i, j] = np.percentile(valid, 100.0 / T)

        elapsed = time.time() - t0
        ocean_frac = (n_valid_months[s_idx] < 30).mean()
        logger.info(f"  Done in {elapsed:.1f}s — ocean/missing: {ocean_frac*100:.1f}%")
        del data

    ds.close()

    # ── Build output xarray Dataset ──
    logger.info("\nBuilding output Dataset...")
    ds_out = xr.Dataset(
        {
            "fitted_threshold": (
                ["spi_period", "return_period", "lat", "lon"],
                fitted_thresholds,
                {
                    "long_name": "SPI drought threshold (fitted normal)",
                    "units": "dimensionless",
                    "description": "mu + sigma * ppf(1/T) from fitted N(mu, sigma) per pixel",
                },
            ),
            "empirical_threshold": (
                ["spi_period", "return_period", "lat", "lon"],
                empirical_thresholds,
                {
                    "long_name": "SPI drought threshold (empirical percentile)",
                    "units": "dimensionless",
                    "description": "SPI value at (1/T)×100 percentile of observed series",
                },
            ),
            "standard_threshold": (
                ["return_period"],
                standard_thresholds,
                {
                    "long_name": "SPI drought threshold (standard normal)",
                    "units": "dimensionless",
                    "description": "ppf(1/T) from N(0,1) — same for all pixels by SPI definition",
                },
            ),
            "fit_mu": (
                ["spi_period", "lat", "lon"],
                fit_mu,
                {"long_name": "Fitted normal mean (mu)", "units": "dimensionless"},
            ),
            "fit_sigma": (
                ["spi_period", "lat", "lon"],
                fit_sigma,
                {"long_name": "Fitted normal std dev (sigma)", "units": "dimensionless"},
            ),
            "n_valid_months": (
                ["spi_period", "lat", "lon"],
                n_valid_months,
                {"long_name": "Number of valid (non-NaN) monthly values"},
            ),
        },
        coords={
            "spi_period": ("spi_period", spi_vars, {"long_name": "SPI accumulation period"}),
            "return_period": ("return_period", RETURN_PERIODS, {"units": "years"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF ERA5-Drought SPI — Drought Return Period Thresholds",
            "source": f"s3://{S3_BUCKET}/{PENCIL_S3_PREFIX}",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": f"Created {datetime.now().isoformat()}",
            "distribution": "Normal (standard and fitted) + empirical percentile",
            "methodology": (
                "Fitted N(mu,sigma) per pixel + ppf(1/T) + empirical percentile. "
                "Drought = left tail (negative SPI). Pixels with <30 valid months set to NaN."
            ),
            "return_period_labels": str(RP_LABELS),
            "Conventions": "CF-1.8",
        },
    )

    # ── Set up Icechunk store ──
    # Pan chunk: one SPI-period slab at full spatial extent — (1, n_rp, n_lat, n_lon)
    pan_chunk = (1, n_rp, n_lat, n_lon)
    slab_chunk = (1, n_lat, n_lon)   # for 3-D variables (fit_mu, fit_sigma, n_valid)

    store_path = args.store_path
    if store_path is None:
        # Write to source.coop S3 using AWS_* env vars
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("SOURCE_COOP_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SOURCE_COOP_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN") or os.environ.get("SOURCE_COOP_SESSION_TOKEN")

        if not access_key or not secret_key:
            raise RuntimeError(
                "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN) "
                "in .env or environment for source.coop write access."
            )

        logger.info(f"  Target: s3://{S3_BUCKET}/{RP_ICECHUNK_S3_PREFIX} (Icechunk)")
        storage = icechunk.s3_storage(
            bucket=S3_BUCKET,
            prefix=RP_ICECHUNK_S3_PREFIX,
            region=S3_REGION,
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=session_token,
        )
    else:
        logger.info(f"  Target: {store_path} (local Icechunk)")
        storage = icechunk.local_filesystem_storage(path=store_path)

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new Icechunk repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing Icechunk repository (overwriting)")

    session = repo.writable_session("main")
    ds_out.to_zarr(
        session.store,
        mode="w",
        consolidated=False,
        encoding={
            "fitted_threshold":    {"chunks": pan_chunk},
            "empirical_threshold": {"chunks": pan_chunk},
            "fit_mu":              {"chunks": slab_chunk},
            "fit_sigma":           {"chunks": slab_chunk},
            "n_valid_months":      {"chunks": slab_chunk},
        },
    )
    session.commit("ECMWF SPI return period thresholds — all SPI periods")

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("COMPUTE-STORE COMPLETE")
    logger.info(f"  SPI periods: {spi_vars}")
    logger.info(f"  Return periods: {RETURN_PERIODS}")
    logger.info(f"  Pan chunk: {pan_chunk}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 70)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ECMWF SPI — Drought Return Period Analysis",
    )
    sub = parser.add_subparsers(dest="command")

    # ── compute ──
    p_compute = sub.add_parser("compute", help="Compute return period thresholds")
    p_compute.add_argument("--data-dir", type=str, default="data/",
                           help="Directory containing SPI NetCDF files")
    p_compute.add_argument("--output", type=str,
                           default="ecmwf_spi_return_periods.nc",
                           help="Output NetCDF path")

    # ── verify ──
    p_verify = sub.add_parser("verify", help="Validate output NetCDF")
    p_verify.add_argument("--input", type=str,
                          default="ecmwf_spi_return_periods.nc")

    # ── plot ──
    p_plot = sub.add_parser("plot",
                            help="Plot SPI distribution with thresholds")
    p_plot.add_argument("--input", type=str,
                        default="ecmwf_spi_return_periods.nc")
    p_plot.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing SPI NetCDF files (for time series)")
    p_plot.add_argument("--spi-period", type=str, default="SPI3",
                        help="SPI period to plot (e.g. SPI3, SPI12)")
    p_plot.add_argument("--lat", type=float, default=9.0,
                        help="Latitude of pixel to plot")
    p_plot.add_argument("--lon", type=float, default=40.0,
                        help="Longitude of pixel to plot")
    p_plot.add_argument("--highlight-date", type=str, default=None,
                        help="Date to highlight (YYYY-MM-DD)")
    p_plot.add_argument("--plot-output", type=str, default=None,
                        help="Output PNG path")

    # ── plot-map ──
    p_map = sub.add_parser("plot-map", help="Plot threshold maps")
    p_map.add_argument("--input", type=str,
                       default="ecmwf_spi_return_periods.nc")

    # ── plot-map-cartopy ──
    p_cartopy = sub.add_parser(
        "plot-map-cartopy",
        help=(
            "Plot RP threshold maps using cartopy + GeoJSON country overlay. "
            "Reads RP Icechunk store from source.coop anonymously by default."
        ),
    )
    p_cartopy.add_argument(
        "--store-path", type=str, default=None,
        help="Local Icechunk RP store path. Default: reads from source.coop (anonymous).",
    )
    p_cartopy.add_argument(
        "--geojson", type=str, default="ea_ghcf_simple.geojson",
        help="Path to GeoJSON file with country boundaries.",
    )
    p_cartopy.add_argument(
        "--output-dir", type=str, default="maps",
        help="Directory for output PNG files (default: maps/).",
    )

    # ── compute-store ──
    p_cs = sub.add_parser(
        "compute-store",
        help=(
            "Read pencil zarr from source.coop (or local), compute return period "
            "thresholds per pixel, write as pan-chunk Icechunk store to source.coop. "
            "Reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN from .env."
        ),
    )
    p_cs.add_argument(
        "--source-path", type=str, default=None,
        help=(
            "Path to source pencil zarr. "
            "Default: s3://us-west-2.opendata.source.coop/e4drr-project/observations/era5_ecmwf_pencil"
        ),
    )
    p_cs.add_argument(
        "--store-path", type=str, default=None,
        help=(
            "Local path for output Icechunk store. "
            "Default: writes directly to source.coop S3 at "
            "e4drr-project/observations/era5_ecmwf_rp_icechunk"
        ),
    )

    args = parser.parse_args()

    if args.command == "compute":
        compute_return_periods(args)
    elif args.command == "verify":
        run_verify(args)
    elif args.command == "plot":
        run_plot(args)
    elif args.command == "plot-map":
        run_plot_map(args)
    elif args.command == "plot-map-cartopy":
        run_plot_map_cartopy(args)
    elif args.command == "compute-store":
        compute_from_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
