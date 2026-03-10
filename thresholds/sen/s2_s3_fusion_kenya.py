"""
Sentinel-2 / Sentinel-3 Fusion for Vegetation Monitoring — Kenya
================================================================
Adapted from the EOPF-101 S2/S3 fusion tutorial for a Kenya case region.

Workflow:
  1. Search S2 L2A and S3 OLCI L2 LFR via EOPF STAC API
  2. Load S2 L2A cloud-optimized Zarr data (public HTTPS)
  3. Cloud-mask using Scene Classification Layer (SCL)
  4. Compute NDVI (B8A − B04) / (B8A + B04) at 20 m
  5. Build a multi-date NDVI composite for a Kenya AOI
  6. Compute fPAR, LAI, fCover via ESA SNAP BiophysicalOp
  7. Search S3 OLCI for daily temporal coverage
  8. Visualize results

Reference: https://eopf-toolkit.github.io/eopf-101/06_eopf_zarr_in_action/610_s2_s3_fusion.html
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pystac_client import Client

# ── Configuration ────────────────────────────────────────────────────────────

# Kenya — central highlands (covers parts of Nairobi, Mt Kenya foothills)
SEARCH_BBOX = (36.5, -1.5, 37.5, -0.5)   # (lon_min, lat_min, lon_max, lat_max)
DATE_START = "2024-11-01"
DATE_END = "2024-12-31"

# S2 tile covering the AOI
S2_TILE = "T36MZE"

# Bands for NDVI
S2_NDVI_BANDS = ["b04", "b8a"]          # Red, NIR
S2_RGB_BANDS = ["b04", "b03", "b02"]    # for true-colour preview
RESOLUTION = 20                          # metres — use 20m (b8a native res)

# Subset AOI in UTM (easting, northing) — within tile T36MZE
# Tile covers x: 800000–910000, y: 9890000–10000000
AOI_X = (840_000, 870_000)   # easting range
AOI_Y = (9_930_000, 9_960_000)  # northing range (south of equator)

# S3 OLCI bands for NDVI proxy
S3_BANDS = ["rc681", "rc865"]

# Output directory
OUTPUT_DIR = Path("./output_kenya_fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_ENDPOINT = "https://stac.core.eopf.eodc.eu/"

# ── SCL cloud mask ───────────────────────────────────────────────────────────

# SCL values: 0=no-data, 1=saturated, 2=dark, 3=shadow, 4=vegetation,
# 5=bare-soil, 6=water, 7=unclassified, 8=cloud-med, 9=cloud-high,
# 10=cirrus, 11=snow/ice
VALID_SCL = {2, 4, 5, 6, 11}  # dark, vegetation, bare-soil, water, snow


def validate_scl(scl):
    """Create boolean mask: True where pixel is valid (not cloud/shadow)."""
    mask = xr.zeros_like(scl, dtype=bool)
    for val in VALID_SCL:
        mask = mask | (scl == val)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Connect to EOPF STAC and search for Sentinel-2 L2A
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Searching EOPF STAC for Sentinel-2 L2A over Kenya")
print("=" * 70)

catalog = Client.open(url=STAC_ENDPOINT)

s2_items = list(
    catalog.search(
        bbox=SEARCH_BBOX,
        datetime=f"{DATE_START}T00:00:00Z/{DATE_END}T23:59:59Z",
        collections="sentinel-2-l2a",
    ).item_collection()
)

# Filter by tile
s2_urls = [
    item.assets["product"].href
    for item in s2_items
    if f"_{S2_TILE}_" in item.assets["product"].href
]

print(f"  Total S2 L2A items found : {len(s2_items)}")
print(f"  Items for tile {S2_TILE}  : {len(s2_urls)}")
for url in s2_urls[:5]:
    print(f"    {url.split('/')[-1]}")
if len(s2_urls) > 5:
    print(f"    ... and {len(s2_urls) - 5} more")

if not s2_urls:
    print("ERROR: No S2 L2A scenes found. Try adjusting SEARCH_BBOX / dates / tile.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load & explore a single S2 scene (Zarr)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2: Loading a single S2 L2A scene from Zarr")
print("=" * 70)

s2_url = s2_urls[-1]  # most recent
print(f"  Opening: {s2_url.split('/')[-1]}")

s2_zarr = xr.open_datatree(s2_url, engine="zarr", chunks={}, decode_timedelta=False)

# Reflectance bands at chosen resolution
zarr_meas = s2_zarr["measurements"]["reflectance"][f"r{RESOLUTION}m"]
print(f"  Available bands at {RESOLUTION}m: {list(zarr_meas.data_vars)}")
print(f"  Grid size: {dict(zarr_meas.sizes)}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Cloud masking with SCL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 3: Applying cloud mask (SCL)")
print("=" * 70)

scl = s2_zarr["conditions"]["mask"]["l2a_classification"][f"r{RESOLUTION}m"]["scl"]
valid_mask = validate_scl(scl)

cloud_pct = float((~valid_mask).sum() / valid_mask.size * 100)
print(f"  Cloud/invalid pixels: {cloud_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Compute NDVI for a single scene
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 4: Computing NDVI for single scene")
print("=" * 70)

b04 = zarr_meas["b04"]  # Red
b8a = zarr_meas["b8a"]  # NIR

denom = b8a + b04
ndvi = xr.where(denom != 0, (b8a - b04) / denom, np.nan)
ndvi = xr.where(valid_mask, ndvi, np.nan)
ndvi = ndvi.clip(-1, 1)

# Subset to AOI
ndvi_aoi = ndvi.sel(x=slice(*AOI_X), y=slice(AOI_Y[1], AOI_Y[0]))

print(f"  NDVI AOI shape: {ndvi_aoi.shape}")
print(f"  Computing (lazy → in-memory) ...")
ndvi_aoi_data = ndvi_aoi.compute()
valid_vals = ndvi_aoi_data.values[~np.isnan(ndvi_aoi_data.values)]
if valid_vals.size > 0:
    print(f"  NDVI range: {valid_vals.min():.3f} — {valid_vals.max():.3f}")
    print(f"  NDVI mean : {valid_vals.mean():.3f}")
else:
    print("  WARNING: no valid NDVI pixels in AOI (all cloudy?)")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Multi-date NDVI composite (cloud-free median)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 5: Building multi-date NDVI composite")
print("=" * 70)

ndvi_stack = []
dates = []

for i, url in enumerate(s2_urls):
    scene_name = url.split("/")[-1].replace(".zarr", "")
    # Extract date from product name (e.g. S2A_MSIL2A_20241228T...)
    parts = scene_name.split("_")
    date_str = [p for p in parts if len(p) == 15 and p.startswith("20")][0][:8]
    acq_date = datetime.strptime(date_str, "%Y%m%d")

    print(f"  [{i+1}/{len(s2_urls)}] {scene_name[:60]}  ({acq_date.date()})")

    try:
        dt = xr.open_datatree(url, engine="zarr", chunks={}, decode_timedelta=False)
        meas = dt["measurements"]["reflectance"][f"r{RESOLUTION}m"]
        scl_i = dt["conditions"]["mask"]["l2a_classification"][f"r{RESOLUTION}m"]["scl"]

        red = meas["b04"].sel(x=slice(*AOI_X), y=slice(AOI_Y[1], AOI_Y[0]))
        nir = meas["b8a"].sel(x=slice(*AOI_X), y=slice(AOI_Y[1], AOI_Y[0]))
        scl_sub = scl_i.sel(x=slice(*AOI_X), y=slice(AOI_Y[1], AOI_Y[0]))

        mask_i = validate_scl(scl_sub)
        denom_i = nir + red
        ndvi_i = xr.where((mask_i) & (denom_i != 0), (nir - red) / denom_i, np.nan)
        ndvi_i = ndvi_i.clip(-1, 1)
        ndvi_i = ndvi_i.compute()
        ndvi_stack.append(ndvi_i.values)
        dates.append(acq_date)
    except Exception as e:
        print(f"    SKIPPED: {e}")

if ndvi_stack:
    ndvi_cube = np.stack(ndvi_stack, axis=0)  # (time, y, x)
    ndvi_median = np.nanmedian(ndvi_cube, axis=0)
    ndvi_mean = np.nanmean(ndvi_cube, axis=0)

    valid_med = ndvi_median[~np.isnan(ndvi_median)]
    print(f"\n  Composite ({len(dates)} dates)")
    print(f"  Median NDVI range: {valid_med.min():.3f} — {valid_med.max():.3f}")
    print(f"  Median NDVI mean : {valid_med.mean():.3f}")
else:
    print("  No scenes loaded — cannot build composite.")
    ndvi_median = None
    ndvi_mean = None

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Compute fPAR, LAI, fCover via ESA SNAP BiophysicalOp
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6: Computing fPAR, LAI, fCover with ESA SNAP BiophysicalOp")
print("=" * 70)

import esa_snappy
from esa_snappy import Product, ProductData, GPF as SNAP_GPF, HashMap as SnapHashMap

# Mapping from EOPF Zarr band names → SNAP BiophysicalOp expected names
# BiophysicalOp needs: B3, B4, B5, B6, B7, B8A, B11, B12
# Plus view/sun angles: view_zenith_mean, view_azimuth_mean, sun_zenith, sun_azimuth
ZARR_TO_SNAP = {
    "b03": "B3", "b04": "B4", "b05": "B5", "b06": "B6",
    "b07": "B7", "b8a": "B8A", "b11": "B11", "b12": "B12",
}
SNAP_GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()


def zarr_scene_to_snap_product(zarr_url, aoi_x, aoi_y, resolution=20):
    """Load S2 bands from Zarr and construct a SNAP Product for BiophysicalOp."""
    dt = xr.open_datatree(zarr_url, engine="zarr", chunks={}, decode_timedelta=False)
    meas = dt["measurements"]["reflectance"][f"r{resolution}m"]

    # Subset to AOI
    subset = meas.sel(x=slice(*aoi_x), y=slice(aoi_y[1], aoi_y[0]))
    first_var = list(subset.data_vars)[0]
    h, w = subset[first_var].shape

    # Detect sensor from product name
    scene_name = zarr_url.split("/")[-1]
    sensor = "S2A" if "S2A" in scene_name else "S2B"

    product = Product(scene_name, "S2_MSI_Level-2A", w, h)

    # Load reflectance bands
    for zarr_name, snap_name in ZARR_TO_SNAP.items():
        if zarr_name not in subset.data_vars:
            continue
        data = subset[zarr_name].compute().values.astype(np.float32)
        band = product.addBand(snap_name, ProductData.TYPE_FLOAT32)
        band.setNoDataValue(0.0)
        band.setNoDataValueUsed(True)
        band.setData(ProductData.createInstance(data.flatten()))

    # Sun/view angles — use scene-level means from geometry group
    geom = dt["conditions"]["geometry"]
    sun_angles = geom["mean_sun_angles"].compute().values  # [zenith, azimuth]
    view_angles = geom["mean_viewing_incidence_angles"].compute().values  # (band, angle)
    # Average over bands for view angles
    view_zen = float(np.nanmean(view_angles[:, 0]))
    view_azi = float(np.nanmean(view_angles[:, 1]))
    sun_zen = float(sun_angles[0])
    sun_azi = float(sun_angles[1])

    for aname, val in [("view_zenith_mean", view_zen),
                       ("view_azimuth_mean", view_azi),
                       ("sun_zenith", sun_zen),
                       ("sun_azimuth", sun_azi)]:
        band = product.addBand(aname, ProductData.TYPE_FLOAT32)
        arr = np.full((h, w), val, dtype=np.float32)
        band.setData(ProductData.createInstance(arr.flatten()))

    # Cloud mask from SCL
    scl_data = dt["conditions"]["mask"]["l2a_classification"][f"r{resolution}m"]["scl"]
    scl_sub = scl_data.sel(x=slice(*aoi_x), y=slice(aoi_y[1], aoi_y[0])).compute()

    return product, sensor, scl_sub, w, h


def run_biophysical(snap_product, sensor="S2A"):
    """Run SNAP BiophysicalOp and return fPAR, LAI, fCover as numpy arrays."""
    params = SnapHashMap()
    params.put("computeLAI", True)
    params.put("computeFapar", True)
    params.put("computeFcover", True)
    params.put("computeCab", True)
    params.put("computeCw", True)
    params.put("sensor", sensor)
    params.put("resolution", "10")

    result = SNAP_GPF.createProduct("BiophysicalOp", params, snap_product)

    w = result.getSceneRasterWidth()
    h = result.getSceneRasterHeight()
    outputs = {}
    for bname in result.getBandNames():
        band = result.getBand(bname)
        buf = np.zeros((h, w), dtype=np.float32)
        band.readPixels(0, 0, w, h, buf)
        outputs[bname] = buf

    result.dispose()
    return outputs


# Process the most recent scene
print(f"  Processing: {s2_urls[0].split('/')[-1]}")
snap_prod, sensor, scl_aoi, snap_w, snap_h = zarr_scene_to_snap_product(
    s2_urls[0], AOI_X, AOI_Y, RESOLUTION
)
print(f"  Sensor: {sensor}, AOI size: {snap_w} x {snap_h}")
print(f"  Running BiophysicalOp ...")

bio_results = run_biophysical(snap_prod, sensor)
snap_prod.dispose()

# Apply cloud mask to biophysical outputs
scl_mask = validate_scl(scl_aoi).values

print(f"  Biophysical bands: {list(bio_results.keys())}")
for bname in ["fapar", "lai", "fcover", "lai_cab", "lai_cw"]:
    if bname in bio_results:
        arr = bio_results[bname]
        arr_masked = np.where(scl_mask, arr, np.nan)
        bio_results[bname + "_masked"] = arr_masked
        valid = arr_masked[~np.isnan(arr_masked)]
        valid = valid[(valid > -1) & (valid < 100)]  # filter outliers
        if valid.size > 0:
            print(f"    {bname:10s}: min={valid.min():.4f}  max={valid.max():.4f}  mean={valid.mean():.4f}")
        else:
            print(f"    {bname:10s}: (no valid pixels)")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Search Sentinel-3 OLCI L2 LFR (daily temporal coverage)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6: Searching S3 OLCI L2 LFR for temporal context")
print("=" * 70)

# S3 has wider swath — search broader bbox
s3_bbox = (33.0, -3.0, 40.0, 2.0)

s3_items = list(
    catalog.search(
        bbox=s3_bbox,
        datetime=f"{DATE_START}T00:00:00Z/{DATE_END}T23:59:59Z",
        collections="sentinel-3-olci-l2-lfr",
    ).item_collection()
)

print(f"  S3 OLCI L2 LFR items found: {len(s3_items)}")

if s3_items:
    for item in s3_items[:5]:
        print(f"    {item.id}")
        print(f"      bbox: {item.bbox}")
        # S3 assets include gifapar (instantaneous FAPAR)
        asset_names = list(item.assets.keys())
        print(f"      assets: {asset_names}")
    if len(s3_items) > 5:
        print(f"    ... and {len(s3_items) - 5} more")

    print("\n  NOTE: S3 OLCI Zarr data uses s3:// protocol and requires")
    print("  EODC object storage credentials. To access S3 data:")
    print("    export AWS_ACCESS_KEY_ID=<your_key>")
    print("    export AWS_SECRET_ACCESS_KEY=<your_secret>")
    print("    export AWS_ENDPOINT_URL=https://objects.eodc.eu")
    print("  Then open with: xr.open_datatree(url, engine='zarr', ...)")
    print("\n  Key S3 OLCI assets for fusion:")
    print("    - rc681  : Reflectance at 681 nm (Red)")
    print("    - rc865  : Reflectance at 865 nm (NIR)")
    print("    - gifapar: Global Instantaneous FAPAR")
    print("    - otci   : OLCI Terrestrial Chlorophyll Index")
else:
    print("  No S3 OLCI products found in the EOPF archive for this period.")
    print("  The EOPF sample archive may have limited S3 temporal coverage.")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Visualization
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 8: Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) RGB preview of last scene
print("  Loading RGB bands for preview ...")
try:
    rgb_data = zarr_meas.to_dataset()[S2_RGB_BANDS].sel(
        x=slice(*AOI_X), y=slice(AOI_Y[1], AOI_Y[0])
    )
    rgb_arr = np.stack(
        [rgb_data[b].compute().values for b in S2_RGB_BANDS], axis=-1
    )
    # Normalize for display (reflectance values are typically 0–0.3)
    rgb_disp = np.clip(rgb_arr / 0.3, 0, 1)

    axes[0].imshow(rgb_disp, extent=[AOI_X[0], AOI_X[1], AOI_Y[0], AOI_Y[1]])
    axes[0].set_title(f"True Colour (last scene)\n{dates[-1].date() if dates else ''}")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
except Exception as e:
    axes[0].text(0.5, 0.5, f"RGB failed:\n{e}", ha="center", va="center",
                 transform=axes[0].transAxes, fontsize=9)
    axes[0].set_title("True Colour")

# (b) Single-date NDVI
im1 = axes[1].imshow(
    ndvi_aoi_data.values,
    cmap="YlGn", vmin=-0.2, vmax=0.9,
    extent=[AOI_X[0], AOI_X[1], AOI_Y[0], AOI_Y[1]],
)
axes[1].set_title(f"NDVI (single scene)\n{dates[-1].date() if dates else ''}")
axes[1].set_xlabel("Easting (m)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# (c) Multi-date median NDVI
if ndvi_median is not None:
    im2 = axes[2].imshow(
        ndvi_median,
        cmap="YlGn", vmin=-0.2, vmax=0.9,
        extent=[AOI_X[0], AOI_X[1], AOI_Y[0], AOI_Y[1]],
    )
    axes[2].set_title(f"Median NDVI composite\n{len(dates)} scenes, {DATE_START}–{DATE_END}")
    axes[2].set_xlabel("Easting (m)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
else:
    axes[2].text(0.5, 0.5, "No composite", ha="center", va="center",
                 transform=axes[2].transAxes)
    axes[2].set_title("Median NDVI composite")

plt.suptitle("Sentinel-2 Vegetation Monitoring — Kenya", fontsize=14, y=1.02)
plt.tight_layout()

fig_path = OUTPUT_DIR / "kenya_ndvi_composite.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved: {fig_path}")
plt.show()

# ── fPAR, LAI, fCover maps from BiophysicalOp ────────────────────────────────

bio_plot_bands = {
    "fapar_masked": ("fPAR (SNAP BiophysicalOp)", "YlGn", 0, 1),
    "lai_masked": ("LAI (SNAP BiophysicalOp)", "Greens", 0, 6),
    "fcover_masked": ("fCover (SNAP BiophysicalOp)", "YlGn", 0, 1),
}
available_bio = {k: v for k, v in bio_plot_bands.items() if k in bio_results}

if available_bio:
    fig_bio, axes_bio = plt.subplots(1, len(available_bio), figsize=(6 * len(available_bio), 5))
    if len(available_bio) == 1:
        axes_bio = [axes_bio]
    for ax, (bname, (title, cmap, vmin, vmax)) in zip(axes_bio, available_bio.items()):
        im = ax.imshow(
            bio_results[bname], cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[AOI_X[0], AOI_X[1], AOI_Y[0], AOI_Y[1]],
        )
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("ESA SNAP Biophysical Variables — Kenya", fontsize=14, y=1.02)
    plt.tight_layout()
    bio_fig_path = OUTPUT_DIR / "kenya_biophysical_fapar_lai.png"
    fig_bio.savefig(bio_fig_path, dpi=150, bbox_inches="tight")
    print(f"  Biophysical figure saved: {bio_fig_path}")
    plt.show()

# ── NDVI time series plot ────────────────────────────────────────────────────

if ndvi_stack:
    mean_ndvi_ts = [np.nanmean(arr) for arr in ndvi_stack]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(dates, mean_ndvi_ts, "o-", color="green", linewidth=2, markersize=6)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Mean NDVI")
    ax2.set_title(f"NDVI Time Series — Kenya AOI ({S2_TILE})")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.0)
    fig2.autofmt_xdate()
    plt.tight_layout()

    ts_path = OUTPUT_DIR / "kenya_ndvi_timeseries.png"
    fig2.savefig(ts_path, dpi=150, bbox_inches="tight")
    print(f"  Time series saved: {ts_path}")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — Summary & next steps
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Region     : Kenya ({SEARCH_BBOX})")
print(f"  Period     : {DATE_START} to {DATE_END}")
print(f"  S2 tile    : {S2_TILE}")
print(f"  S2 scenes  : {len(s2_urls)} found, {len(dates)} processed")
print(f"  S3 scenes  : {len(s3_items)} found (access requires credentials)")
print(f"  Output     : {OUTPUT_DIR.resolve()}")
print()
print("NEXT STEPS for full S2/S3 fusion (requires EFAST package + S3 credentials):")
print("  1. Configure EODC S3 credentials for S3 OLCI Zarr access")
print("  2. Install efast package (not yet on PyPI — check EOPF GitLab)")
print("  3. Reproject S3 OLCI bands (rc681, rc865) to S2 grid")
print("  4. Run EFAST fusion to interpolate S2 NDVI at daily S3 cadence")
print("  5. Extract phenological metrics from fused time series")
print()
print("Done.")
