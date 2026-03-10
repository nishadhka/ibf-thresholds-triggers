"""
Sentinel-2 fPAR & LAI Calculation using ESA SNAP BiophysicalOp
==============================================================
Computes biophysical variables (fPAR, LAI, fCover, Cab, Cw) from a single
Sentinel-2 L2A image using ESA SNAP's BiophysicalOp operator via esa_snappy.

Workflow:
  1. Download a sample Sentinel-2 L2A product (Copernicus Browser)
  2. Read the product with esa_snappy
  3. Resample bands to a common resolution (required by BiophysicalOp)
  4. Run BiophysicalOp to compute fPAR, LAI, etc.
  5. Export and visualize results
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 1. Initialize esa_snappy
# ---------------------------------------------------------------------------
import esa_snappy
from esa_snappy import ProductIO, GPF, HashMap, Product

# Ensure operator SPIs are loaded
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

# ---------------------------------------------------------------------------
# 2. Set input path — change this to your Sentinel-2 L2A product
# ---------------------------------------------------------------------------
# Accepts: .SAFE directory, .zip, or .dim (BEAM-DIMAP) file
INPUT_PATH = os.environ.get(
    "S2_INPUT",
    "/home/mambauser/nb-chl/S2_input/MTD_MSIL2A.xml",  # adjust as needed
)

if not os.path.exists(INPUT_PATH):
    print(f"ERROR: Input product not found at: {INPUT_PATH}")
    print("Set S2_INPUT env var or edit INPUT_PATH in this script.")
    print(
        "\nTo download a sample product, use the Copernicus Data Space Browser:\n"
        "  https://browser.dataspace.copernicus.eu/\n"
        "Search for 'S2A_MSIL2A' or 'S2B_MSIL2A', download, and extract the .SAFE folder."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Read the Sentinel-2 product
# ---------------------------------------------------------------------------
print(f"Reading product: {INPUT_PATH}")
product = ProductIO.readProduct(INPUT_PATH)

print(f"Product name : {product.getName()}")
print(f"Product type : {product.getProductType()}")
print(f"Band names   : {list(product.getBandNames())}")
print(f"Scene size   : {product.getSceneRasterWidth()} x {product.getSceneRasterHeight()}")

# ---------------------------------------------------------------------------
# 4. Resample to a common resolution
# ---------------------------------------------------------------------------
# BiophysicalOp requires all bands at the same resolution.
# Sentinel-2 L2A has bands at 10m, 20m, and 60m — resample to 10m.
print("\nResampling all bands to 10 m ...")

resample_params = HashMap()
resample_params.put("targetResolution", 10)

resampled = GPF.createProduct("Resample", resample_params, product)

print(f"Resampled bands: {list(resampled.getBandNames())}")
print(f"Resampled size : {resampled.getSceneRasterWidth()} x {resampled.getSceneRasterHeight()}")

# ---------------------------------------------------------------------------
# 5. Run BiophysicalOp  (fPAR, LAI, fCover, Cab, Cw)
# ---------------------------------------------------------------------------
# KEY FIX: The original code incorrectly passed biophysical parameters to
# the 'Resample' operator. The correct operator is 'BiophysicalOp'.
print("\nRunning BiophysicalOp ...")

bio_params = HashMap()
bio_params.put("computeLAI", True)
bio_params.put("computeFapar", True)
bio_params.put("computeFcover", True)
bio_params.put("computeCab", True)
bio_params.put("computeCw", True)
bio_params.put("sensor", "S2A")       # Change to "S2B" / "S2C" if needed
bio_params.put("resolution", "10")    # Must match resampled resolution

biophys_product = GPF.createProduct("BiophysicalOp", bio_params, resampled)

bio_bands = list(biophys_product.getBandNames())
print(f"Biophysical bands: {bio_bands}")

# ---------------------------------------------------------------------------
# 6. Read results into numpy arrays
# ---------------------------------------------------------------------------
width = biophys_product.getSceneRasterWidth()
height = biophys_product.getSceneRasterHeight()

results = {}
for band_name in bio_bands:
    band = biophys_product.getBand(band_name)
    if band is None:
        continue
    buf = np.zeros((height, width), dtype=np.float32)
    band.readPixels(0, 0, width, height, buf)
    results[band_name] = buf
    valid = buf[buf != band.getNoDataValue()]
    if valid.size > 0:
        print(f"  {band_name:12s}  min={valid.min():.4f}  max={valid.max():.4f}  mean={valid.mean():.4f}")
    else:
        print(f"  {band_name:12s}  (no valid pixels)")

# ---------------------------------------------------------------------------
# 7. Export to GeoTIFF (optional)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_PATH), "biophysical_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file = os.path.join(OUTPUT_DIR, "biophysical_result.tif")
print(f"\nWriting output to: {output_file}")
ProductIO.writeProduct(biophys_product, output_file, "GeoTIFF")
print("GeoTIFF written successfully.")

# Also save as BEAM-DIMAP for further SNAP processing
dim_file = os.path.join(OUTPUT_DIR, "biophysical_result")
ProductIO.writeProduct(biophys_product, dim_file, "BEAM-DIMAP")
print(f"BEAM-DIMAP written to: {dim_file}.dim")

# ---------------------------------------------------------------------------
# 8. Quick visualization with matplotlib (if available)
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Plot fPAR and LAI side by side
    plot_bands = [b for b in ["fapar", "lai"] if b in results]
    if not plot_bands:
        # Try alternate naming
        plot_bands = [b for b in bio_bands if "fapar" in b.lower() or "lai" in b.lower()]

    if plot_bands:
        fig, axes = plt.subplots(1, len(plot_bands), figsize=(6 * len(plot_bands), 5))
        if len(plot_bands) == 1:
            axes = [axes]
        for ax, bname in zip(axes, plot_bands):
            data = results.get(bname, results.get(bname.lower()))
            if data is not None:
                im = ax.imshow(data, cmap="YlGn", vmin=0)
                ax.set_title(bname.upper())
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "fpar_lai_preview.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Preview saved to: {fig_path}")
        plt.show()
    else:
        print("No fPAR/LAI bands found for plotting.")
except ImportError:
    print("matplotlib not available — skipping visualization.")

# ---------------------------------------------------------------------------
# 9. Cleanup
# ---------------------------------------------------------------------------
biophys_product.dispose()
resampled.dispose()
product.dispose()

print("\nDone.")
