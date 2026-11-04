"""Report-ready, visually explicit WorldCover 2021 vs SARTrack v3 2025 map.

Run from anywhere: python SARTrack_WorldCover_Report_Figure.py
Put this script in the SARTrack project root beside app.py.
Requires numpy, rasterio, and matplotlib (same dependencies as the case-study script).
Does not update model weights or overwrite existing comparison metrics/figures.
"""
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "results_v3" / "esch_independent_comparison" / "esch_prediction_v3.tif"
WC_PATH = ROOT / "data" / "external" / "Esch_WorldCover_2021_Aligned_to_SAR.tif"
OUT_PATH = ROOT / "results_v3" / "worldcover_case_study" / "worldcover_vs_v3_report.png"

# Exact mapping used by SARTrack_WorldCover_Case_Study.py.
WC_TO_MODEL = {80: 0, 10: 1, 30: 2, 40: 3, 50: 4}
CLASS_NAMES = ["Water", "Trees", "Grass", "Crops", "Built-up"]
CLASS_COLORS = ["#3388cc", "#176b33", "#e7da54", "#b78dcf", "#cf3535"]
EXCLUDED = "#c5c5c5"
AGREEMENT = "#2c974b"
DISAGREEMENT = "#d93838"


def main():
    for path in (PRED_PATH, WC_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    with rasterio.open(PRED_PATH) as pr, rasterio.open(WC_PATH) as wr:
        if pr.count != 1 or wr.count != 1:
            raise ValueError("Expected single-band categorical GeoTIFFs")
        if not (pr.crs == wr.crs and pr.shape == wr.shape and
                pr.transform.almost_equals(wr.transform)):
            raise ValueError("Raster grids differ; stop rather than compare unaligned pixels")
        pred = pr.read(1)
        wc = wr.read(1)
        pred_mask = pr.read_masks(1) != 0
        wc_mask = wr.read_masks(1) != 0

    eligible = (pred_mask & wc_mask & np.isin(pred, np.arange(5)) &
                np.isin(wc, np.fromiter(WC_TO_MODEL, dtype=np.uint8)))
    n = int(eligible.sum())
    if n == 0:
        raise ValueError("No comparable pixels")

    mapped_wc = np.full(wc.shape, 255, dtype=np.uint8)
    for wc_class, model_class in WC_TO_MODEL.items():
        mapped_wc[wc == wc_class] = model_class

    agreement = float(np.count_nonzero((pred == mapped_wc) & eligible) / n)
    ref_display = np.ma.array(mapped_wc, mask=~eligible)
    pred_display = np.ma.array(pred, mask=~eligible)
    # 0 means agreement (green), 1 disagreement (red).
    diff_display = np.ma.array((pred != mapped_wc).astype(np.uint8), mask=~eligible)

    class_map = ListedColormap(CLASS_COLORS)
    class_map.set_bad(EXCLUDED)
    class_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), class_map.N)
    diff_map = ListedColormap([AGREEMENT, DISAGREEMENT])
    diff_map.set_bad(EXCLUDED)
    diff_norm = BoundaryNorm([-0.5, 0.5, 1.5], diff_map.N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.79, bottom=0.24, wspace=0.055)

    axes[0].imshow(ref_display, cmap=class_map, norm=class_norm, interpolation="nearest")
    axes[1].imshow(pred_display, cmap=class_map, norm=class_norm, interpolation="nearest")
    axes[2].imshow(diff_display, cmap=diff_map, norm=diff_norm, interpolation="nearest")
    axes[0].set_title("ESA WorldCover reference (2021)", fontsize=12, pad=10)
    axes[1].set_title("SARTrack U-Net v3 (June 2025)", fontsize=12, pad=10)
    axes[2].set_title(f"Class agreement: {agreement:.2%}", fontsize=12, pad=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle("SARTrack | Esch/Belval cross-year land-cover comparison",
                 fontsize=18, fontweight="semibold", y=0.972)
    fig.text(0.5, 0.913, "Matched 10 m raster grid  •  Five common land-cover classes  •  Same comparable-pixel mask",
             ha="center", fontsize=10.5, color="#444444")
    fig.text(0.5, 0.199, "LAND-COVER CLASSES (FIRST TWO PANELS)",
             ha="center", fontsize=9, fontweight="bold", color="#333333")
    class_handles = [Patch(facecolor=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=class_handles, ncol=5, loc="center", bbox_to_anchor=(0.5, 0.166),
               frameon=False, fontsize=10, handlelength=1.5, columnspacing=2.5)

    fig.text(0.5, 0.119, "THIRD PANEL — PIXEL-WISE COMPARISON",
             ha="center", fontsize=9, fontweight="bold", color="#333333")
    diff_handles = [Patch(facecolor=AGREEMENT, label="Agreement"),
                    Patch(facecolor=DISAGREEMENT, label="Disagreement"),
                    Patch(facecolor=EXCLUDED, label="Excluded / NoData")]
    fig.legend(handles=diff_handles, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.087),
               frameon=False, fontsize=10, handlelength=1.5, columnspacing=3.5)

    fig.text(0.5, 0.025,
             f"Compared {n:,} of {pred.size:,} pixels ({n/pred.size:.2%}). "
             "2021 reference vs 2025 prediction: agreement, NOT independently verified 2025 accuracy.",
             ha="center", fontsize=9.1, color="#444444")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    print("EXACT GRID VERIFIED")
    print(f"Comparable pixels: {n:,} / {pred.size:,}")
    print(f"Cross-year agreement: {agreement:.2%}")
    print("Report figure:", OUT_PATH)
    print("REPORT FIGURE COMPLETE")


if __name__ == "__main__":
    main()
