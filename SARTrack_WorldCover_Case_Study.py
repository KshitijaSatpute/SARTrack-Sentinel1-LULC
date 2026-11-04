"""SARTrack external reference sanity check: June 2025 v3 vs ESA WorldCover 2021.

Run from the SARTrack project root using Python with NumPy, rasterio, matplotlib.
No PyTorch/GPU needed. This is NOT a contemporaneous ground-truth evaluation.
"""
from pathlib import Path
import csv
import json

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
PREDICTION = ROOT / "results_v3" / "esch_independent_comparison" / "esch_prediction_v3.tif"
REFERENCE = ROOT / "data" / "external" / "Esch_WorldCover_2021_Aligned_to_SAR.tif"
OUTPUT = ROOT / "results_v3" / "worldcover_case_study"
CLASSES = ["Water", "Trees", "Grass", "Crops", "Built-up"]
# ESA WorldCover 2021 v200 Map codes -> SARTrack output codes.
# Other WorldCover classes (e.g. Bare/Sparse=60) have no model equivalent.
WC_TO_MODEL = {80: 0, 10: 1, 30: 2, 40: 3, 50: 4}
PALETTE = ["#3388cc", "#176b33", "#e7da54", "#b78dcf", "#cf3535"]


def save_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    for path in (PREDICTION, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(f"Required raster not found: {path}")

    with rasterio.open(PREDICTION) as prediction_src, rasterio.open(REFERENCE) as reference_src:
        if prediction_src.count != 1 or reference_src.count != 1:
            raise ValueError("Both files must be single-band categorical rasters")
        grid_ok = (
            prediction_src.crs == reference_src.crs
            and prediction_src.shape == reference_src.shape
            and prediction_src.transform.almost_equals(reference_src.transform)
        )
        print("Prediction grid:", prediction_src.shape, prediction_src.crs, prediction_src.transform)
        print("WorldCover grid:", reference_src.shape, reference_src.crs, reference_src.transform)
        if not grid_ok:
            raise ValueError("GRID MISMATCH: stop. Do not compare pixels before exact alignment")

        pred = prediction_src.read(1)
        wc = reference_src.read(1)
        pred_data_valid = prediction_src.read_masks(1) != 0
        wc_data_valid = reference_src.read_masks(1) != 0
        transform = prediction_src.transform
        crs = str(prediction_src.crs)

    pred_valid = pred_data_valid & np.isin(pred, [0, 1, 2, 3, 4])
    wc_valid = wc_data_valid & np.isin(wc, list(WC_TO_MODEL))
    eligible = pred_valid & wc_valid
    eligible_count = int(eligible.sum())
    total_count = int(pred.size)
    if eligible_count == 0:
        raise ValueError("No comparable pixels after filtering reference/prediction masks")

    label = np.full(wc.shape, 255, dtype=np.uint8)
    for wc_code, model_code in WC_TO_MODEL.items():
        label[wc == wc_code] = model_code

    y_true = label[eligible].astype(np.int64)
    y_pred = pred[eligible].astype(np.int64)
    cm = np.bincount(y_true * 5 + y_pred, minlength=25).reshape(5, 5)
    tp = np.diag(cm)
    true_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    union = true_count + pred_count - tp
    iou = np.divide(tp, union, out=np.full(5, np.nan), where=union != 0)
    denom_f1 = true_count + pred_count
    f1 = np.divide(2 * tp, denom_f1, out=np.full(5, np.nan), where=denom_f1 != 0)
    accuracy = float(tp.sum() / eligible_count)
    miou = float(np.nanmean(iou))
    macro_f1 = float(np.nanmean(f1))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_class = []
    for idx, name in enumerate(CLASSES):
        per_class.append([
            name, int(true_count[idx]), int(pred_count[idx]), int(tp[idx]),
            round(float(iou[idx]) * 100, 4) if np.isfinite(iou[idx]) else "NA",
            round(float(f1[idx]) * 100, 4) if np.isfinite(f1[idx]) else "NA",
        ])

    save_csv(
        OUTPUT / "worldcover_v3_per_class.csv",
        ["class", "worldcover_pixels", "predicted_pixels", "true_positive_pixels", "iou_pct", "f1_pct"],
        per_class,
    )
    save_csv(
        OUTPUT / "worldcover_v3_confusion_matrix.csv",
        ["WorldCover_2021\\Pred_V3_2025"] + CLASSES,
        [[name] + cm[i].tolist() for i, name in enumerate(CLASSES)],
    )
    save_csv(
        OUTPUT / "worldcover_v3_summary.csv",
        ["metric", "value"],
        [
            ["reference_year", "2021"], ["sar_prediction_year", "2025"],
            ["total_pixels", total_count], ["comparable_pixels", eligible_count],
            ["excluded_pixels", total_count - eligible_count],
            ["comparable_fraction_pct", round(100 * eligible_count / total_count, 4)],
            ["overall_agreement_pct", round(100 * accuracy, 4)],
            ["mean_iou_pct", round(100 * miou, 4)],
            ["macro_f1_pct", round(100 * macro_f1, 4)],
        ],
    )

    mapped_wc = np.ma.masked_where(~eligible, label)
    masked_pred = np.ma.masked_where(~eligible, pred)
    disagreement = np.ma.masked_where(~eligible, (pred != label).astype(np.uint8))
    cmap = ListedColormap(PALETTE)
    cmap.set_bad("#c5c5c5")
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(mapped_wc, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("ESA WorldCover 2021 (mapped)")
    axes[1].imshow(masked_pred, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("SARTrack U-Net v3 (June 2025)")
    diff_cmap = ListedColormap(["#2c974b", "#d93838"])
    diff_cmap.set_bad("#c5c5c5")
    axes[2].imshow(disagreement, cmap=diff_cmap, vmin=-0.5, vmax=1.5, interpolation="nearest")
    axes[2].set_title("Cross-year agreement / disagreement")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    legend = [Patch(facecolor=PALETTE[i], label=CLASSES[i]) for i in range(5)]
    legend += [Patch(facecolor="#c5c5c5", label="Excluded/NoData")]
    fig.legend(handles=legend, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.025))
    fig.savefig(OUTPUT / "worldcover_vs_v3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    details = {
        "comparison_type": "cross-year external reference agreement, not ground-truth accuracy",
        "reference": "ESA WorldCover 2021 v200",
        "prediction": "SARTrack v3 Sentinel-1 June 2025",
        "reference_path": str(REFERENCE), "prediction_path": str(PREDICTION),
        "shape_rows_cols": list(wc.shape), "crs": crs, "transform": list(transform)[:6],
        "mapping_worldcover_to_model": {str(k): v for k, v in WC_TO_MODEL.items()},
        "excluded_worldcover_codes": sorted(int(c) for c in np.unique(wc) if int(c) not in WC_TO_MODEL),
        "total_pixels": total_count, "comparable_pixels": eligible_count,
        "overall_agreement": accuracy, "mean_iou": miou, "macro_f1": macro_f1,
        "caveats": [
            "2021 and 2025 land cover may genuinely differ",
            "WorldCover classes and SARTrack classes are not perfectly synonymous",
            "WorldCover is itself a model-generated map using Sentinel-1 and Sentinel-2, not field ground truth",
            "Other WorldCover classes and invalid pixels excluded from scores",
            "Metrics are descriptive cross-year agreement and must not be presented as independent 2025 accuracy",
        ],
    }
    (OUTPUT / "comparison_provenance.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    print("GRID ALIGNMENT VERIFIED")
    print(f"Comparable pixels: {eligible_count:,} / {total_count:,} ({100*eligible_count/total_count:.2f}%)")
    print(f"Excluded WorldCover codes: {details['excluded_worldcover_codes']}")
    print(f"Cross-year agreement (NOT 2025 ground-truth accuracy): OA={accuracy:.2%}, mIoU={miou:.2%}, Macro F1={macro_f1:.2%}")
    for row in per_class:
        print(f"  {row[0]:<9} IoU={row[4]}%  F1={row[5]}%  WorldCover pixels={row[1]:,}")
    print("Results saved:", OUTPUT)
    print("EXTERNAL SANITY CHECK COMPLETE")


if __name__ == "__main__":
    main()
