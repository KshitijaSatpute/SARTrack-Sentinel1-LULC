"""Fair, reproducible independent Esch/Belval comparison of SARTrack U-Net v2 and v3.

Place this beside SARTrack_v3_DataLab_Train.py in vault/SARTrack_v3.
Run in SARTrack_v3_Training.ipynb with: %run SARTrack_Esch_Compare_V2_V3.py

Replicates the original 03_Independent_Test.ipynb evaluation protocol:
- full Esch June 2025 raster, Sentinel-1 VV/VH only
- VV clip [-30, 5] dB; VH clip [-35, 0] dB; scale both to [0,1]
- non-overlapping 128x128 tiles, zeros for incomplete tile edges
- raw Dynamic World class 0/1/2/4/6 -> 0/1/2/3/4; ignore all others
- only pixels with finite SAR and valid reference labels scored
- global confusion matrices and pixel-level OA, mIoU, macro F1
The reference is optical-derived Dynamic World pseudo-ground-truth, not field truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    import rasterio
except ImportError as e:
    raise ImportError(
        "rasterio is required for geo-referenced Esch evaluation. "
        "In JupyterLab run `%pip install rasterio` once and restart kernel if needed."
    ) from e

from SARTrack_v3_DataLab_Train import UNet, CLASS_NAMES

PROJECT = Path(__file__).resolve().parent
TEST_DIR = PROJECT / "independent_test"
TEST_RASTER = TEST_DIR / "Esch_Belval_June2025_Test.tif"
OUTPUT = PROJECT / "results_v3" / "esch_independent_comparison"
MODEL_PATHS = {
    "v2": TEST_DIR / "best_unet_v2.pth",
    "v3": TEST_DIR / "best_unet_v3.pth",
}
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
N_CLASSES = 5
TILE_SIZE = 128
IGNORE = 255


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(reference: np.ndarray, prediction: np.ndarray):
    valid = (reference != IGNORE) & (prediction != IGNORE)
    if not valid.any():
        raise RuntimeError("No valid reference/prediction pixels for evaluation")
    cm = np.bincount(
        reference[valid].astype(np.int64) * N_CLASSES + prediction[valid].astype(np.int64),
        minlength=N_CLASSES * N_CLASSES,
    ).reshape((N_CLASSES, N_CLASSES))
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    f1_denom = support + predicted
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp), where=f1_denom > 0)
    oa = tp.sum() / cm.sum()
    summary = {
        "evaluated_pixels": int(cm.sum()),
        "OA_pct": float(100 * oa),
        "mIoU_pct": float(100 * iou.mean()),
        "macro_F1_pct": float(100 * f1.mean()),
    }
    class_rows = [
        {"class_id": i, "class_name": CLASS_NAMES[i], "reference_pixels": int(support[i]),
         "predicted_pixels": int(predicted[i]), "IoU_pct": float(100 * iou[i]),
         "F1_pct": float(100 * f1[i])}
        for i in range(N_CLASSES)
    ]
    return cm, summary, class_rows


def run_inference(model: torch.nn.Module, x: np.ndarray, finite_sar: np.ndarray, device: torch.device):
    height, width = x.shape[1:]
    prediction = np.zeros((height, width), dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for row in range(0, height, TILE_SIZE):
            for col in range(0, width, TILE_SIZE):
                tile = x[:, row:row + TILE_SIZE, col:col + TILE_SIZE]
                h, w = tile.shape[-2:]
                padded = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                padded[:, :h, :w] = tile
                inp = torch.from_numpy(padded[None]).to(device)
                pred = model(inp).argmax(dim=1)[0, :h, :w]
                prediction[row:row + h, col:col + w] = pred.cpu().numpy().astype(np.uint8)
    prediction[~finite_sar] = IGNORE
    return prediction


def save_geotiff(path: Path, arr: np.ndarray, profile: dict):
    profile = profile.copy()
    profile.update(count=1, dtype="uint8", nodata=IGNORE, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def plot_overview(x: np.ndarray, reference: np.ndarray, pred_v2: np.ndarray, pred_v3: np.ndarray):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib unavailable; skipping comparison PNG", flush=True)
        return
    cmap = ListedColormap(["#267ac0", "#1e8e3e", "#a6cf66", "#f1cd55", "#db554f"])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    axes[0].imshow(np.ma.masked_where(reference == IGNORE, x[0]), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Sentinel-1 VV (normalized)")
    for ax, arr, title in [
        (axes[1], reference, "Dynamic World reference"),
        (axes[2], pred_v2, "U-Net v2"),
        (axes[3], pred_v3, "U-Net v3"),
    ]:
        ax.imshow(np.ma.masked_equal(arr, IGNORE), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Esch/Belval, June 2025 — independent spatial holdout (pseudo-label evaluation)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "esch_v2_v3_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    for path in (TEST_RASTER, *MODEL_PATHS.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required test file missing: {path}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; PyTorch: {torch.__version__}", flush=True)
    with rasterio.open(TEST_RASTER) as dataset:
        if dataset.count != 3:
            raise ValueError(f"Expected 3 TIFF bands (VV, VH, DW label), received {dataset.count}")
        data = dataset.read().astype(np.float32)
        profile = dataset.profile.copy()
        print(f"Raster: {dataset.width} x {dataset.height}; CRS: {dataset.crs}", flush=True)
        print(f"Band descriptions: {dataset.descriptions}", flush=True)
    sar = data[:2]
    labels = data[2]
    finite_sar = np.isfinite(sar).all(axis=0)
    finite_all = np.isfinite(data).all(axis=0)
    reference = np.full(labels.shape, IGNORE, dtype=np.uint8)
    for raw_class, class_index in LABEL_MAPPING.items():
        reference[finite_all & (labels == raw_class)] = class_index
    del data

    lower = np.array([-30.0, -35.0], dtype=np.float32)[:, None, None]
    upper = np.array([5.0, 0.0], dtype=np.float32)[:, None, None]
    # Matches original 03_Independent_Test.ipynb preprocessing exactly.
    x = np.nan_to_num(
        (np.clip(sar, lower, upper) - lower) / (upper - lower),
        nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)
    del sar

    summaries = []
    preds = {}
    for version, weights_path in MODEL_PATHS.items():
        model = UNet().to(device)
        if sum(p.numel() for p in model.parameters()) != 488629:
            raise RuntimeError("Unexpected model architecture / parameter count")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        prediction = run_inference(model, x, finite_sar, device)
        del model, checkpoint
        preds[version] = prediction
        cm, summary, class_rows = evaluate(reference, prediction)
        summary = {"model": version, **summary}
        summaries.append(summary)
        np.savetxt(OUTPUT / f"confusion_matrix_{version}.csv", cm, fmt="%d", delimiter=",")
        write_csv(
            OUTPUT / f"per_class_metrics_{version}.csv",
            ["class_id", "class_name", "reference_pixels", "predicted_pixels", "IoU_pct", "F1_pct"],
            class_rows,
        )
        save_geotiff(OUTPUT / f"esch_prediction_{version}.tif", prediction, profile)
        print(
            f"{version.upper()} | pixels={summary['evaluated_pixels']:,} | "
            f"OA={summary['OA_pct']:.2f}% | mIoU={summary['mIoU_pct']:.2f}% | "
            f"macro F1={summary['macro_F1_pct']:.2f}%",
            flush=True,
        )
        for item in class_rows:
            print(f"  {item['class_name']:<9} IoU={item['IoU_pct']:.2f}%  F1={item['F1_pct']:.2f}%", flush=True)

    assert summaries[0]["evaluated_pixels"] == summaries[1]["evaluated_pixels"], "Different test masks!"
    write_csv(OUTPUT / "v2_vs_v3_summary.csv", ["model", "evaluated_pixels", "OA_pct", "mIoU_pct", "macro_F1_pct"], summaries)
    plot_overview(x, reference, preds["v2"], preds["v3"])
    save_geotiff(OUTPUT / "esch_reference_5class.tif", reference, profile)
    provenance = {
        "test_raster": TEST_RASTER.name,
        "test_raster_sha256": sha256(TEST_RASTER),
        "model_sha256": {k: sha256(p) for k, p in MODEL_PATHS.items()},
        "reference": "Dynamic World optical-derived pseudo-labels (not independent ground truth)",
        "test_role": "Held-out Esch/Belval geography, no training or model tuning",
        "evaluation": "Exact protocol matched to original 03_Independent_Test.ipynb; 128x128 zero-padded tiles",
        "ignore_raw_labels": "everything except 0,1,2,4,6, plus any nonfinite raster pixels",
        "normalization": {"VV_dB": [-30, 5], "VH_dB": [-35, 0]},
        "models": summaries,
        "limitations": [
            "No independent human-annotated reference; reported scores measure agreement with Dynamic World pseudo-labels.",
            "v2 and v3 experiments changed data confidence filtering, augmentation and training duration together.",
            "No hyperparameter or checkpoint selection should use this held-out Esch evaluation.",
        ],
    }
    (OUTPUT / "evaluation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    delta = summaries[1]["mIoU_pct"] - summaries[0]["mIoU_pct"]
    print(f"\nDELTA v3 - v2 mIoU: {delta:+.2f} percentage points", flush=True)
    print("Results saved:", OUTPUT, flush=True)
    print("INDEPENDENT COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
