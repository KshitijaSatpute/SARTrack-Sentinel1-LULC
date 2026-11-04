"""SARTrack v3 training on the Data Lab JupyterLab GPU server.

Put this file in vault/SARTrack_v3 next to processed_v2/{train,val}.
Launch in SARTrack_v3_Training.ipynb via: %run SARTrack_v3_DataLab_Train.py

The script does not alter the original dataset or model. It writes only
results_v3/ under this script's directory, and resumes from an epoch checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SEED = 42
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IGNORE_INDEX = 255
CLASS_NAMES = ["Water", "Trees", "Grass", "Crops", "Built-up"]

PROJECT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT / "processed_v2"
OUTPUT = PROJECT / "results_v3"
ARCHIVE = PROJECT / "SARTrack_processed_v2.zip"


def save_torch_atomic(obj, path: Path):
    """Avoid partial checkpoints on disconnection/interrupted writes."""
    temp = path.with_name(path.name + ".partial")
    torch.save(obj, temp)
    os.replace(temp, path)


def save_text_atomic(content: str, path: Path):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def get_archive_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Source ZIP missing: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


class SARTrackDataset(Dataset):
    def __init__(self, paths: list[Path], augment: bool = False):
        self.paths = paths
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with np.load(self.paths[i]) as data:
            x = data["x"].astype(np.float32)
            y = data["y"].astype(np.int64)
        if self.augment:
            k = random.randrange(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Exact 488,629-parameter two-channel/five-class SARTrack v2 architecture."""

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(2, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.bottleneck = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DoubleConv(128 + 64, 64)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.dec1 = DoubleConv(32 + 16, 16)
        self.output = nn.Conv2d(16, 5, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output(d1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; check the Data Lab GPU server.")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Project:", PROJECT, flush=True)

    train_files = sorted((DATA_ROOT / "train").glob("*.npz"))
    val_files = sorted((DATA_ROOT / "val").glob("*.npz"))
    if (len(train_files), len(val_files)) != (294, 196):
        raise RuntimeError(f"Expected 294 train, 196 val; found {len(train_files)}, {len(val_files)}")

    dataset_hash = get_archive_hash(ARCHIVE)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Pixel weights and water-aware patch sampling are based on training data only.
    class_counts = np.zeros(5, dtype=np.int64)
    sample_weights = []
    for path in train_files:
        with np.load(path) as z:
            x = z["x"]
            y = z["y"]
        if x.shape != (2, 128, 128) or y.shape != (128, 128):
            raise ValueError(f"Unexpected patch shape: {path.name}")
        if not np.isfinite(x).all() or not (0 <= x.min() and x.max() <= 1):
            raise ValueError(f"Invalid SAR normalization: {path.name}")
        if not set(np.unique(y)).issubset({0, 1, 2, 3, 4, 255}):
            raise ValueError(f"Invalid labels: {path.name}")
        valid = y != IGNORE_INDEX
        class_counts += np.bincount(y[valid].astype(np.int64), minlength=5)
        water_fraction = float((y == 0).sum() / max(int(valid.sum()), 1))
        sample_weights.append(1.0 + 4.0 * min(water_fraction / 0.10, 1.0))

    if np.any(class_counts == 0):
        raise RuntimeError("At least one class is missing from corrected training labels")
    freq = class_counts / class_counts.sum()
    weights = torch.tensor(1.0 / np.sqrt(freq), dtype=torch.float32)
    weights = (weights / weights.mean()).clamp(max=5.0).to(device)
    print("Train class frequencies (%):", (100 * freq).round(3).tolist(), flush=True)
    print("Loss class weights:", weights.cpu().numpy().round(3).tolist(), flush=True)

    train_loader = DataLoader(
        SARTrackDataset(train_files, augment=True), batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(train_files), replacement=True),
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        SARTrackDataset(val_files), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    model = UNet().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 488629:
        raise RuntimeError(f"Unexpected U-Net parameter count: {parameters}")
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    OUTPUT.mkdir(exist_ok=True)
    best_path = OUTPUT / "best_unet_v3.pth"   # Standard state_dict compatible with SARTrack inference.
    resume_path = OUTPUT / "last_training_state_v3.pt"
    history_path = OUTPUT / "history_v3.csv"
    summary_path = OUTPUT / "summary_v3.json"
    config_path = OUTPUT / "config_v3.json"

    config = {
        "experiment": "v3_corrected_june_water_sampling_flip_rotate_20epoch_max",
        "dataset_archive_sha256": dataset_hash,
        "seed": SEED,
        "training_patches": len(train_files),
        "validation_patches": len(val_files),
        "classes": CLASS_NAMES,
        "class_frequencies": freq.tolist(),
        "class_weights": weights.cpu().tolist(),
        "architecture": "SARTrack U-Net 2-in 5-out 488629 parameters",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "checkpoint_selection": "validation mIoU, validation loss tie break",
        "train_augmentation": "random 90 degree rotations and horizontal/vertical flips",
        "sampler": "water-aware weighted sampler with replacement",
        "limitations": [
            "Dynamic World pseudo-labels, not independent ground truth",
            "validation patches overlap",
            "March export confidence threshold not fully verified",
            "augmentation may alter radar orientation statistics",
            "data correction, augmentation and longer training change together",
        ],
    }
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config.get("dataset_archive_sha256") != dataset_hash:
            raise RuntimeError("Existing results_v3 config belongs to a different dataset")
    else:
        save_text_atomic(json.dumps(config, indent=2), config_path)

    history = []
    best_iou, best_loss, best_epoch, stale, start_epoch = -1.0, float("inf"), 0, 0, 1
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=True)
        if state["dataset_archive_sha256"] != dataset_hash:
            raise RuntimeError("Resume checkpoint dataset fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_iou, best_loss = state["best_iou"], state["best_loss"]
        best_epoch, stale = state["best_epoch"], state["stale"]
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming after epoch {start_epoch-1}; best epoch: {best_epoch}", flush=True)
    else:
        print("Starting fresh v3 experiment", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if stale >= PATIENCE:
            print("Previous checkpoint already met early-stopping patience", flush=True)
            break
        model.train()
        cumulative_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item()
        train_loss = cumulative_loss / len(train_loader)

        model.eval()
        confusion = torch.zeros((5, 5), dtype=torch.int64)
        val_total_loss = 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_total_loss += loss_fn(logits, y).item()
                prediction = logits.argmax(dim=1)
                valid = y != IGNORE_INDEX
                confusion += torch.bincount(
                    (y[valid] * 5 + prediction[valid]).cpu(), minlength=25
                ).reshape(5, 5)
        val_loss = val_total_loss / len(val_loader)
        tp = confusion.diag().float()
        actual, predicted = confusion.sum(1).float(), confusion.sum(0).float()
        per_class_iou = (tp / (actual + predicted - tp).clamp_min(1)).tolist()
        per_class_f1 = (2 * tp / (actual + predicted).clamp_min(1)).tolist()
        miou = float(sum(per_class_iou) / 5)
        oa = float((tp.sum() / confusion.sum().clamp_min(1)).item())
        macro_f1 = float(sum(per_class_f1) / 5)
        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_OA": oa, "val_mIoU": miou, "val_macro_F1": macro_f1,
        }
        record.update({f"IoU_{name}": val for name, val in zip(CLASS_NAMES, per_class_iou)})
        history.append(record)

        improved = (miou > best_iou + 1e-6) or (abs(miou - best_iou) <= 1e-6 and val_loss < best_loss)
        if improved:
            best_iou, best_loss, best_epoch, stale = miou, val_loss, epoch, 0
            save_torch_atomic(model.state_dict(), best_path)
        else:
            stale += 1

        # All artifacts remain under the persistent vault project directory.
        headers = list(history[0])
        import io
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(history)
        save_text_atomic(stream.getvalue(), history_path)
        state = {
            "dataset_archive_sha256": dataset_hash, "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best_iou": best_iou, "best_loss": best_loss,
            "best_epoch": best_epoch, "stale": stale,
        }
        save_torch_atomic(state, resume_path)
        save_text_atomic(json.dumps({
            "last_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_mIoU": best_iou,
            "best_val_loss": best_loss,
            "epochs_without_improvement": stale,
            "last_val_OA": oa,
            "last_val_macro_F1": macro_f1,
            "last_val_class_IoUs": dict(zip(CLASS_NAMES, per_class_iou)),
            "last_val_class_F1s": dict(zip(CLASS_NAMES, per_class_f1)),
        }, indent=2), summary_path)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val mIoU {100*miou:.2f}% | "
            f"OA {100*oa:.2f}% | macro F1 {100*macro_f1:.2f}% "
            f"{'[BEST]' if improved else ''}", flush=True
        )
        if stale >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without mIoU improvement", flush=True)
            break

    print("TRAINING SESSION COMPLETE", flush=True)
    print("Best epoch:", best_epoch)
    print("Best val mIoU (%):", round(100 * best_iou, 2))
    print("Best model saved:", best_path)
    print("Epoch history saved:", history_path)
    print("Next: evaluate best checkpoint on untouched Esch independent test")


if __name__ == "__main__":
    main()
