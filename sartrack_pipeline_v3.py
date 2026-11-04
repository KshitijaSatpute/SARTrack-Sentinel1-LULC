"""SARTrack: 12-month Sentinel-1 inference for a user-selected point."""
from pathlib import Path
from datetime import date
import calendar
import csv

import ee
import requests
import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent
PROJECT_ID = "sartrack-lulc-2026"
WEIGHTS = ROOT / "models" / "best_unet_v3.pth"
CLASSES = ["Water", "Trees", "Grass", "Crops", "Built-up"]
COLORS = ["#2787c7", "#217a43", "#a5d66d", "#e9c95f", "#d85b55"]
LOWER = np.array([-30, -35], dtype=np.float32)[:, None, None]
UPPER = np.array([5, 0], dtype=np.float32)[:, None, None]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = DoubleConv(2, 16), DoubleConv(16, 32), DoubleConv(32, 64)
        self.bottleneck = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec3, self.dec2, self.dec1 = DoubleConv(192, 64), DoubleConv(96, 32), DoubleConv(48, 16)
        self.output = nn.Conv2d(16, 5, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], 1))
        return self.output(d1)


def add_months(year, month, offset):
    index = year * 12 + (month - 1) + offset
    return date(index // 12, index % 12 + 1, 1)


def infer(model, device, sar):
    valid = np.isfinite(sar).all(axis=0)
    x = np.nan_to_num((np.clip(sar, LOWER, UPPER) - LOWER) / (UPPER - LOWER),
                      nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h, w = x.shape[1:]
    prediction = np.full((h, w), 255, dtype=np.uint8)
    with torch.inference_mode():
        for r in range(0, h, 128):
            for c in range(0, w, 128):
                tile = x[:, r:r + 128, c:c + 128]
                th, tw = tile.shape[1:]
                padded = np.zeros((2, 128, 128), dtype=np.float32)
                padded[:, :th, :tw] = tile
                result = model(torch.from_numpy(padded[None]).to(device)).argmax(1)
                prediction[r:r + th, c:c + tw] = result[0, :th, :tw].cpu().numpy()
    prediction[~valid] = 255
    return prediction, valid


def save_png(values, path, title):
    masked = np.ma.masked_where(values == 255, values)
    cmap = ListedColormap(COLORS)
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(masked, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tif(values, path, profile):
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype="uint8", nodata=255,
             compress="lzw", photometric="MINISBLACK")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(values, 1)


def run_pipeline(latitude, longitude, start_year, start_month, progress=None):
    """Generate real SAR-only predictions. Returns directory containing outputs."""
    latitude, longitude = float(latitude), float(longitude)
    start_year, start_month = int(start_year), int(start_month)
    if not (-80 <= latitude < 84 and -180 <= longitude <= 180):
        raise ValueError("Select coordinates within the supported UTM latitude range (-80 to 84).")
    if not (1 <= start_month <= 12 and start_year >= 2015):
        raise ValueError("Invalid starting month/year.")
    if add_months(start_year, start_month, 12) > date.today().replace(day=1):
        raise ValueError("Choose 12 complete past calendar months.")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights missing: {WEIGHTS}")

    def notify(message):
        print(message, flush=True)
        if progress is not None:
            progress(message)

    ee.Initialize(project=PROJECT_ID)
    area = ee.Geometry.Point([longitude, latitude]).buffer(2500).bounds()
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    epsg = f"EPSG:{32600 + zone if latitude >= 0 else 32700 + zone}"
    dates = [add_months(start_year, start_month, i) for i in range(13)]
    folder = ROOT / "outputs" / "live_v3" / f"lat{latitude:+.4f}_lon{longitude:+.4f}_{start_year}_{start_month:02d}"
    folder.mkdir(parents=True, exist_ok=True)

    collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(area).filterDate(dates[0].isoformat(), dates[-1].isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")))
    histogram = collection.aggregate_histogram("relativeOrbitNumber_start").getInfo()
    if not histogram:
        raise RuntimeError("No matching ascending dual-polarization Sentinel-1 scenes.")
    orbit = int(float(max(histogram, key=histogram.get)))
    notify(f"Using ascending relative orbit {orbit}; projection {epsg}")
    collection = collection.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

    rasters = []
    grid = None
    profile = None
    for i in range(12):
        month_start, month_end = dates[i], dates[i + 1]
        tag = month_start.strftime("%Y_%m")
        path = folder / f"SAR_{tag}.tif"
        scenes = collection.filterDate(month_start.isoformat(), month_end.isoformat())
        count = scenes.size().getInfo()
        if count == 0:
            raise RuntimeError(f"No matching SAR scenes in {month_start:%B %Y} on orbit {orbit}.")
        if not path.exists() or path.stat().st_size == 0:
            image = scenes.select(["VV", "VH"]).median().toFloat().clip(area)
            url = image.getDownloadURL({"region": area, "scale": 10, "crs": epsg, "format": "GEO_TIFF"})
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            temp = path.with_suffix(".part")
            temp.write_bytes(response.content)
            temp.replace(path)
        with rasterio.open(path) as src:
            if src.count != 2:
                raise ValueError(f"Expected 2 SAR bands in {path.name}")
            current_grid = (src.shape, src.transform, src.crs)
            if grid is None:
                grid, profile = current_grid, src.profile.copy()
            elif current_grid != grid:
                raise ValueError(f"Raster grid mismatch for {tag}; cannot compare monthly pixels.")
            sar = src.read([1, 2])
        rasters.append(sar)
        notify(f"SAR {i + 1}/12: {month_start:%B %Y}, {count} scenes")

    valid_common = np.logical_and.reduce([np.isfinite(a).all(axis=0) for a in rasters])
    if not valid_common.any():
        raise RuntimeError("No pixels valid across all 12 months.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    notify(f"U-Net v3 loaded on {device}; common valid pixels: {valid_common.sum():,}")

    maps = []
    stats = []
    for i, sar in enumerate(rasters):
        month_start = dates[i]
        tag = month_start.strftime("%Y_%m")
        pred, _ = infer(model, device, sar)
        pred[~valid_common] = 255
        save_tif(pred, folder / f"LULC_{tag}.tif", profile)
        save_png(pred, folder / f"LULC_{tag}.png", f"SARTrack — {month_start:%B %Y}")
        maps.append(pred)
        stats.append([100 * np.count_nonzero(pred == j) / int(valid_common.sum()) for j in range(5)])
        notify(f"Prediction {i + 1}/12 complete: {month_start:%B %Y}")

    stats = np.asarray(stats)
    with (folder / "monthly_land_cover_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Month"] + [f"{c}_pct" for c in CLASSES]
                        + [f"{c}_monthly_change_pp" for c in CLASSES]
                        + [f"{c}_since_start_pp" for c in CLASSES])
        for i in range(12):
            delta = stats[i] - stats[i - 1] if i else np.zeros(5)
            from_first = stats[i] - stats[0]
            writer.writerow([dates[i].strftime("%B %Y")]
                            + [f"{v:.4f}" for v in np.r_[stats[i], delta, from_first]])

    stack = np.stack(maps)
    change = np.full(valid_common.shape, 255, dtype=np.uint8)
    change[valid_common] = np.count_nonzero(stack[1:] != stack[:-1], axis=0)[valid_common]
    stable = np.full(valid_common.shape, 255, dtype=np.uint8)
    stable_pixels = valid_common & (change == 0)
    stable[stable_pixels] = stack[0][stable_pixels]
    stability_pct = 100 * stable_pixels.sum() / valid_common.sum()
    save_tif(change, folder / "change_count.tif", profile)
    save_tif(stable, folder / "stable_class.tif", profile)
    save_png(stable, folder / "stable_class.png", "Stable predicted land cover")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.ma.masked_where(change == 255, change), cmap="YlOrRd", vmin=0, vmax=11)
    ax.axis("off")
    ax.set_title("Monthly prediction changes")
    fig.colorbar(im, ax=ax, label="Class changes (0–11)")
    fig.tight_layout()
    fig.savefig(folder / "change_count.png", dpi=150)
    plt.close(fig)
    (folder / "stability_summary.txt").write_text(
        f"Valid pixels: {int(valid_common.sum())}\nStable pixels: {int(stable_pixels.sum())}\n"
        f"Overall stability: {stability_pct:.2f}%\n", encoding="utf-8")
    notify(f"SUCCESS: 12 maps, CSV and stability outputs saved to {folder}")
    return folder