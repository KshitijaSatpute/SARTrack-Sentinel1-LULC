# SARTrack v3 — corrected-data training and independent test

**Status (4 November 2026):** v3 training and the Esch/Belval comparison have been run. v3 is the preferred model candidate, but the Streamlit application in the repository still defaults to **v2**. The external ESA WorldCover sanity check, final report, and final submission QA are pending.

## Scope and data

- **Task:** Five-class pixel-wise segmentation from Sentinel-1 SAR VV/VH imagery: Water, Trees, Grass, Crops, Built-up.
- **Reference labels:** Google Dynamic World optical-derived pseudo-labels, mapped from raw codes `0,1,2,4,6` to model codes `0,1,2,3,4`. Unsupported labels use `255` (ignore). These are **not manually verified ground truth**.
- **Train geography:** Dudelange and Bastogne; **validation geography:** Thionville; **independent spatial test:** Esch/Belval, Luxembourg.
- **Corrected dataset:** 294 training and 196 validation 128×128 patches. June 2025 Dudelange and Thionville labels were re-exported with Dynamic World confidence filtering at 0.50. The original v2 data/models were retained separately; full consistency of the March export confidence thresholds has not been independently established.
- **Input normalization:** VV dB clipped to [-30, 5], VH dB clipped to [-35, 0], each scaled into [0, 1].
- **Architecture:** Lightweight U-Net, 2 input channels, 5 output classes, 488,629 trainable parameters.

## v3 training

- **Script:** [`SARTrack_v3_DataLab_Train.py`](../SARTrack_v3_DataLab_Train.py).
- **Environment used:** Data Lab NVIDIA A6000 (~47.4 GiB available GPU memory), PyTorch 2.9.1+cu128.
- **Optimizer:** Adam, learning rate 0.001; batch size 4; seed 42.
- **Loss:** Class-weighted cross-entropy, ignore index 255. Weights calculated from training labels only.
- **Sampling:** Water-aware weighted patch sampling with replacement.
- **Augmentation:** Random multiples of 90° rotation and horizontal/vertical flips, training only.
- **Training policy:** Maximum 20 epochs; early stopping after five epochs without validation mIoU improvement; best checkpoint chosen using validation mIoU with validation-loss tie-break.
- **Observed training:** 20 epochs, **best checkpoint at epoch 15**, best validation mIoU **61.31%**, validation overall accuracy **92.52%**, validation macro F1 **73.66%** (best-epoch values reported from the completed session).

Training outputs were stored in the Data Lab `vault/SARTrack_v3/results_v3/`, including `best_unet_v3.pth`, `history_v3.csv` and checkpoint metadata. These artifacts are backed up locally but **are not claimed to be downloadable from this repository**.

## Independent Esch/Belval test (June 2025)

Both models were evaluated on **the same spatial holdout, same reference mask, same normalization and same 128×128 tiled inference protocol**.

- **Script:** [`SARTrack_Esch_Compare_V2_V3.py`](../SARTrack_Esch_Compare_V2_V3.py).
- **Reference:** Dynamic World pseudo-labels embedded in the three-band test GeoTIFF; not a human-annotated or independent field reference.
- **Scoring:** Valid reference pixels only; 5×5 confusion matrices; overall accuracy, mean IoU and macro F1. The script saves classwise metrics, prediction GeoTIFFs and provenance information when run with the required local assets.

| Metric | U-Net v2 | U-Net v3 |
|---|---:|---:|
| Overall accuracy | 79.90% | **83.71%** |
| Mean IoU | 38.72% | **46.37%** |
| Macro F1 | 49.45% | **59.33%** |
| Water IoU | 7.04% | **18.83%** |
| Trees IoU | 64.35% | **65.88%** |
| Grass IoU | **25.54%** | 24.39% |
| Crops IoU | 14.02% | **36.88%** |
| Built-up IoU | 82.63% | **85.89%** |

Using displayed rounded percentages, v3 improves test mIoU by approximately **7.65 percentage points**; four out of five class IoUs improve, while Grass declines slightly.

## Reproduction (Data Lab / compatible PyTorch environment)

Put the two scripts **in the same working project directory** as the following inputs:

```text
SARTrack_v3_DataLab_Train.py
SARTrack_Esch_Compare_V2_V3.py
SARTrack_processed_v2.zip
processed_v2/
  train/   # 294 NPZ files
  val/     # 196 NPZ files
independent_test/
  Esch_Belval_June2025_Test.tif
  best_unet_v2.pth
  best_unet_v3.pth
```

Run training when a CUDA GPU is available:

```bash
python SARTrack_v3_DataLab_Train.py
```

Or, for evaluation with existing v2/v3 checkpoints:

```bash
python SARTrack_Esch_Compare_V2_V3.py
```

Dependencies used by the scripts include PyTorch, NumPy, rasterio and (for visualization) matplotlib. The existing repository `requirements.txt` documents a *different*, Windows-oriented environment; reproducibility of this exact Data Lab setup on a fresh machine remains to be validated.

## Interpretation and limitations

1. **Pseudo-label agreement, not real-world ground-truth accuracy:** The reference comes from Dynamic World (optical-derived).
2. **Not a controlled ablation:** v3 changes corrected June labels, augmentation and training duration together; performance gains cannot be attributed to one change.
3. **Weak minority classes:** Water IoU remains low (18.83%). Grass IoU slightly decreases.
4. **Spatial and temporal generalization:** A single independent Esch test does not demonstrate global transferability; v3 has not been demonstrated over all possible months/orbits.
5. **Prediction changes are not verified land-use changes:** Model flicker, seasonal SAR signatures and pseudo-label noise can create apparent transitions.
6. **Validation patches may overlap:** Patch-level validation statistics do not imply independent pixel samples.
7. **External validation pending:** ESA WorldCover 2021 vs SAR 2025 can only be treated as a temporal-mismatch sanity check, not definitive 2025 ground truth.

## Current next tasks

- Export WorldCover reference aligned with the Esch grid; document the 2021–2025 mismatch.
- Integrate the selected checkpoint into Streamlit **without replacing the original v2 asset until smoke tests pass**.
- Package model weights and data with validated links; archive detailed v3 results.
- Write the final technical report and demo, and complete submission QA.
