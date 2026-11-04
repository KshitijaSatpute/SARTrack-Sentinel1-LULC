# SARTrack — ESA WorldCover external-reference case study

**Case study:** Esch/Belval, Luxembourg. **Reference:** ESA WorldCover 2021 v200. **Prediction:** SARTrack U-Net v3, Sentinel-1 SAR, June 2025.

> This is a **2021-vs-2025 cross-year agreement/sanity check**, **not independent 2025 ground-truth accuracy**. WorldCover itself is a model-produced land-cover map, and land cover may have changed over the four-year interval.

## Method

1. Export WorldCover `ESA/WorldCover/v200` `Map` band using Google Earth Engine.
2. Align categorical WorldCover pixels exactly to the existing Esch/Belval test grid. The validated files have **521 rows × 519 columns**, **10 m** pixels, CRS `EPSG:32632`, and the identical affine transform (origin X=277810, Y=5490070; 10 m X and -10 m Y).
3. Convert the five comparable WorldCover codes to SARTrack labels:

   | ESA WorldCover code | WorldCover class | SARTrack code |
   |---|---|---|
   | 80 | Permanent water bodies | 0 — Water |
   | 10 | Tree cover | 1 — Trees |
   | 30 | Grassland | 2 — Grass |
   | 40 | Cropland | 3 — Crops |
   | 50 | Built-up | 4 — Built-up |

4. Evaluate only pixels valid in **both** the v3 prediction and WorldCover mapped reference. Exclude nonmatching categories such as WorldCover Bare/Sparse (`60`) and NoData.
5. Compute an aggregate five-class confusion matrix, observed agreement, mean class IoU, macro F1, and class-wise IoU/F1. Produce a three-panel comparison visualization with a common mask.

**Implementation:**
- [Comparison/scoring script](../SARTrack_WorldCover_Case_Study.py)
- [Report-ready figure script](../SARTrack_WorldCover_Report_Figure.py)

Both scripts were run on the project machine; the user-provided terminal output confirmed exact grid alignment and completed the comparison. Figure rendering was visually inspected.

## Observed results

| Cross-year metric | Value |
|---|---:|
| Compared pixels | 248,273 / 270,399 |
| Comparable fraction | 91.82% |
| Overall pixel agreement | **60.86%** |
| Mean IoU | **30.70%** |
| Macro F1 | **44.18%** |

| Class | IoU | F1 | WorldCover comparable pixels |
|---|---:|---:|---:|
| Water | 10.36% | 18.78% | 1,659 |
| Trees | 45.44% | 62.49% | 96,861 |
| Grass | 18.98% | 31.90% | 42,930 |
| Crops | 20.70% | 34.29% | 14,329 |
| Built-up | 58.03% | 73.44% | 92,494 |

The comparison map shows substantial spatial disagreement, particularly in areas where v3 predicts large contiguous Built-up regions while WorldCover maps a more varied land-cover mosaic. **The figure alone cannot establish why these differences occur.** Candidate explanations include model bias, spatial over-smoothing, sensor and taxonomy differences, and genuine 2021–2025 changes.

## Distinguish the two test references

- **Same-year June 2025 independent Esch test vs Dynamic World pseudo-labels:** v3 overall accuracy/agreement 83.71%, mIoU 46.37%, Macro F1 59.33%.
- **Cross-year Esch test vs ESA WorldCover 2021:** v3 pixel agreement 60.86%, mIoU 30.70%, Macro F1 44.18%.

These are **not directly interchangeable measures**. Neither comparison uses manually surveyed ground-truth labels.

## Reproduction with local project assets

Required files:

```text
data/external/Esch_WorldCover_2021_Aligned_to_SAR.tif
results_v3/esch_independent_comparison/esch_prediction_v3.tif
SARTrack_WorldCover_Case_Study.py
SARTrack_WorldCover_Report_Figure.py
```

Run from project root (Python environment with NumPy, rasterio, matplotlib):

```bash
python SARTrack_WorldCover_Case_Study.py
python SARTrack_WorldCover_Report_Figure.py
```

Expected output folder: `results_v3/worldcover_case_study/`, containing a summary CSV, class-wise CSV, confusion matrix CSV, provenance JSON, initial comparison figure, and improved report figure `worldcover_vs_v3_report.png`.

The reference GeoTIFF, prediction GeoTIFF and generated result binaries were packaged in the local `SARTrack_WorldCover_Case_Study_Package.zip` archive (~0.44 MB reported in the terminal). **This archive and those binaries are not uploaded to the GitHub repository in this commit.** Obtain the locally saved artifacts separately to reproduce the numerical result.

## Limits and pending work

- Four-year temporal mismatch; possible real land-cover changes.
- WorldCover / SARTrack five-class taxonomy cannot represent every WorldCover class.
- WorldCover uses multisensor classification, and is not verified field reference.
- Water has relatively low IoU; there are large Built-up prediction regions that warrant further investigation.
- Cross-year agreement cannot be used to claim verified land-use change or 2025 model accuracy.
- Final report integration, v3 dashboard integration and end-to-end reproduction audit are pending.
