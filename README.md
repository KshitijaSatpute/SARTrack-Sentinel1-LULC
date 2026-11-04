# SARTrack — Sentinel-1 Land-Cover Classification and Stability Monitoring

Master's deep-learning prototype that generates **12 monthly, five-class land-cover predictions** from Sentinel-1 SAR (VV/VH) using a lightweight U-Net v2, then reports predicted class-area percentages, percentage-point changes and prediction stability.

> **Research prototype, not a verified land-use change product.** Model evaluation uses Dynamic World pseudo-labels, not an independent manually annotated reference dataset.

## What the app does

1. Accepts a latitude, longitude and starting month.
2. Uses Google Earth Engine (GEE) to obtain 12 monthly Sentinel-1 GRD VV/VH median composites for an approximately 5 km buffer around the point (using the buffered bounding box).
3. Runs the saved five-class U-Net v2 on each month's SAR composite.
4. Produces 12 georeferenced GeoTIFF prediction maps and PNG previews.
5. Computes the predicted area share of each class, monthly and start-to-end changes (percentage points), and the number of month-to-month class changes per pixel.
6. Displays trends, maps, stability summaries and downloadable files in Streamlit.

**Classes and raster codes:** `0=Water`, `1=Trees`, `2=Grass`, `3=Crops`, `4=Built-up`; `255=NoData`.

## Requirements

- Python **3.12** (version used during local testing).
- Google Earth Engine account with access to Sentinel-1 and a configured Google Cloud project.
- Internet access while fetching imagery.
- Saved weights at `models/best_unet_v2.pth` (required and **not** automatically downloaded).
- Windows PowerShell instructions are below. CUDA is optional for inference; CPU execution is supported by the code but has not been benchmarked here.

### 1. Install dependencies

From the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided `requirements.txt` specifies the tested **CUDA 12.8 PyTorch build** using the PyTorch wheel index. On machines where this wheel is unsupported, install a compatible PyTorch build from the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and adjust the PyTorch requirement as appropriate. A fresh-machine installation of the supplied requirements has **not** yet been validated.

If PowerShell blocks virtual-environment activation, use the Python interpreter inside `.venv` directly or adjust your shell's execution policy in accordance with your machine's security policy.

### 2. Authenticate Google Earth Engine

```powershell
earthengine authenticate --auth_mode=localhost
```

The code currently sets:

```python
PROJECT_ID = "sartrack-lulc-2026"
```

in `sartrack_pipeline.py`. **Replace this value with your own Earth Engine-enabled Google Cloud project ID** if running under a different account. Authenticate with an account permitted to use that project. A Google Cloud project ID is not a credential; OAuth authentication still needs to be completed.

### 3. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Enter a point and starting month, then click **Generate 12 Monthly Maps**. You must choose 12 **complete past calendar months**; a year-crossing range is supported. The *day* portion of the date selector is ignored.

For the precomputed Esch/Belval example only (49.5000° N, 5.9670° E, January 2025), **Open saved Esch demo (instant)** reads the saved demo outputs without invoking GEE.

### 4. Optional: run the pipeline without Streamlit

```powershell
python -c "from sartrack_pipeline import run_pipeline; print(run_pipeline(49.5000, 5.9670, 2025, 1))"
```

For an alternative area and range, replace the arguments `latitude, longitude, start_year, start_month`.

## Output structure

Each live run writes to a location- and date-specific folder:

```text
outputs/
  live/
    lat+49.5000_lon+5.9670_2025_01/
      SAR_2025_01.tif ... SAR_2025_12.tif
      LULC_2025_01.tif ... LULC_2025_12.tif
      LULC_2025_01.png ... LULC_2025_12.png
      monthly_land_cover_statistics.csv
      change_count.tif
      change_count.png
      stable_class.tif
      stable_class.png
      stability_summary.txt
```

The `monthly_land_cover_statistics.csv` contains `Month`, five `<class>_pct` fields, five `<class>_monthly_change_pp` fields and five `<class>_since_start_pp` fields. All percentages use pixels valid in **all 12** months as the denominator. Changes are **percentage points**, not percentage growth rates.

`change_count.tif` stores how many times each pixel's **predicted class** changed between 11 adjacent-month pairs (0–11). `stable_class.tif` retains the class only for pixels with zero such changes; other pixels are encoded as NoData (`255`). Stability percentage is the fraction of commonly valid pixels with zero predicted changes.

## Preprocessing and method

- GEE source: `COPERNICUS/S1_GRD`, Interferometric Wide (IW), dual polarization `VV` and `VH`, **ascending pass**.
- Chooses the most frequent available ascending **relative orbit** over the requested 12-month interval and applies that orbit consistently. A run **fails** if any selected month has no matching images.
- Monthly composite: pixelwise median of available VV/VH backscatter images.
- GEE S1 GRD provides calibrated, terrain-corrected backscatter in dB; this pipeline **does not perform explicit speckle filtering or radiometric terrain flattening**. Do not claim those were added.
- Region: a 2,500 m geodesic buffer around the selected point, exported as its bounding box at nominal 10 m sampling. Output uses an appropriate local UTM zone chosen from longitude/latitude.
- Model input: two channels (VV/VH), clipped and min–max normalized using fixed dB bounds: **VV [−30, 5]**, **VH [−35, 0]**.
- Inference: 128 × 128 non-overlapping tiles, including zero padding at edges; saves one prediction per commonly valid pixel.
- Training reference: Google Dynamic World pseudo-labels were used to train/evaluate the segmentation network, **not as inference inputs**. Dynamic World is not downloaded for each new app run.

## Model and evaluation

- Lightweight 2-input, 5-output U-Net v2, **488,629 parameters**.
- Training areas: **Dudelange (Luxembourg)** and **Bastogne (Belgium)**.
- Validation area: **Thionville (France)**.
- Independent holdout used for reported metrics: **Esch/Belval (Luxembourg)**, evaluated against Dynamic World pseudo-labels.
- Independent holdout metrics (saved project results): **overall accuracy 79.90%**, **mIoU 38.72%**, **macro F1 49.45%**.

These values do **not** establish performance for arbitrary new coordinates. Water and crops are relatively weak classes; domain shift, seasonal effects and noisy pseudo-labels can affect outcomes. A high stability percentage indicates model-prediction agreement, **not proven real-world land-cover stability**.

## Main project files

```text
app.py                         Streamlit interface
sartrack_pipeline.py           GEE retrieval, inference, statistics and stability
models/best_unet_v2.pth        Trained weights required for live inference
notebooks/01_Data_Preprocessing.ipynb
notebooks/02_UNet_Training.ipynb
notebooks/03_Independent_Test.ipynb
requirements.txt               Python environment dependencies
data/raw/                       Original training/reference GeoTIFFs (if included)
outputs/                        Demo and generated results
```

The notebooks contain the development, training and evaluation workflow. Run them with a Python environment containing the dependencies needed by their individual cells. Source data and saved model weights must be supplied alongside the notebook(s); the repository alone does not reconstruct third-party Earth Engine permissions. Large datasets may be provided separately if submission size limits require it; include a working dataset link in the final submission materials.

## Known limitations and reproducibility

- Geographic input is flexible, but **Sentinel-1 coverage and a consistent ascending relative orbit are required for all 12 months**. The pipeline may fail for some locations or date ranges.
- A 12-month run uses multiple GEE requests; execution depends on network connectivity, GEE access and service quotas.
- Model predictions can show month-to-month *classification flicker* unrelated to real land-cover change.
- The system does not currently quantify pixel-level predictive uncertainty or validate outputs against manually labelled target-area samples.
- Precomputed Esch demo and newly generated Esch results can differ due to projection and processing choices; do not treat the percentages as interchangeable.
- The original project was tested in a local Windows Python 3.12 environment with an NVIDIA RTX 4050 GPU. Portability to other machines, fresh installation and notebook reruns still require independent validation.

## Academic provenance

Source imagery: ESA Sentinel-1, accessed through Google Earth Engine (`COPERNICUS/S1_GRD`). Pseudo-label source for development: Google Dynamic World (`GOOGLE/DYNAMICWORLD/V1`). Cite the original data products and relevant scientific papers in the final technical report.

**Status:** Functionality tested for Esch/Belval and Luxembourg City, including a July 2025–June 2026 cross-year run. This README documents the observed prototype, not a claim of global validation.

## Research update — U-Net v3 (4 November 2026)

The original Streamlit app and main pipeline in this repository **still use U-Net v2 by default**. A separate v3 model was trained on corrected June pseudo-label exports and compared against v2 on the geographically held-out Esch/Belval test raster.

| Held-out Esch/Belval metric | U-Net v2 | U-Net v3 |
|---|---:|---:|
| Overall accuracy | 79.90% | **83.71%** |
| Mean IoU | 38.72% | **46.37%** |
| Macro F1 | 49.45% | **59.33%** |

**Interpretation:** v3 improves agreement with Dynamic World pseudo-labels on this holdout. The results are not accuracy measured against manually verified ground truth. Water remains challenging (v3 water IoU 18.83%). The experiments are not a controlled single-variable ablation, and the final model has not yet been integrated into the dashboard.

New verified experiment files:

- [v3 Data Lab training script](SARTrack_v3_DataLab_Train.py) — two-channel U-Net, training configuration, validation, early stopping and checkpoint saving.
- [Independent v2-v3 comparison script](SARTrack_Esch_Compare_V2_V3.py) — matched preprocessing, tiled inference, confusion matrices, classwise scores and GeoTIFF outputs.
- [Detailed v3 experiment record](docs/v3_experiment_results.md) — dataset corrections, hyperparameters, per-class IoUs, scientific limitations and reproduction guidance.

**Asset availability:** The corrected NPZ dataset, model checkpoint binaries, training history CSV, and full evaluation ZIP were backed up separately during experimentation. Their presence in this repository is **not guaranteed**; do not assume a GitHub clone alone reproduces reported results.

**Pending:** ESA WorldCover external sanity-check case study (2021 reference vs 2025 SAR), v3 dashboard integration and smoke tests, final report, demo video, dataset/model delivery, and submission QA.

## External-reference case study — WorldCover 2021 vs SARTrack v3 (June 2025)

The Esch/Belval v3 prediction has also been compared with an aligned ESA WorldCover 2021 categorical raster, using five comparable land-cover classes and a common valid-pixel mask (248,273 of 270,399 pixels, 91.82%).

| Cross-year agreement statistic | Result |
|---|---:|
| Overall pixel agreement | **60.86%** |
| Mean IoU | **30.70%** |
| Macro F1 | **44.18%** |

**Do not interpret these scores as independently verified 2025 accuracy.** The reference is from 2021, SAR prediction from 2025, and WorldCover is also a model-generated land-cover map. Class taxonomy and real land-cover changes can contribute to disagreements.

- [WorldCover case-study methodology, results, caveats](docs/worldcover_case_study.md)
- [Cross-year comparison and confusion-matrix script](SARTrack_WorldCover_Case_Study.py)
- [Report-ready visualization script](SARTrack_WorldCover_Report_Figure.py)

The data/reference rasters, comparison PNG, metric CSVs and packaged results are **local project artifacts** and were **not uploaded with this README**. The dashboard still uses **v2** unless subsequently integrated and verified.
