# SARTrack — Master Project Handoff + Data/Label Verification Audit

**Handoff date:** 31 October 2026 (Europe/Berlin)  
**Project:** SARTrack — Sentinel-1 Land-Cover Classification and Temporal Stability Monitoring  
**Target:** Master's Deep Learning individual project  
**Stated deadline:** Monday **9 November 2026 at 23:59**, Moodle — verify against the live course page before submitting.  
**Next working step:** **Step 23 — Notebook and preprocessing/label QA**, not redoing the already completed Streamlit tests.  
**Purpose:** Self-contained context for a new chat, plus evidence, uncertainties, and actions needed to validate the dataset and labels. Treat this as an audit record, **not** certification of ground-truth accuracy.

## Executive verdict (read this first)

- **The end-to-end application is operational for the tested areas/dates.** GEE SAR fetching, 12-month U-Net inference, georeferenced maps, area statistics, stability maps, download, a second location, and a cross-year interval were shown working in terminal/browser screenshots.
- **The official Dynamic World class code meanings are correct**, and all 11 uploaded three-band source GeoTIFFs have the expected `VV`, `VH`, `label` bands. The actual source labels are in Dynamic World's 0–8 range plus custom `-1` and outside-area NaN.
- **Important outstanding label-audit issue:** two June GeoTIFFs contain **zero `-1`/confidence-masked pixels**, whereas most other monthly exports contain substantial `-1` areas, reaching **46%**. This is not automatically an error, but it could mean confidence filtering was inconsistent or source scripts changed. **Investigate before stating labels are uniformly confidence-filtered.**
- The exact historical **confidence-threshold (`>=0.50`), temporal-mode composite and raw-to-five-class remapping implementation are *not independently verifiable*** from the currently provided artifacts because `01_Data_Preprocessing.ipynb`, the original GEE export scripts and processed `.npz` patches were not supplied for this audit.
- U-Net's label semantics look internally consistent with the *intended* five-class mapping, but no manually annotated independent ground truth has been provided. All reported metrics are against **Dynamic World pseudo-labels**, not verified land-cover truth.
- **Do not submit yet:** notebooks, confidence masking, statistics reproducibility, dataset link, final report, video, and package integrity need an explicit final pass.

## Evidence and assurance levels

**Directly checked here:** 11 original 3-band raw GeoTIFFs; the cross-year `LULC_2026_06.tif`; the uploaded executed `02_UNet_Training.ipynb`; `sartrack_pipeline_fixed.py` source (full 231 lines); uploaded `app_updated.py` source; `README.md`; 99-package `requirements.txt` and 8-direct-dependency cleaned file. Official Earth Engine documentation was cross-checked for Dynamic World class codes and Sentinel-1 GRD preprocessing.

**Demonstrated in previous screenshots / terminal, but not rerun in this audit environment:** GEE live calls, saved model weight loading, complete 12-month folders, Esch holdout metrics, Streamlit tabs, city/starting-month cases, notebook `01` and `03` outputs. The screenshots and execution logs are evidence of successful prior runs, **not** new reruns here.

**Not available for independent inspection:** `01_Data_Preprocessing.ipynb`, `03_Independent_Test.ipynb`, `data/processed/train/*.npz` and `val/*.npz`, `models/best_unet_v2.pth`, raw GEE JS export scripts, full complete output folders, manually labeled reference samples, dataset-hosting URL, final report or demo recording. Avoid inferring a clean bill of health for these.

**File-version caveat:** uploaded `app.py` is the *earlier 528-line precomputed-demo version*; the user's successful current local live app matches the uploaded `app_updated.py` (approximately 422 lines) renamed to `app.py`. `sartrack_pipeline_fixed.py` is the corrected version that produced successful runs; the user renamed it to `sartrack_pipeline.py`. Never accidentally replace the working local app with the old attached `app.py`.

## 1. Original objective and professor-facing requirements

**Problem:** Use Sentinel-1 C-band SAR to infer land-cover classes over a user-specified geographic region across **12 consecutive complete calendar months**, then quantify month-wise class area and classification stability.

**Input to the neural network:** *only* Sentinel-1 VV and VH SAR backscatter. **Dynamic World is training/evaluation pseudo-labels**, not an additional model input. Using optical-derived labels for SAR-supervised training is not the same as feeding optical imagery into U-Net.

**Expected project artifacts and behaviors (course/project requirements as conveyed during our work; compare with official assignment):**

1. User enters latitude, longitude and a starting month; the application fetches matching monthly SAR and produces **12** georeferenced five-class predictions.
2. Show each month on its own, class-area percentages, adjacent-month percentage-point differences, and start-to-end differences; optionally report per-pixel stability.
3. Explain data collection, preprocessing, radar calibration/dB/terrain correction, normalization, masking, tiling, class distribution, data splits, augmentation (if any), architecture, training and limitations.
4. Evaluate U-Net on geographic validation and independent holdout with overall accuracy (OA), mIoU, classwise IoU and F1; show actual saved training curves and representative prediction vs. labels.
5. Submit working code, training weights, executed notebooks **with outputs**, dataset or working dataset link, `README.md`, `requirements.txt`, detailed technical report, and demonstrative screen recording.
6. Make defensible uncertainty statements: pseudo-label evaluation, weaknesses in rare classes and geographically novel locations, predicted temporal flicker versus verified physical land-use change.

**Project prioritization decisions:** Functional correctness and submission obligations first; cosmetic redesign only when compulsory tasks are complete. Avoid repeating instructions already finished. Give one actionable step at a time; checkpoint remaining tasks every 5–10 steps. User prefers concise Marathi/Hinglish during implementation, full replacement files rather than piecemeal edits, and strict honest feedback.

## 2. End-to-end chronology (ordered milestones, exact calendar dates not all established)

1. **GEE setup:** Connected Earth Engine via `earthengine authenticate --auth_mode=localhost` and initialized with Google Cloud project `sartrack-lulc-2026`. Tested data access in GEE Code Editor and local Python.
2. **Region exploration:** Dudelange, Bastogne, Thionville and Esch/Belval tested; selected geographically separate sources. GEE scripts reportedly included `01_Luxembourg_Sentinel1_Test`, `02_Dudelange_Data_Check`, `03_Thionville_Data_Check`, `04_MultiRegion_Dataset_Export`, `05_March2025_Data_Check`, `06_September2025_Data_Check`, `07_Bastogne_September2025_Export`, `09_Esch_Belval_Test_Export` (full JS not in audit).
3. **Data preparation:** Exported 10 multi-month/monthly raw GeoTIFFs for training+validation and one independent Esch holdout GeoTIFF. Five target classes selected, ambiguous/unused labels ignored. Created 128×128 NPZ patches with stride 64 (counts reported below).
4. **Model development:** Created lightweight two-channel five-output U-Net (~488,629 trainable parameters). Trained baseline v1 and v2 with water-aware oversampling; each had 3 training epochs. Saved weights `models/best_unet.pth`, `models/best_unet_v2.pth`.
5. **Evaluation:** Validation on Thionville, independent Esch/Belval June pseudo-label test; generated training and evaluation plots in notebooks. Weak water and crops segmentation identified, especially holdout water.
6. **One-month inference:** Built `gee_fetch.py` and `predict_month.py`; checked June 2025 Esch SAR band/grid alignment and produced a LULC GeoTIFF/PNG.
7. **12-month inference:** `gee_fetch_all_months.py` + `predict_all_months.py` created Jan–Dec 2025 Esch maps and statistics; `stability_map.py` produced change-count and stable-class maps.
8. **Dashboard v1:** `app.py` initially displayed precomputed Esch data, with Overview, Monthly Maps, Stability, Downloads and Model Details tabs. Pandas/scikit-learn DLL execution policy problems led to a pure `csv`/NumPy/Matplotlib Streamlit implementation; tabs and downloads verified.
9. **Parameterized backend:** `sartrack_pipeline.py` introduced live GEE download, per-month inference, statistics, stability, and output folders. Fixed a truncated function (`Output folder: None`) and `int('88.0')` orbit-key error via `int(float(...))`.
10. **Successful local live run:** Esch 49.5000, 5.9670, Jan–Dec 2025: 12 SAR months, U-Net v2 loaded on CUDA, 12 predictions, CSV and stability images; output `outputs/live/lat+49.5000_lon+5.9670_2025_01/`.
11. **Streamlit live integration:** Replaced current local `app.py` with the new UI; browser showed live 12/12, correct month switching, stability, downloads. Old `app_backup.py` retained.
12. **Different location test:** Luxembourg City 49.6116, 6.1319, Jan–Dec 2025 ran successfully with distinct statistics/maps.
13. **Cross-year test:** Luxembourg City, **July 2025–June 2026** ran successfully with correct month labels, statistics, maps, and stability. `LULC_2026_06.tif` downloaded and later inspected.
14. **Documentation:** 99-package `pip freeze` generated, replaced locally with clean eight-direct-dependency `requirements.txt`; `README.md` added to the project and saved.
15. **Current step:** Step **23** — inspect and cross-verify notebooks, preprocessing label generation, split/NPZ contents, and final deliverables. Last completed step **22** (README saved).

## 3. Key paths, exact environment, and project structure

**User Windows project root:** `C:\Users\admin\Downloads\Deep learning\Project`  
**Virtual environment:** `.venv`, Python 3.12, tested on NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).  
**App launch:** `python -m streamlit run app.py`, browser normally `http://localhost:8501`.  
**GEE:** `earthengine authenticate --auth_mode=localhost`; `ee.Initialize(project="sartrack-lulc-2026")` confirmed in previous local tests. Credentials are machine/account-specific; do not put tokens in submitted zip.

```text
Project/
  app.py                          CURRENT WORKING LIVE app (do not overwrite with old uploaded app.py)
  app_backup.py                   earlier app backup
  sartrack_pipeline.py           working corrected live inference backend
  gee_fetch.py, gee_fetch_all_months.py
  predict_month.py, predict_all_months.py, stability_map.py
  requirements.txt               cleaned eight-direct-dependency version
  README.md
  models/
    best_unet.pth                 baseline
    best_unet_v2.pth              main model (required)
    [saved training/prediction plots if locally present]
  notebooks/
    01_Data_Preprocessing.ipynb   executed according to prior screenshots; needs audit
    02_UNet_Training.ipynb        executed and uploaded: audited below
    03_Independent_Test.ipynb    executed according to prior screenshots; needs audit
  data/
    raw/                          10 training+val files, 1 independent Esch file
    processed/train/              294 saved NPZ patches (reported)
    processed/val/                196 saved NPZ patches (reported)
    monthly/                      Esch cached Jan–Dec 2025 SAR
  outputs/
    monthly/                      precomputed Esch 2025 12 GeoTIFF + PNG + statistics CSV
    stability/                    precomputed Esch stability maps/summary
    live/
      lat+49.5000_lon+5.9670_2025_01/
      lat+49.6116_lon+6.1319_2025_01/
      lat+49.6116_lon+6.1319_2025_07/
```

**Versioned dependency file actually supplied for replacement:**

```text
--extra-index-url https://download.pytorch.org/whl/cu128
earthengine-api==1.7.50
matplotlib==3.11.2
numpy==2.5.2
rasterio==1.5.2
requests==2.34.2
streamlit==1.66.1
torch==2.11.0+cu128
ipykernel==7.4.0
```

App avoids importing pandas and scikit-learn owing to Windows Application Control/DLL blocks. **Notebook cells have not all been retested under this cleaned dependency set.** Original `pip freeze` had 99 packages, including pandas and scikit-learn. A fresh-machine install is not yet validated; CUDA-specific torch wheel may be unsuitable for CPU-only platforms.

## 4. Data sources, scene specifications and georeferencing

- **Inference input source:** `COPERNICUS/S1_GRD` in Earth Engine, Interferometric Wide (IW), dual-pol **VV/VH**, ascending-pass scenes, one consistent relative orbit per selected run, monthly **median** of available SAR images. The corrected live pipeline chooses the most frequent ascending relative orbit in the 12-month interval and fails if any month lacks scenes on that orbit. It does not enforce orbit 88 everywhere.
- **Initial Esch demonstration:** ascending relative orbit **88**; 2025 monthly Sentinel-1 scene counts in chronological order: **2, 3, 2, 3, 7, 7, 8, 7, 8, 9, 7, 8**. Terminal confirms same sequence for successful live Esch run.
- **Raw reference label source:** `GOOGLE/DYNAMICWORLD/V1` derived from **Sentinel-2** optical imagery and providing nine semantic classes (0–8) + class probabilities. Labels are *pseudo-labels* and may be temporally imperfect relative to SAR data.
- **Area:** `ee.Geometry.Point([longitude, latitude]).buffer(2500).bounds()` — the square bounding box of a **2.5 km radius buffer**, not exactly the circular ROI. At 10 m nominal spacing, raster dimensions around 518–523 × 518–521. Fractional coverage / clipping means useful pixels are fewer than total raster cells.
- **Static training GeoTIFFs:** 10 m resolution, EPSG:32632 (UTM zone 32N), 3 bands (`VV`,`VH`,`label`) and `float32`. Each geographic region uses the same width/height/geotransform and identical outside-area NaN mask in all available months; checked directly.
- **Live dynamic app:** EPSG local zone derived from longitude. Esch 5.9670°E yields **EPSG:32631**; static Esch exports were EPSG:32632. Both can be valid projections, but the grid is not identical and may explain demo-vs-live differences. Luxembourg City at 6.1319°E uses EPSG:32632.
- **Expected Sentinel-1 processing:** GEE `COPERNICUS/S1_GRD` provides orbit correction, noise removal, radiometric calibration, terrain correction and sigma-nought in **dB**. The code itself does **not** add an explicit speckle filter or radiometric terrain flattening: do not claim those were done.
- **Normalization:** VV dB clipped to **[-30, 5]**, VH to **[-35, 0]**, then min–max normalized to [0,1]. Values outside bounds are clipped, not inherently corrupt. Training-time preprocessing implementation must still be verified against `01_Data_Preprocessing.ipynb`.

Official reference: https://developers.google.com/earth-engine/guides/sentinel1 and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

## 5. Label mapping: exact semantics and conditions for correctness

**Critical distinction: source Dynamic World code 3 is _flooded vegetation_; target model code 3 is _crops_. Source code 4 is crops, and source code 6 is built. Therefore raw labels **must** be remapped before training.**

| Raw label value | Official Dynamic World meaning | Five-class target code | Expected action |
| --- | --- | --- | --- |
| 0 | Water | 0 Water | Keep |
| 1 | Trees | 1 Trees | Keep |
| 2 | Grass | 2 Grass | Keep |
| 3 | Flooded vegetation | 255 Ignore | Exclude; not crops |
| 4 | Crops | 3 Crops | Remap 4→3 |
| 5 | Shrub and scrub | 255 Ignore | Exclude |
| 6 | Built area | 4 Built-up | Remap 6→4 |
| 7 | Bare ground | 255 Ignore | Exclude |
| 8 | Snow and ice | 255 Ignore | Exclude |
| -1 | Not a native DW class | 255 Ignore | Custom invalid/confidence-masked indicator; verify provenance |
| NaN | Outside/masked raster area | 255 Ignore | Do not treat as a class |

**Official reference:** https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1 (nine-class code table). **Verdict:** the *intended mapping is semantically correct*; raw-file value domains are consistent with it. `-1` is not an official DW class: it appears to have been generated by the export/filter workflow. The training notebook shows processed masks having exactly **`[0,1,2,3,4,255]`**, which supports a remap/ignore step, **but does not prove each source class was mapped correctly**. For example, blindly keeping source label 3 would yield an incorrect crops class: inspect preprocessing code and recompute a handful of pixel labels to rule this out.

**The 0.50 confidence threshold, maximum of nine probability bands, monthly temporal `mode()`, DW cloud masking and choice of `-1` are historical implementation notes — not independently verified in the uploaded source.** Check source GEE JS and `01_Data_Preprocessing.ipynb`, and optionally reconstruct one month directly in GEE to compare sampled pixels with exported TIFF label values.

## 6. Direct raw-raster audit: all 11 supplied source GeoTIFFs

Raster-level checks **passed** for all 11: 3 bands named VV, VH and label; float32; 10.0 m square pixels; EPSG:32632; label values only native DW codes 0–8 plus optional `-1`/NaN; VV/VH/label have exactly coincident outside-area NaN masks. Within each location, months share the same grid and outside-area mask. These checks do **not** validate semantic correctness of labeling or matching SAR/optical acquisitions.

Legend: **5-class valid %** = pixels in raw `{0,1,2,4,6}` divided by pixels finite across all bands. **`-1` %** = custom rejected/raw-uncertain pixels divided by finite pixels. `Other DW` = source classes 3,5,7,8, which should be ignored in five-class training.

| Raster | Split | Width×Height | 5-class valid | `-1` | Other DW |
| --- | --- | --- | --- | --- | --- |
| Bastogne_June2025 | Training | 521x523 | 54.0% | 46.0% | 32 |
| Bastogne_March2025 | Training | 521x523 | 58.9% | 41.1% | 10 |
| Bastogne_September2025 | Training | 521x523 | 55.8% | 44.2% | 8 |
| Dudelange_June2025_Test | Training | 519x520 | 98.9% | 0.0% | 2689 |
| Dudelange_March2025 | Training | 519x520 | 69.1% | 30.9% | 106 |
| Dudelange_September2025 | Training | 519x520 | 68.7% | 31.1% | 387 |
| Esch_Belval_June2025_Test | Independent test | 519x521 | 72.5% | 26.2% | 3362 |
| Thionville_December2025 | Validation | 518x520 | 59.1% | 40.9% | 81 |
| Thionville_June2025 | Validation | 518x520 | 99.1% | 0.0% | 2361 |
| Thionville_March2025 | Validation | 518x520 | 84.9% | 15.1% | 32 |
| Thionville_September2025 | Validation | 518x520 | 76.8% | 23.2% | 0 |

### Most important red flag: possible inconsistent confidence filtering

- **Dudelange_June2025_Test.tif:** **0 `-1` out of 250,469** finite pixels; **98.9%** mapped to the five kept classes.
- **Thionville_June2025.tif:** **0 `-1` out of 250,443** finite pixels; **99.1%** mapped to the five kept classes.
- **Bastogne_June2025.tif:** **115,254 `-1` pixels (46.0%)**.
- Non-June counterparts for Dudelange have about **31% `-1`**, and Thionville counterparts have **15–41%**.

**Interpretation:** zero invalid-confidence pixels for two June exports might reflect an earlier export recipe without the confidence filter, different mask semantics, a permissive threshold, or another valid export difference; **we cannot determine which** from the TIFF alone. This is not proof of mislabeled pixels. It is a **high-priority consistency audit** before claiming 0.50 threshold applied uniformly to all images. Verify JS source and, if necessary, re-export the two June scenes using the final uniform labeling recipe. Re-extraction/retraining is conditional on what the audit finds, not automatically required.

`Dudelange_June2025_Test.tif` includes `Test` in its filename but belongs to the **training region** as currently described. Verify that the preprocessing notebook assigns it to training only and does not confuse the name with an independent holdout.

### Aggregate source-class distributions (all pixels from each group, NOT unique across months)

The counts below pool monthly raster pixels. They are descriptive **source-raster distributions**, not the actual sampled patch counts, and they should not be compared to class prevalence in the whole region without accounting for masks and overlapping periods.

| Split | GeoTIFFs | Kept class pixels | `-1` pixels | Water / Trees / Grass / Crops / Built-up |
| --- | --- | --- | --- | --- |
| Training | 6 | 1,015,532 | 484,344 | 0.66% / 38.23% / 15.74% / 9.94% / 35.43% |
| Validation | 4 | 801,072 | 198,226 | 6.76% / 17.46% / 2.51% / 5.56% / 67.72% |
| Independent test | 1 | 181,543 | 65,593 | 0.48% / 33.59% / 3.21% / 5.02% / 57.71% |

The training notebook's hardcoded five-class frequencies are `[0.0062, 0.3607, 0.1347, 0.0946, 0.4037]` (0.62% water, 36.07% trees, 13.47% grass, 9.46% crops, 40.37% built-up). They **differ** from pooled raw-raster frequencies, which is expected if training patches were filtered/sampled/overlapped. Verify these values were actually computed from the processed train NPZ mask pixels rather than copied from an earlier preprocessing run. The `02_UNet_Training.ipynb` labels them as "from Day 4," not computed in that notebook.

## 7. Processed patches, splits, and validation leakage checks

**Reported structure:** 128 × 128 patches, stride 64 (overlapping), normalized input `x` float32 shape `(2,128,128)`, mask `y` int64/uint8 shape `(128,128)`, 5 classes 0–4, `255` for ignore. Six Dudelange+Bastogne source GeoTIFFs supply training; four Thionville source GeoTIFFs supply validation; Esch/Belval June is separate holdout test.

**Executed training notebook confirms:** `294` training NPZ files, `196` validation NPZ files; batch image tensor `[4,2,128,128]`, mask `[4,128,128]`, unique mask values `[0,1,2,3,4,255]`.

**What still needs verification:**

1. Every processed NPZ preserves band order **VV then VH** and the exact normalization used by live inference (VV [-30,5], VH [-35,0]).
2. `y=3` must originate from **DW raw 4**, and `y=4` from **DW raw 6**; raw 3,5,7,8,-1,NaN become 255. Check code, and use sampled georeferenced pixels.
3. There is no **geographic train/validation overlap** or shared scene used for both (Dudelange/Bastogne vs Thionville should be segregated). Confirm actual NPZ path sets/file prefixes.
4. Stride 64 means adjacent **validation patches overlap by 50%** in each axis, so the validation confusion matrix counts some pixels multiple times. Report this as **patch-sampled/overlap-weighted validation**, not an unbiased regional pixel metric. The independent Esch holdout mitigates this, but still uses pseudo-labels.
5. Check the number of kept vs ignored pixels per patch, training subset selection, random seeds, empty-label patches and any filters for zero-label tiles.
6. **Augmentation:** no explicit geometric/photometric augmentation is visible in uploaded training notebook. Water-aware **sampling** is *not augmentation*. Do not claim implemented augmentations unless verified in preprocessing.
7. Inspect whether model training or validation used Esch pixels accidentally. **A file named `_Test` in Dudelange is not an independent test area** despite its filename.

## 8. Neural network and training — verified notebook outputs

**Source examined:** uploaded executed `02_UNet_Training.ipynb` (16 cells, all 16 have execution counts and outputs, including loss curves, confusion plots, class-IoU comparison, and SAR/label/prediction example).

- Architecture: `DoubleConv` blocks with channels 16,32,64, bottleneck 128; skip-connected U-Net decoder 64,32,16; 2 input bands; 5 output logits. **488,629 parameters** as printed in notebook.
- Loss: `CrossEntropyLoss(weight=weights, ignore_index=255)`, weights from inverse square root of the five hardcoded training class frequencies, rescaled and clamped; printed weights **[2.898, 0.380, 0.622, 0.742, 0.359]**.
- Optimizer: Adam, LR **0.001**. Batch size **4**. Each experiment trained **3 epochs** (proof-of-concept, not convergence claim).
- v1 baseline: regular shuffled train loader, save best by validation loss to `models/best_unet.pth`.
- v2: `WeightedRandomSampler` based on water-pixel fraction with replacement (weight 1–5 approx.), fresh model, save best val loss to `models/best_unet_v2.pth`. Exactly **100/294** training patches contained any water; **10/294** exceeded 5% water; maximum single-patch water fraction **12.6%**, none exceeded 20%.
- v2 training uses `torch.manual_seed(42)` before creating the model, but full end-to-end deterministic reproducibility has *not* been established.

| Experiment / split | OA | mIoU | Macro F1 | Evidence |
| --- | --- | --- | --- | --- |
| v1 Baseline | 85.47% | 42.81% | 56.58% | Executed notebook cell 6 |
| v2 Water-aware | 86.48% | 49.27% | 62.61% | Executed notebook cell 12 |
| v2 Esch independent | 79.90% | 38.72% | 49.45% | Prior independent-test result/screenshot; notebook 03 not attached |

**v1 validation per-class IoU:** water 25.09%, trees 47.58%, grass 27.14%, crops 26.91%, built-up 87.32%.

**v2 validation per-class IoU:** water **41.13%**, trees **61.41%**, grass **21.67%**, crops **31.55%**, built-up **90.57%**. v2 improves **water, trees, crops, built-up**, but **grass declines** versus v1. Overall metrics improve; do not claim every class improved.

**v2 independent Esch/Belval per-class IoU (previously logged, not independently rerun here):** water **7.04%**, trees **64.35%**, grass **25.55%**, crops **14.01%**, built-up **82.63%**. Support: **181,543** kept-label pixels, matching the exact count of raw five-class Esch pixels in the uploaded holdout TIFF; independent metric computation itself still needs notebook `03` inspection. This is **strong indirect consistency evidence**, not a proof the evaluator was otherwise correct.

**Learning curves (saved notebook outputs):** v1 train loss 1.2244→0.9714→0.8893, validation loss 1.0602→0.9568→1.1966 (best epoch 2). v2 train 1.2757→0.9884→0.8683, validation 0.8512→1.0998→0.7791 (best epoch 3). **Three epochs are too few to claim convergence**, and v1 validation loss increases at epoch 3. Report modest prototype scope and additional training as future work.

**Critical metric caution:** validation 86.48% is not equivalent to independent Esch 79.90%. Do not use the former as final independent-generalization accuracy. The Streamlit Model Details tab displays **79.90%, 38.72%, 49.45%** and states metrics are DW pseudo-label-based, not automatically applicable to newly selected regions.

## 9. Inference/analysis code: what was directly source-checked

**Corrected working backend** `sartrack_pipeline_fixed.py` (231 lines; renamed locally `sartrack_pipeline.py`):

- Rejects coordinates outside UTM latitude range [-80,84), invalid month/year, less than 12 completed historical months, missing model weights.
- Uses GEE project `sartrack-lulc-2026`, area `point.buffer(2500).bounds()`, local UTM, consistent **ascending** orbit by histogram maximum. Fix for keys like `'88.0'` is `int(float(...))`.
- Downloads 12 months' SAR in two-band GeoTIFF at scale 10; verifies 2 bands and **exactly matching raster grids**. Caches existing SAR TIFFs in a location/date-specific folder.
- Computes shared validity `logical_and.reduce` over 12 months, runs inference using **non-overlapping 128×128 tiles**, zero-pads edges and masks NoData=255.
- Saves 12 `LULC_YYYY_MM.tif` + 12 `LULC_YYYY_MM.png`; each LULC TIFF is `uint8`, EPSG projected, nodata 255; `0..4` classes. Saves CSV with `Month`, `{class}_pct`, `{class}_monthly_change_pp`, `{class}_since_start_pp` for five classes.
- Percentages use **one fixed denominator**: pixels valid across all 12 months. Start-to-end change is **percentage points (pp)**, not percent growth.
- Change count is per-pixel number of unequal adjacent predicted classes across 11 transitions (0–11). Stable class retains original class only where change count==0; other cells 255. Stability = stable/common-valid.
- Generates summary and map PNGs. Distinguishes predicted model agreement from actual land-use stability.

**Known technical limitations / caveats:** no uncertainty calibration, validation for geographic generalization, edge artifact testing or boundary smoothing; zero padding at tile borders can introduce seams; a single consistent orbit may not exist for some 12-month windows; GEE service limits/quotas and network failures may interrupt. Cached SAR files are used when same output path already exists; no complete provenance manifest/checksums currently stored.

**Verified frontend source** `app_updated.py`: live `run_pipeline()` invoked on generate; validates 12 PNG+TIFF, CSV and five stability artifacts before displaying; separate instant demo for original Esch Jan–Dec 2025; input date and coordinates tied to active session selection; 5 tabs retained; 12-month dates support year crossings; `csv` reader rather than pandas. Error display does not silently fall back to stale older location results.

**Two competing Esch result variants are expected and should be labeled separately:**

- Original *precomputed* Jan–Dec 2025 Esch: Jan–Dec Built-up **55.31%→51.98%**, **−3.33 pp**; stability **64.45%** (common-valid 250,498, stable 161,440). Different one-month June statistics were computed with a different valid-pixel denominator; do not mix.
- New *live* Jan–Dec 2025 Esch: December Built-up **52.28%**, change **−3.56 pp**, stability **66.10%** (terminal common-valid 250,484). Live Esch projection EPSG:32631 rather than original EPSG:32632. Differences should be explained as different grids/masks/processing, not mistaken for 2025 real change.

## 10. End-to-end demonstration evidence (previous screenshot/terminal verification)

| Place | Lat | Lon | Interval | Maps | End built-up | Built-up pp delta | Prediction stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Esch/Belval | 49.5000 | 5.9670 | Jan–Dec 2025 | 12/12 | 52.28% | −3.56 pp | 66.10% |
| Luxembourg City | 49.6116 | 6.1319 | Jan–Dec 2025 | 12/12 | 76.77% | −2.67 pp | 83.65% |
| Luxembourg City | 49.6116 | 6.1319 | Jul 2025–Jun 2026 | 12/12 | 78.21% | +1.38 pp | 85.47% |

Visual/browser checks showed maps switch correctly, monthly statistics and monthly-change tables render, stability maps display, Model Details shows the independent Esch pseudo-label metrics, and GeoTIFF downloads work. **Cross-year** sample `LULC_2026_06.tif` was uploaded and directly checked as valid georeferenced 10m `uint8` GeoTIFF with five classes +255.

**Direct June 2026 GeoTIFF verification:** EPSG:32632, width 518, height 520, 269,360 raster cells. Class counts **0:6,555; 1:35,452; 2:10,087; 3:2,485; 4:195,885; nodata 255:18,896**. Valid pixels = **250,464**; built-up **195,885 / 250,464 = 78.21%**, agreeing with the app's June 2026 KPI. This independently checks the output class accounting but not real-world accuracy.

**Date coverage condition:** live run must contain 12 *complete past* calendar months; no assumption that future month's imagery exists. If data are unavailable for a specific orbit/month, pipeline should fail rather than invent 12 maps.

## 11. Data/label correctness verdict — status matrix

| Claim | Status | Evidence / required verification |
| --- | --- | --- |
| Source LULC taxonomy and label code meanings | PASS | Official Dynamic World 0–8 table checked. |
| Raw 3-band order and native raw label range | PASS | 11/11 TIFFs show VV/VH/label, values 0–8, -1/NaN only. |
| Geospatial raster resolution, CRS and within-city grid | PASS | 11/11 TIFFs 10 m EPSG:32632; consistent per city. |
| Outside-area missing-data alignment | PASS | NaN masks identical across bands and months per city. |
| Target 5-class code values in training masks | PARTIAL | Training notebook shows [0,1,2,3,4,255]; mapping origin unverified. |
| Correct raw→target code mapping in preprocessing | OPEN / HIGH | Need preprocessing notebook, pixel test (4→3,6→4; raw 3 ignored). |
| Uniform top-class probability threshold 0.50 | OPEN / HIGH | Two June exports 0% -1 vs others up to 46%. |
| Correct temporal DW label aggregation / SAR alignment | OPEN | Need actual GEE JS scripts and month/sample comparison. |
| NN really consumes VV/VH only | LIKELY / PARTIAL | Training tensors have 2 channels, live code selects VV/VH; inspect NPZ producer. |
| Geographic train/val/test isolation | LIKELY / PARTIAL | Reported distinct cities; inspect patch file provenance. |
| Model evaluation implementation | PARTIAL | Training notebook confusion and metrics code checked; holdout notebook missing. |
| Model generalization to arbitrary locations | NOT ESTABLISHED | Only two successful input regions, no new-location ground truth. |
| True land-use change detection | NOT ESTABLISHED | Stability counts predicted class changes, not verified transitions. |
| Cross-year TIFF downloadable class accounting | PASS | June 2026 TIFF counts match 78.21% app KPI. |
| Fresh-machine setup and full notebook reproducibility | OPEN | Requirements not clean-installed; notebook 01/03 not checked. |

**Bottom line:** We can accurately say "the source class codes are valid and the intended label taxonomy is correct." We cannot yet say "all labels are correct" or "all months use the same confidence policy." Two June files are a priority. Low independent per-class scores, particularly water (7.04%) and crops (14.01%), mean the system should be presented as a **proof-of-concept SAR-only supervised classifier with optical pseudo-labels**, not reliable production land-cover change detection.

## 12. Remaining work, priority, effort, completion criteria

Previous quick summaries described "four major tasks" because notebook QA, report, recording and submission were grouped. For precise execution, split this into **seven concrete work packages**. Estimates are **work planning ranges**, not promises of delivery time; fixes found in label audit can expand scope.

| Priority | Task | Focused estimate | Definition of done |
| --- | --- | --- | --- |
| P0 | 1. Label provenance and confidence-mask audit | 1–3 h | Compare GEE export JS and preprocessing notebook. Explain/re-export suspect June masks if needed. |
| P0 | 2. Notebooks, split integrity, metric QA | 1–3 h | Open executed notebook 01 & 03; confirm NPZ remap, saved plots, held-out Esch and metric denominators. |
| P0 | 3. Dataset package and shareable dataset link | 0.5–2 h | Provide uploaded accessible dataset or verified link, weights, and exact reproducible folder layout. |
| P1 | 4. Final reproducibility & package QA | 1–2 h | Check cleaned requirements, paths, model weights, 12 outputs and a fail-safe run; make clean handoff archive. |
| P0 | 5. Technical report and references | 4–7 h | Methods, true outputs, validation/test tables, per-class metrics, plots, constraints and limitations. |
| P1 | 6. Full demonstration screen recording | 0.5–1.5 h | Show 2nd location selection, 12 maps, metrics, stability, downloads and honest limitations. |
| P0 | 7. Submission audit & Moodle delivery | 0.5–1.5 h | Verify everything uploaded/linked, notebook outputs preserved, file opens, deadline met. |

**Estimated remainder before surprises: ~8.5–20 hours focused work**. The *software prototype* is largely functional; the submission is **not** finished. The highest-risk issue is label confidence/mapping consistency. Do **not** invest in appearance polishing before QA and required artifacts are done.

**Ordered steps for next chat (one step per user turn):**

- **Step 23:** Upload/inspect `notebooks/01_Data_Preprocessing.ipynb` and original GEE scripts for Dudelange June / Thionville June / Bastogne June. Audit source labels / `-1` masks, raw→target remap, normalization, patch provenance. This is the single highest-leverage next action.
- **Step 24:** Inspect `notebooks/03_Independent_Test.ipynb`: confusion matrix, ignored-label handling, 181,543 valid mask, class IoU and absence of Esch in training.
- **Step 25:** Verify the 294/196 NPZ file split, class counts, augmentation claims and exported weights/checksums. Compare preprocessing and inference normalizers.
- **Step 26:** Confirm dataset deliverable, accessible link, reproducible folder layout and exact list of artifacts.
- **Steps 27 onward:** Write technical report (include both val/holdout metrics and data-quality caveats), then record demo, run final QA, and submit.

**Progress check-ins every 5–10 steps:** give actual completed and remaining work packages, not invented precision or automatic "100%" claims. Keep communication concise Hinglish/Marathi, step-by-step; do not repeatedly bring up UI styling while required work remains.

## 13. Cross-verification tests to run locally (read-only first)

**A. Upload raw preprocessing notebook and JS export definitions; inspect exact expressions:**

```text
GEE raw labels should originate from GOOGLE/DYNAMICWORLD/V1.
Verify top1 = max(water, trees, grass, flooded_vegetation, crops,
                  shrub_and_scrub, built, bare, snow_and_ice).
Verify final confidence threshold policy (claimed >= 0.50).
Verify whether temporal mode is applied before/after masking.
Verify masks preserve low confidence as -1, not an accidental DW class.
Confirm Dudelange June and Thionville June export code paths.

After georeferenced pixel alignment, raw label to NN mask must be:
0->0 Water, 1->1 Trees, 2->2 Grass, 4->3 Crops, 6->4 Built-up;
(-1,3,5,7,8,NaN) -> 255 ignored.
```

**B. Recompute class statistics from the processed NPZ patches** (Python, no pandas):

```python
from pathlib import Path
from collections import Counter
import numpy as np
root = Path('data/processed')
for split in ('train', 'val'):
    counter = Counter(); bad = []
    files = sorted((root / split).glob('*.npz'))
    for file in files:
        with np.load(file) as z:
            x, y = z['x'], z['y']
        if x.shape != (2, 128, 128) or y.shape != (128, 128):
            bad.append(file.name)
        if not set(np.unique(y).tolist()) <= {0,1,2,3,4,255}:
            bad.append(file.name + ': invalid class code')
        vals, cnt = np.unique(y, return_counts=True)
        counter.update(dict(zip([int(v) for v in vals], [int(c) for c in cnt])))
    total_kept = sum(counter[k] for k in range(5))
    print(split, 'patches:', len(files), 'bad:', bad[:10],
          'included:', total_kept, 'ignored:', counter[255])
    if total_kept:
        print('class_%:', [round(100*counter[k]/total_kept, 3) for k in range(5)])
```

**C. Cross-year output GeoTIFF & statistics check:** for a live folder, require exactly 12 georeferenced TIFFs with the same CRS/transform/shape; all non-nodata classes in `{0..4}`; all 12 months use the same valid-pixel footprint; each statistics row sums to about 100%; per-month deltas equal differences of adjacent percentages within rounding; stability equals fraction of unchanged predicted classes. **This has been partially verified, not yet run programmatically for all output folders.**

**D. No-label-leakage check:** Inspect `data/processed/*/*.npz`: `x.shape[0]==2`, never include the label band as a third channel; `y` is separate. Confirm labels are not used to calculate monthly live inference output (current live backend selects `VV,VH` only).

**E. Manual sanity-check proposal (recommended, not yet done):** sample 30–50 stratified points per class in one validation city and Esch or Luxembourg City, compare to high-resolution visible imagery/source Dynamic World confidence while accounting for acquisition dates; summarize agreement and ambiguity. This does **not** replace a rigorous ground-truth test, but improves credibility and reveals pseudo-label errors.

## 14. What the technical report must explicitly say

- **Method is *SAR-only at inference*, not SAR-only labeling.** Dynamic World labels were generated using Sentinel-2 optical information. Calling the entire training pipeline purely SAR-only would be misleading.
- **Source backscatter is GEE-processed sigma-nought dB:** calibration, thermal noise removal and terrain correction are GEE ingestion steps. No bespoke speckle filter, explicit radiometric terrain flattening or additional SAR physics calibration was implemented in our Python inference source.
- **Source five-class taxonomy is a deliberate subset of Dynamic World nine-class taxonomy.** It is not a learned nine-to-five merger of all classes; unused classes are ignored. `raw 4→target 3`, `raw 6→target 4` must be explicit.
- **Confidence/ignore mask implementation requires verification**, particularly the two June source GeoTIFFs with 0% `-1`. Do not hide this or claim verified uniform 0.50 threshold until checked.
- **Overlapping validation patches** and **geographic holdout** play different roles; avoid counting duplicated pixels as fully independent samples.
- **Independent holdout Esch metrics**: OA 79.90%, mIoU 38.72%, macro F1 49.45% versus DW; water IoU 7.04%, crops IoU 14.01%. New Luxembourg City maps have **no directly measured accuracy**.
- **Stability 64.45/66.10/83.65/85.47%** is a *model prediction consistency index*, not a validated real-world stability/change rate. The project should not claim genuine land-use transitions from these numbers alone.
- **Different projections and common valid-pixel sets** explain why original precomputed and new live Esch percentages are not numerically identical; use clearly labeled experiments throughout tables and screenshots.
- Include input data area/scene count, monthly availability, class support, sampling scheme, model spec, fixed normalization, loss weights, optimizer, v1/v2 curves, all five class IoUs, failure modes, small number of epochs, limitations and future improvements.
- Include references to the primary products: ESA/Copernicus Sentinel-1, Google Earth Engine S1 GRD, Google/World Resources Institute Dynamic World, and the original Dynamic World science paper.

## 15. External authoritative reference links

1. **Official Dynamic World dataset and 9-class code table:** https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
2. **Dynamic World compositing / confidence discussion:** https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-1
3. **Official Sentinel-1 GRD processing guide:** https://developers.google.com/earth-engine/guides/sentinel1
4. **Official Sentinel-1 Earth Engine catalog:** https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
5. **PyTorch compatible install selector:** https://pytorch.org/get-started/locally/

These authoritative links support **taxonomy and standard data-product descriptions only**; they do not validate the user's specific exported labels or trained model.

## 16. One-paragraph new-chat instruction (paste if context is lost)

> We are finishing SARTrack, a Master's Sentinel-1 VV/VH-only-inference U-Net v2 five-class monthly land-cover project, due Nov 9, 2026. Our live Streamlit app works for Esch Jan–Dec 2025 and Luxembourg City Jan–Dec 2025 and Jul 2025–Jun 2026: 12 maps, statistics, stability, downloads. Model v2 trained with 294 patches, validation 196; independent Esch DW pseudo-label OA 79.90%, mIoU 38.72%, F1 49.45%. README/requirements already added. **Next step is Step 23: audit preprocessing/labels**, especially `Dudelange_June2025_Test.tif` and `Thionville_June2025.tif` having zero `-1` pixels while other month exports have many. Need verify actual 0.50 DW probability mask, source-code remapping 4→3 crops and 6→4 built-up, split integrity, notebooks 01 and 03, dataset link; then report, recording, final QA. Read `SARTrack_MASTER_HANDOFF_AND_DATA_LABEL_AUDIT_2026-10-31.md` in full. Give one step at a time and a remaining-step checkpoint every 5–10 steps. Do not tell us to redo working app steps.

**Handoff produced from uploaded evidence, notebook output inspection, real Rasterio pixel distributions, prior chat logs/screenshots, and official Google data-product documentation. All unknowns are intentionally marked rather than assumed.**
