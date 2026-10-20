# Lunar Computer Vision (Lunar-CV)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/dataset-Roboflow%20ISRO-lightgrey.svg)](https://universe.roboflow.com/isro-w900k/craters-boulders)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end computer vision project for detecting craters and rilles on lunar surface imagery using YOLO object detection models.

This repository contains the dataset auditing, quality control, and preprocessing pipeline developed to establish dataset integrity, identify source overlap risks, and provide reproducible artifacts for model development and project handoff.

---

## Dataset Overview

- **Source:** ISRO Craters & Boulders (Roboflow Universe Export, version 1)
- **License:** CC BY 4.0
- **Annotation Format:** YOLO darknet format (`<class_id> <x_center> <y_center> <width> <height>`)
- **Class Mapping:**
  - `0`: `crater` (3,648 annotations)
  - `1`: `rille` (3,803 annotations)
- **Image Resolution:** 640 × 640 pixels (uniform across all splits)

### Verified Structural Counts

| Split | Images | Label Files | Total Annotations | Crater (ID 0) | Rille (ID 1) | Unreadable | Missing Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 1,493 | 1,493 | 6,499 | 3,141 (48.33%) | 3,358 (51.67%) | 0 | 0 |
| **Validation** | 143 | 143 | 636 | 350 (55.03%) | 286 (44.97%) | 0 | 0 |
| **Test** | 71 | 71 | 316 | 157 (49.68%) | 159 (50.32%) | 0 | 0 |
| **Total** | **1,707** | **1,707** | **7,451** | **3,648 (48.96%)** | **3,803 (51.04%)** | **0** | **0** |

All 1,707 images and 7,451 annotations pass non-destructive structural checks (0 unreadable images, 0 missing label files, 0 malformed coordinate rows).

---

## Project Structure

```text
Lunar-cv/
├── assets/
│   └── samples/                           # Clean lunar test samples for zero-download inference
├── configs/                               # Dataset & model YAML configurations
│   ├── data.yaml                          # Baseline 2-class (crater, rille) dataset config
│   ├── data_merged.yaml                   # 3-class (crater, rille, boulder) dataset config
│   ├── data_boulder_only.yaml             # Dedicated boulder detection dataset config
│   ├── data_eval_craters_rilles.yaml      # Crater/rille benchmark evaluation config
│   └── data_eval_boulders.yaml            # Boulder benchmark evaluation config
├── src/
│   ├── preprocessing/                     # Core 7-stage reproducible audit pipeline
│   │   ├── build_image_manifest.py        # Stage 1: Dataset inventory & manifest
│   │   ├── dataset_quality_audit.py       # Stage 2: Structural checks & YAML audit
│   │   ├── bbox_statistics.py             # Stage 3: Bounding-box distribution stats
│   │   ├── duplicate_audit.py             # Stage 4: Exact (SHA-256) & dHash screening
│   │   ├── source_overlap_audit.py        # Stage 5: Cross-split SIFT overlap screening
│   │   ├── generate_visualizations.py     # Stage 6: Annotation rendering & contact sheets
│   │   └── finalize_audit_report.py       # Stage 7: Report compilation & handoff docs
│   └── data/                              # Compatibility shims
├── scripts/
│   ├── app.py                             # Interactive Tkinter GUI detector
│   ├── train.py                           # Training pipeline with validation metric tracking
│   ├── evaluate.py                        # Standalone evaluation & F1-optimal threshold search
│   ├── merge_boulders.py                  # Boulder dataset merging & re-splitting utility
│   └── review/                            # Targeted exploratory inspection tools
├── weights/                               # Pretrained backbones & checkpoints (yolo11n, yolo11s)
├── train/                                 # Training split (labels tracked; images via Roboflow)
├── valid/                                 # Validation split (labels tracked; images via Roboflow)
├── test/                                  # Test split (labels tracked; images via Roboflow)
├── reports/
│   └── isro/                              # Quality reports, summaries, and visualization sheets
├── .env.example                           # Template for optional API keys & environment vars
├── .gitignore                             # Git exclusion rules (prevents heavy/binary/secret leaks)
├── requirements.txt                       # Project dependencies
├── data.yaml                              # Root dataset pointer (Ultralytics default)
└── README.md                              # Project overview (this file)
```

---

## Quickstart: Running the Preprocessing Pipeline

Run the complete pipeline from the project root in Windows PowerShell using the project virtual environment:

```powershell
# Set Python executable
$py = ".\.venv\Scripts\python.exe"

# Stage 1: Build image-level inventory manifest
& $py src/preprocessing/build_image_manifest.py

# Stage 2: Audit dataset quality, coordinate bounds, and YAML mappings
& $py src/preprocessing/dataset_quality_audit.py

# Stage 3: Compute bounding-box distribution statistics
& $py src/preprocessing/bbox_statistics.py

# Stage 4: Screen for exact and perceptual near-duplicates (SHA-256 + dHash)
& $py src/preprocessing/duplicate_audit.py

# Stage 5: Perform cross-split SIFT feature matching and homography screening
& $py src/preprocessing/source_overlap_audit.py

# Stage 6: Render annotation visualizations and split contact sheets
& $py src/preprocessing/generate_visualizations.py

# Stage 7: Finalize QC summaries, overlap findings, and handoff documentation
& $py src/preprocessing/finalize_audit_report.py
```

### Running Targeted Review Tools

To inspect specific image pairs or generate targeted contact sheets:

```powershell
# Inspect the top two train-test candidate overlap groups
& $py scripts/review/inspect_final_train_test.py

# Render SIFT inlier match lines for Screenshot-2022-02-25-16-19-45
& $py scripts/review/inspect_sift_matches.py

# Inspect the unresolved valid/test pair (valid/4165 vs test/4191)
& $py scripts/review/review_cross_split_pair.py
```

---

## Audit Findings & Leakage Status

Our non-destructive audit categorized findings into four clear categories:

1. **Verified Structural Checks:**
   - Image readability: 1,707/1,707 readable (0 unreadable).
   - Label alignment: 1,707 image-label counterparts (0 missing, 0 empty).
   - Bounding boxes: 7,451 annotations all have finite coordinates within normalized `[0, 1]` bounds and positive widths/heights.
   - Exact duplicates: 0 byte-identical duplicate pairs found by SHA-256 across all splits.

2. **Probable Cross-Split Source Overlap:**
   - `Screenshot-2022-02-25-15-00-39_png_jpg`: Train and test images share distinctive terrain and craters. SIFT feature matching produced **1,274 good matches** and **1,272 geometric inliers (99.84%)**.
   - `Screenshot-2022-02-25-15-01-32_png_jpg`: Train and test images share identical crater landmarks. SIFT feature matching produced **1,104 good matches** and **1,104 geometric inliers (100.0%)**.
   - *Status:* Classified as **probable source overlap**. Requires group-aware split reassignment if strictly independent testing is required.

3. **Inconclusive Candidates:**
   - `102_png_jpg`, `Screenshot-2022-02-25-14-05-01`, and `Screenshot-2022-02-25-14-12-12` have shared prefixes but weak SIFT match counts (< 10 inliers).
   - `valid/4165_jpg` vs `test/4191_jpg` share low dHash distance (Hamming distance 2) but remain inconclusive under geometric verification.
   - *Status:* Retained as documented candidates without modifying files or split assignments.

4. **Checks Not Yet Completed:**
   - Semantic annotation accuracy (expert lunar domain verification of ambiguous crater vs. boulder/rille boundaries).
   - Training smoke test with intended YOLO training framework.

> [!IMPORTANT]
> **Dataset Integrity Policy:**
> - No images or labels were deleted, moved, augmented, or overwritten during this audit.
> - The train/validation/test split was intentionally kept untouched to preserve experimental baselines.
> - The dataset is **not** claimed to be completely leakage-free due to the probable source overlap identified above.

---

## Key Reports and Artifacts

- [Preprocessing Pipeline Documentation](src/preprocessing/README.md)
- [Targeted Review Scripts Guide](scripts/review/README.md)
- [Dataset Quality Report](reports/isro/dataset_quality_report.md)
- [Comprehensive Preprocessing Quality Report](reports/isro/dataset_preprocessing_quality_report.md)
- [Final Dataset Audit Handoff](reports/isro/final_dataset_audit_handoff.md)
- [Source Overlap Findings](reports/isro/automated_audit/batch_review/source_overlap_findings.md)
- [Image-Level Manifest](reports/isro/automated_audit/batch_review/image_level_manifest.csv)
- [Train-Test SIFT Pair Results](reports/isro/automated_audit/batch_review/train_test_sift_pair_results.csv)
- [Train-Valid SIFT Review](reports/isro/automated_audit/batch_review/train_valid_sift_review.csv)

---

## Setup & Installation

### 1. Clone & Set Up Python Environment
Clone the repository and set up a Python 3.11 virtual environment:

```powershell
# Clone repository
git clone https://github.com/<your-username>/Lunar-cv.git
cd Lunar-cv

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download (ISRO Craters & Boulders)
To train or run full split evaluations, download the ISRO export from Roboflow:
1. Visit the [ISRO Craters & Boulders Roboflow Universe dataset](https://universe.roboflow.com/isro-w900k/craters-boulders).
2. Export the dataset in **YOLOv8 PyTorch format**.
3. Extract the downloaded image files into their corresponding directories:
   - `train/images/`
   - `valid/images/`
   - `test/images/`

*(Note: Bounding-box label annotations for all splits are already tracked in `train/labels/`, `valid/labels/`, and `test/labels/`.)*

---

## Quickstart: Running Inference

You do not need to download the full dataset to test detection. 17 sample images are included in `assets/samples/`.

### 1. Interactive Desktop GUI Detector
Launch the GUI application:
```powershell
python scripts/app.py
```
- Select an image or folder from `assets/samples/`.
- Adjust confidence threshold and IoU sliders in real time.
- View per-class counts, bounding boxes, and save annotated outputs.

### 2. Training
Train a YOLO11 model on the lunar dataset:
```powershell
# Baseline 2-class training (crater, rille)
python scripts/train.py --model yolo11s.pt --epochs 100 --batch 8 --name v2

# 3-class merged training (crater, rille, boulder)
python scripts/train.py --data configs/data_merged.yaml --model yolo11s.pt --name v3
```

### 3. Evaluation
Evaluate model performance and find the F1-optimal confidence threshold:
```powershell
python scripts/evaluate.py --weights weights/yolo11s.pt --splits test
```

