# Targeted Review and Exploratory Scripts

This directory contains specialized review utilities and exploratory scripts developed during the dataset quality audit. They are organized here to preserve exploratory capability without cluttering the primary preprocessing pipeline in `src/preprocessing/`.

## Script Directory

| Script | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `inspect_final_train_test.py` | 2x2 contact sheet of high-priority train-test overlap groups (`15-00-39` & `15-01-32`). | `train/images`, `test/images` | `reports/isro/automated_audit/batch_review/contact_sheets/final_train_test_review.jpg` |
| `inspect_sift_matches.py` | Visualizes SIFT inlier match lines for `Screenshot-2022-02-25-16-19-45_png_jpg`. | `train/images`, `valid/images` | `reports/isro/automated_audit/batch_review/contact_sheets/sift_inlier_matches.jpg` |
| `inspect_top_overlap.py` | Contact sheet of top train-valid overlap pairs for `16-19-45`. | `train_valid_sift_review.csv` | `reports/isro/automated_audit/batch_review/contact_sheets/top_train_valid_overlap.jpg` |
| `make_102_review.py` | Contact sheet for the `102_png_jpg` cross-split prefix group. | `cross_split_prefix_groups.csv` | `reports/isro/automated_audit/102_png_jpg_review.jpg` |
| `make_prefix_review.py` | Contact sheet for `Screenshot-2022-02-25-14-12-12_png_jpg`. | `cross_split_prefix_groups.csv` | `reports/isro/automated_audit/screenshot_141212_review.jpg` |
| `make_remaining_contact_sheets.py` | Batch contact sheets for remaining candidate train-validation prefixes (`16-20-17`, `16-20-20`, `16-20-58`, `images-_10__jpg`). | `image_level_manifest.csv` | `reports/isro/automated_audit/batch_review/contact_sheets/review_*.jpg` |
| `review_cross_split_pair.py` | Targeted SIFT visualization for `valid/4165_jpg` vs `test/4191_jpg`. | `valid/images/4165_jpg...`, `test/images/4191_jpg...` | `reports/isro/visualizations/cross_split_inliers.jpg` |
| `compare_crater_images.py` | Pixel difference and label comparison for `15-03-38` pair. | `train/images/Screenshot-...15-03-38...` | Console output (MAE, label diff) |
| `compare_crater_features.py` | Feature overlay for `15-03-38` candidate pair. | `train/images/Screenshot-...15-03-38...` | `reports/isro/visualizations/crater_feature_overlay.jpg` |
| `compare_crater_alignment.py` | Homography alignment and difference image for `15-03-38`. | `train/images/Screenshot-...15-03-38...` | `reports/isro/visualizations/crater_aligned_overlay.jpg`, `crater_aligned_difference.jpg` |
| `prepare_top5_review.py` | Filters the top 5 largest crater boxes from `large_bbox_review.csv`. | `large_bbox_review.csv` | `reports/isro/top5_crater_review.csv` |
| `review_top5_craters.py` | Visual contact sheet for the top 5 largest crater bounding boxes. | `top5_crater_review.csv` | `reports/isro/visualizations/top5_high_priority_craters.jpg` |
| `review_remaining_craters.py` | Visual contact sheet for remaining high-priority crater annotations. | `remaining_crater_review.csv` | `reports/isro/visualizations/remaining_high_priority_craters.jpg` |
| `review_large_boxes.py` | Contact sheet for all high-priority crater bounding boxes (area >= 50%). | `large_bbox_review.csv` | `reports/isro/visualizations/high_priority_craters.jpg` |
| `verify_duplicate_candidates.py` | Runs SIFT and homography verification across near-duplicate pairs. | `near_duplicate_candidates.csv` | `reports/isro/automated_audit/verified_near_duplicate_candidates.csv` |

## Usage

All scripts automatically resolve the project root and can be run directly using the project virtual environment:

```powershell
& .\.venv\Scripts\python.exe scripts/review/inspect_final_train_test.py
& .\.venv\Scripts\python.exe scripts/review/inspect_sift_matches.py
```
