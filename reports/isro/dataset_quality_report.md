# Lunar Computer Vision — Dataset Quality Report

Generated: 2026-10-12T23:45:19+05:30

Project folder: `C:\Projects\Lunar-cv`

## 1. Executive summary

- **Images inventoried:** 1,707
- **Label files:** 1,707
- **Counted annotations:** 7,451
- **Classes:** 2 (0 = crater, 1 = rille)
- **Exact duplicate pairs:** 0
- **Near-duplicate candidate pairs:** 84
- **Near-duplicate candidates across splits:** 1

Structural checks confirmed clean (0 unreadable images, 0 missing labels, 0 invalid rows). Source-overlap screening identified probable overlap across splits that must be handled transparently.

## 2. Dataset split summary

| Split | Images | Label files | Annotations | Missing labels | Empty labels | Labels without images | Unreadable images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1493 | 1493 | 6499 | 0 | 0 | 0 | 0 |
| valid | 143 | 143 | 636 | 0 | 0 | 0 | 0 |
| test | 71 | 71 | 316 | 0 | 0 | 0 | 0 |

## 3. Overall class distribution

| Class ID | Class name | Annotation count | Percentage |
| --- | --- | --- | --- |
| 0 | crater | 3648 | 48.96% |
| 1 | rille | 3803 | 51.04% |

## 4. Bounding-box statistics

| Class ID | Class | Valid boxes | Mean width (norm) | Mean height (norm) | Mean relative area | Mean width (px) | Mean height (px) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | crater | 3648 | 0.224015 | 0.214781 | 0.069249 | 143.36938 | 137.459704 |
| 1 | rille | 3803 | 0.337946 | 0.389987 | 0.140911 | 216.285564 | 249.591507 |

## 5. Automated quality checks

| Check | Flagged count |
| --- | --- |
| Images without matching label files | 0 |
| Empty label files | 0 |
| Label files without matching images | 0 |
| Unreadable images in inventory | 0 |
| Issue records in issues.csv | 0 |

**Result:** Structural integrity verified. No coordinate violations, NaN values, or missing label counterparts.

## 6. Duplicate and Cross-Split Overlap Screening

- Images scanned: 1,707
- Exact duplicate pairs (SHA-256): 0
- Perceptual near-duplicate candidates (dHash): 84
- High-priority train-test candidates (SIFT homography):
  - `Screenshot-2022-02-25-15-00-39_png_jpg`: 1,274 good matches, 1,272 inliers (0.9984 ratio) -> **Probable source overlap**
  - `Screenshot-2022-02-25-15-01-32_png_jpg`: 1,104 good matches, 1,104 inliers (1.0000 ratio) -> **Probable source overlap**
- Unresolved validation-test candidate:
  - `valid/images/4165_jpg...` vs `test/images/4191_jpg...` (Hamming distance 2) -> **Inconclusive**

## 7. Status Classification and Recommendations

- **Verified structural checks:** Passed completely.
- **Probable source overlap:** Two train-test groups identified and documented.
- **Inconclusive candidates:** Retained for manual review; do not delete or reassign arbitrarily.
- **Checks not yet completed:** Semantic labeling correctness and training smoke test.
- **Strict policy:** No dataset images or annotations have been modified or deleted. Do not claim dataset is leakage-free.
