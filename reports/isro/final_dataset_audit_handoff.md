# Lunar CV — Dataset Audit Handoff

2026-10-12

## Dataset summary

- Images: 1,707
- Annotations: 7,451
- Train: 1,493 images / 6,499 annotations
- Validation: 143 images / 636 annotations
- Test: 71 images / 316 annotations
- Structural checks: 0 unreadable images, 0 missing labels, 0 invalid rows
- Classes: 0 = crater, 1 = rille

## Current status

Structural integrity checks passed. Cross-split source overlap is not fully
cleared. The train-test groups `Screenshot-2022-02-25-15-00-39_png_jpg`
and `Screenshot-2022-02-25-15-01-32_png_jpg` are high-priority potential
source-overlap candidates. Several train-validation groups remain unresolved
or have weak evidence.

No dataset files or split assignments were changed by this audit.

## Final train–test overlap review

Two high-priority source-prefix groups were visually reviewed alongside SIFT feature-matching results:

- **`Screenshot-2022-02-25-15-00-39_png_jpg`**: Probable source overlap. The train and test images share distinctive crater and terrain features. The strongest pair yielded 1,274 good matches and 1,272 geometric inliers (99.84%).
- **`Screenshot-2022-02-25-15-01-32_png_jpg`**: Probable source overlap. The train and test images exhibit strong geometric feature correspondence. The strongest pair yielded 1,104 good matches and 1,104 geometric inliers (100.0%).

These results strongly suggest related source imagery across the train and test splits. They do not independently establish the exact source-image lineage.

**Recommended action:** obtain approval for a group-aware split correction, keep related source variants within a single split, regenerate the affected split only under a documented procedure, and rerun the final evaluation on an appropriately separated test set.

No images, labels, or split assignments were changed during this audit. The dataset should not be described as fully leakage-free until the remaining candidates and the handling of these two groups are documented.

## Audit Categorization

1. **Verified structural checks:** All 1,707 images and 7,451 annotations validated for format, bounds, and readability.
2. **Probable source overlap:** `15-00-39` and `15-01-32` train-test candidate groups.
3. **Inconclusive candidates:** `102_png_jpg`, `14-05-01`, `14-12-12`, `16-19-45`, and `4165 vs 4191`.
4. **Checks not yet completed:** Semantic bounding-box labeling verification, training smoke test.

## Handoff checklist

- [ ] Share `dataset_quality_report.md`
- [ ] Share `dataset_preprocessing_quality_report.md`
- [ ] Share `dataset_audit_baseline_addendum.md`
- [ ] Share `automated_audit/batch_review/image_level_manifest.csv`
- [ ] Share `automated_audit/batch_review/train_test_sift_pair_results.csv`
- [ ] Share `automated_audit/batch_review/train_valid_sift_review.csv`
- [ ] Share contact sheets and `source_overlap_findings.md`
- [ ] Discuss unresolved cross-split candidates with the model owner

Do not describe the dataset as leakage-free until the remaining high-priority
candidates have been addressed or explicitly documented as accepted limitations.
