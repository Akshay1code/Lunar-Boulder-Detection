# Lunar CV — Cross-Split Source-Overlap Findings

Report generated: 2026-10-12

## Purpose

This report documents candidate source overlap between training and test images using filename-prefix grouping, SIFT feature matching, and manual visual inspection. A matching score alone does not prove leakage.

## Scope

- Train/test prefix groups processed: 33
- Pairwise comparisons recorded: 141
- Pairs flagged by configured screening threshold: 64

These counts describe the screening run, not confirmed duplicates.

## Priority findings

| Prefix group | Strongest good matches | Inliers | Inlier ratio | Classification |
|---|---:|---:|---:|---|
| `102_png_jpg` | 162 | 151 | 0.9321 | Inconclusive / Needs Review |
| `Screenshot-2022-02-25-14-05-01_png_jpg` | 10 | 6 | 0.6000 | Inconclusive / Needs Review |
| `Screenshot-2022-02-25-14-12-12_png_jpg` | 9 | 7 | 0.7778 | Inconclusive / Needs Review |
| `Screenshot-2022-02-25-15-00-39_png_jpg` | 1274 | 1272 | 0.9984 | **Probable Source Overlap** |
| `Screenshot-2022-02-25-15-01-32_png_jpg` | 1104 | 1104 | 1.0000 | **Probable Source Overlap** |

## Manual review notes

### `102_png_jpg`

The three training variants look related. The test image appears different in the contact sheet. Automated flags require reconciliation.

### `Screenshot-2022-02-25-14-05-01_png_jpg`

Training variants look related. Test/train visual correspondence is inconclusive; targeted SIFT produced only 5-6 inliers per comparison.

### `Screenshot-2022-02-25-14-12-12_png_jpg`

Training variants look related. Test/train correspondence remains unresolved; targeted SIFT produced only 7 inliers per comparison.

### `Screenshot-2022-02-25-15-00-39_png_jpg`

Strong visual correspondence between test and training scenes, supported by 1,272 inliers from 1,274 good matches (99.84%). High-priority probable source overlap.

### `Screenshot-2022-02-25-15-01-32_png_jpg`

Strong visual correspondence between test and training scenes. The strongest reported pair has 1,104 inliers from 1,104 good matches (100.0%). High-priority probable source overlap.

## Categorization of Findings

1. **Verified Structural Checks:** 1,707 images and 7,451 annotations pass all format, readability, and boundary checks.
2. **Probable Source Overlap:** `15-00-39` and `15-01-32` show near-identical geometric inliers (>1,100 inliers, >99.8% ratio) and shared scene features across train and test.
3. **Inconclusive Candidates:** `102_png_jpg`, `14-05-01`, `14-12-12`, and `16-19-45` exhibit weak or partial feature matches.
4. **Checks Not Yet Completed:** Semantic ground-truth verification of ambiguous crater/rille boundaries, and training smoke test.

## Recommended actions

1. Prioritize source-overlap verification for `15-00-39` and `15-01-32`.
2. Retain `102_png_jpg`, `14-05-01`, and `14-12-12` as cases requiring reconciliation; do not infer leakage from prefixes alone.
3. Do not delete images, alter annotations, or change split assignments without team approval.
4. If source overlap is confirmed, propose a separate group-aware split correction and evaluate independently.
5. Preserve the existing test set for unbiased evaluation; avoid tuning repeatedly against test.

## Limitations

- SIFT matching can produce coincidental matches on repetitive lunar terrain.
- High inlier ratios with few matches are not sufficient evidence by themselves.
- Findings are preliminary; the dataset must not be claimed to be completely leakage-free.


## Final dataset audit review

Review date: 2026-10-12

### Dataset integrity

- Total images: 1,707
- Train: 1,493 images, 6,499 valid annotations
- Validation: 143 images, 636 valid annotations
- Test: 71 images, 316 valid annotations
- Total valid annotations: 7,451
- Unreadable images: 0
- Missing label files: 0
- Invalid label rows: 0

The manifest counts agree with the earlier structural audit.
These checks do not establish that all annotations are semantically correct
or that the dataset is free of source overlap.

### Train-validation source-prefix review

Five groups were reviewed visually and with SIFT screening:

- Screenshot-2022-02-25-16-19-45_png_jpg:
  unresolved possible partial source overlap. The strongest pair had
  136 good matches and 85 geometric inliers (0.625 ratio).
  The visualization showed correspondences concentrated in part of the images.
- Screenshot-2022-02-25-16-20-17_png_jpg:
  inconclusive; maximum 15 inliers in the reported comparison.
- Screenshot-2022-02-25-16-20-20_png_jpg:
  weak SIFT evidence; maximum 6 inliers.
- Screenshot-2022-02-25-16-20-58_png_jpg:
  weak SIFT evidence; maximum 5 inliers.
- images-_10__jpg:
  possible partial overlap remains unresolved; maximum 24 inliers.

The training variants within some groups appear visually related or transformed.
Matching filename prefixes and SIFT correspondences are screening evidence,
not proof of leakage. Groups with weak matches have not been proven unrelated.

### Earlier train-test review

The earlier audit identified strong potential source-overlap candidates
in these cross-split groups:

- Screenshot-2022-02-25-15-00-39_png_jpg
- Screenshot-2022-02-25-15-01-32_png_jpg

Their reported SIFT results were very strong, and contact-sheet review suggested
related transformed versions of the same scene. Treat these as high-priority
potential cross-split source overlap, pending final source-level verification.

Other train-test prefix groups and SIFT screening candidates were also identified.
Prefix matches and automated flags must not be treated as confirmed duplicates.
An earlier validation-test pair also remains unresolved:
4165_jpg.rf.cae5c14fc0e6abbc22135c127a531299.jpg and
4191_jpg.rf.266ac35f07499b841f2da6d2056b512d.jpg.

### Decisions and limitations

- No images, labels, or split assignments were changed during this review.
- No candidate was removed based only on a filename prefix or SIFT score.
- Cross-split source overlap is not fully cleared; the candidates above must
  remain documented as unresolved or high-priority potential overlap.
- The validation/test contact-sheet review does not replace completion of
  the earlier train-test candidate review.
- Keep the test set untouched for model selection and tuning.

### Recommended handoff

Share this report, the image-level manifest, SIFT pair-result CSVs,
contact sheets, and existing quality/duplicate reports with the model owner.
Explain the unresolved cross-split candidates before interpreting evaluation
metrics. Do not claim that the dataset is leakage-free until those candidates
have been addressed or explicitly accepted with limitations.
