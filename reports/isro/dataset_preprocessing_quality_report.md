# Lunar Computer Vision — Dataset Preprocessing & Quality Report

**Dataset:** ISRO Craters & Boulders — Roboflow export, version 1  
**Prepared:** 12 October 2026  
**Scope:** Dataset inspection, annotation validation, duplicate screening, and preprocessing readiness  
**Dataset source:** https://universe.roboflow.com/isro-w900k/craters-boulders  
**License recorded in export:** CC BY 4.0

---

## 1. Executive summary

The local export contains **1,707 images** across training, validation, and test splits. The completed checks found no unreadable images, no image/label count mismatches, and no annotation geometry issues under the implemented validation rules.

An automated duplicate audit found **zero exact duplicate pairs** and **81 perceptual near-duplicate candidates**. One candidate pair spans validation and test and remains unresolved. Filename-prefix analysis identified 110 groups with filenames appearing across multiple splits; these are investigation clues, not confirmed duplicates.

**No original dataset images or labels have been deleted, moved, or modified during this audit.** Reports and contact sheets were saved separately under `reports/isro/`. The checks do not prove that every annotation is semantically correct or rule out all possible source-image overlap.

## 2. Dataset structure

Expected project-root layout:

```text
Lunar-cv/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── src/data/
└── reports/isro/
```

The local export's `README.dataset.txt` reports 1,707 images. A previously observed Roboflow overview count of 712 differs from this local export count; the cause of the discrepancy has not been established. Counts below reflect the local export.

## 3. Dataset configuration

The `data.yaml` file was updated to use paths relative to the project root:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['crater', 'rille']

roboflow:
  workspace: isro-w900k
  project: craters-boulders
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/isro-w900k/craters-boulders/dataset/1
```

The exported class names are preserved: `crater` and `rille`. Although the project title mentions boulders, the exported YAML lists these two classes; this report does not rename or reinterpret them.

The export README documents 640 × 640 resizing using stretch, plus augmentations including rotations and brightness variation. All local images scanned were already 640 × 640, so no additional resizing was performed.

**Remaining configuration check:** Paths were checked against the local folder structure. A training-framework dataset-loading smoke test has not been recorded as completed, so the configuration should be tested with the intended framework before training.

## 4. Split and annotation counts

| Split | Images | Label files | Annotations |
|---|---:|---:|---:|
| Train | 1,493 | 1,493 | 6,499 |
| Validation | 143 | 143 | 636 |
| Test | 71 | 71 | 316 |
| **Total** | **1,707** | **1,707** | **7,451** |

### Class distribution

| Split | Crater | Rille | Total annotations |
|---|---:|---:|---:|
| Train | 3,141 (48.33%) | 3,358 (51.67%) | 6,499 |
| Validation | 350 (55.03%) | 286 (44.97%) | 636 |
| Test | 157 (49.68%) | 159 (50.32%) | 316 |
| **Total** | **3,648** | **3,803** | **7,451** |

The split proportions are approximately 87.46% train, 8.38% validation, and 4.16% test. These are the proportions in the downloaded export; no split reassignment was performed.

## 5. Checks completed

| Check | Result | Interpretation |
|---|---|---|
| Image readability | 1,707 scanned; 0 unreadable in dimension scan | All scanned images opened successfully |
| Image dimensions | All images 640 × 640 | Consistent with export README |
| Image/label counterparts | Counts match in all splits | No missing counterpart reported by implemented checks |
| Annotation format and geometry | 0 issue records in implemented checks | No malformed rows, invalid class IDs, non-finite/out-of-range coordinates, nonpositive box sizes, or out-of-bounds boxes reported |
| Duplicate filename stems within each split | 0 reported | No repeated stems reported by this check |
| Exact duplicate annotation rows | 0 reported | No identical rows reported by this check |
| Exact duplicate image files | 0 pairs | SHA-256 comparison found no byte-identical pairs |
| Perceptual near-duplicate scan | 81 candidate pairs | Similarity is not proof of shared source |
| Automated audit unreadable images | 0 | All 1,707 images hashed successfully |

One check flagged an `EMPTY_LABEL_FILE`. Empty labels may represent images with no annotated objects, so they should be interpreted in context rather than automatically treated as errors.

These results describe implemented checks; they do not guarantee semantic correctness of every label.

## 6. Duplicate and potential split-overlap audit

The automated audit used:
- SHA-256 hashes to find byte-identical files.
- A 64-bit difference hash (dHash) to identify visually similar candidates, with Hamming distance ≤ 4 as the candidate threshold.
- Filename prefixes before `.rf.` as supplementary grouping clues.

### Automated audit summary

| Metric | Result |
|---|---:|
| Images discovered | 1,707 |
| Images hashed | 1,707 |
| Unreadable images | 0 |
| Exact duplicate pairs | 0 |
| Exact duplicate pairs across splits | 0 |
| Near-duplicate candidates | 81 |
| Near-duplicate candidates across splits | 1 |
| Filename-prefix groups spanning multiple splits | 110 |

### Unresolved validation/test candidate

- **Validation image:** `valid/images/4165_jpg.rf.cae5c14fc0e6abbc22135c127a531299.jpg`
- **Test image:** `test/images/4191_jpg.rf.266ac35f07499b841f2da6d2056b512d.jpg`
- **dHash Hamming distance:** 2 out of 64
- **Same filename prefix:** No
- **Status:** Unresolved candidate — visual similarity observed, shared source not confirmed.

The side-by-side review shows a similar large crater and surrounding terrain patterns, but does not conclusively establish that both images depict the same source scene. Keep the candidate recorded for transparency. Do not describe it as confirmed leakage or automatically move/delete either image based on current evidence.

### Filename-prefix groups

The prefix audit identified 110 groups with filenames appearing in more than one split. Contact-sheet reviews of `24_png_jpg`, `102_png_jpg`, and `107_png_jpg` showed closely related images within some training groups, including different rotations/crops. For the reviewed examples, cross-split identity was not conclusively established from contact sheets.

Filename-prefix matches can indicate related exports or augmented variants, but they are not proof. The audit does not establish that all 110 groups share an original image, nor that widespread cross-split leakage exists.

### Duplicate-audit limitations

Perceptual hashes can miss transformations, crops, contrast changes, and rotations. They can also flag different lunar scenes with similar visual structure. Therefore:
- zero exact duplicates does not prove zero source overlap;
- one near-duplicate candidate is not automatically a duplicate;
- the prefix-group count is not a count of confirmed leakage;
- no dataset files were changed as a result of this audit.

## 7. Preprocessing actions performed

1. Corrected `data.yaml` paths to match the project-root folder structure.
2. Ran dataset analysis for image/label pairing, annotation format, class IDs, coordinate ranges, and bounding-box geometry.
3. Checked image dimensions and readability.
4. Generated annotation visualizations/contact sheets for sample review.
5. Ran exact and perceptual duplicate detection.
6. Generated a cross-split filename-prefix review.
7. Created automated audit CSV/JSON outputs and recorded the unresolved validation/test candidate.

**No original image or label files were deleted, moved, or edited during these steps.** Generated reports, CSVs, JSON files, and contact sheets are separate audit artifacts.

## 8. Report artifacts

Previously generated artifacts include:

```text
reports/isro/
├── dataset_quality_report.md
├── qc_summary.json
├── issues.csv
├── visualizations/
└── duplicates/
    ├── duplicate_summary.json
    ├── exact_duplicates.csv
    ├── near_duplicate_candidates.csv
    ├── unreadable_images.csv
    ├── manual_review.csv
    └── cross_split_prefix_review.csv
```

The automated audit adds:

```text
reports/isro/automated_audit/
├── summary.json
├── exact_duplicate_pairs.csv
├── near_duplicate_candidates.csv
├── cross_split_prefix_groups.csv
└── unreadable_images.csv
```

Actual file presence may vary with the scripts run locally. Verify the current folder before treating this as a complete inventory.

## 9. Remaining actions before training / GitHub handoff

- [ ] Run a dataset-loading smoke test with the intended YOLO training framework to confirm `data.yaml` path resolution.
- [ ] Keep the validation/test near-duplicate pair documented as unresolved unless stronger evidence is found.
- [ ] Consider a rotation/crop-aware similarity method if source-level split independence is a project requirement.
- [ ] Review suspicious annotations semantically; geometry checks alone cannot prove each box correctly labels a crater or rille.
- [ ] Reconcile the 712-image overview count with the 1,707-image local export count if provenance requires it; the cause is currently unknown.
- [ ] Verify licensing and attribution requirements before redistributing the dataset or derivative assets.
- [ ] Review report paths and commit only intended code and documentation to GitHub.

## 10. Conclusion

The local export is structurally consistent under the implemented checks: all 1,707 images are readable, image and label counts match, all images are 640 × 640, and no annotation geometry issues or exact duplicate file pairs were reported. The perceptual duplicate scan produced 81 candidates, including one unresolved validation/test candidate, while filename-prefix analysis produced 110 cross-split groups that are not confirmed duplicates.

The dataset has **not** been altered by deleting or moving files. Documentation and configuration validation can continue while the duplicate-audit caveats remain recorded. Do not claim that every label is semantically correct or that all source-level overlap has been ruled out.


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
