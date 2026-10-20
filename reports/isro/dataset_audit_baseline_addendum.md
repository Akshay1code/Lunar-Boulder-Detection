# Lunar CV — Dataset Audit and Baseline Addendum

## Dataset
- Images: 1,707
- Annotations: 7,451
- Train: 1,493 images / 6,499 annotations
- Validation: 143 images / 636 annotations
- Test: 71 images / 316 annotations
- Classes: 0 = crater, 1 = rille
- Image dimensions: 640 x 640

## Quality audit
- No unreadable images or missing image/label counterparts were reported.
- No annotation geometry/format issues were reported by the implemented checks.
- No exact duplicate image pairs were detected.
- Perceptual-hash scripts reported 84 and 81 near-duplicate candidates using different methods.
- One validation/test similarity candidate remains unresolved; shared source is not confirmed.
- Large bounding boxes were reviewed as candidates, not automatically classified as errors.
- No original images, labels, or split assignments were changed.

## YOLOv8n reference results

Original 3-epoch validation baseline:
- Precision: 50.3%
- Recall: 51.6%
- mAP@50: 48.2%
- mAP@50-95: 26.3%

Best observed extended-run validation metrics in the shared logs (epoch 15):
- Precision: 56.3%
- Recall: 60.2%
- mAP@50: 62.4%
- mAP@50-95: 38.4%

Independent test evaluation of best.pt:
- Precision: 52.1%
- Recall: 53.9%
- mAP@50: 49.4%
- mAP@50-95: 26.1%

Per-class test mAP@50:
- Crater: 69.2%
- Rille: 29.6%

## Limitations and handoff
The test results are lower than the best observed validation metrics. This is a generalization gap worth investigating, but does not alone prove overfitting. Rille detection is the main observed weakness.

Verify artifact paths before sharing. Preserve the test split for final evaluation. Automated checks do not prove semantic annotation perfection or rule out all source-level overlap. Confirm dataset attribution/license requirements before redistribution.
