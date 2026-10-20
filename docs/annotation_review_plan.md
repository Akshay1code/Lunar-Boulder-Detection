# Proposed annotation-review plan

This is a targeted 20-image human-review shortlist, not a representative evaluation
sample or an annotation release. All entries are unreviewed. No boundaries, masks,
review decisions, split changes, or model-training actions are created by this plan.

## Scope and provenance

There are 14 training and 6 validation images: 12 with rectangular landslide labels
and 8 without landslide labels, spanning 16 unverified filename prefixes. Test
images are not selected. Existing splits are retained. Source prefixes do not
establish independent scenes; numeric patch suffixes are not coordinates.

Full exported image paths, existing-mask paths, COCO paths, image and annotation
IDs, original filenames, and freshly computed SHA-256 hashes are recorded in
review_shortlist.proposed.json. All 359 train/valid image and mask hashes and both
COCO file hashes were checked against the existing audit before selection.
IDs from COCO are scoped to their original split. Unknown reviewer details,
review decisions, verified scene IDs, and coordinates remain null.

## Reproducible selection method

Selection uses existing labels, full-resolution mask coverage, grayscale image
measurements, and earlier visual-screening concerns only. No checkpoint or model
predictions are used. Coverage is the fraction of existing mask pixels above zero;
it describes rectangular supervision, not verified geological area.

1. Select eight labeled training images at coverage target quantiles
   0, .15, .30, .45, .60, .75, .90, 1, followed by four labeled validation images
   at quantiles 0, .33, .67, 1. Compute targets over each split's nonempty-label
   pool using NumPy's default linear quantiles. For each target, prefer a filename
   prefix not previously selected in either split. Within that pool choose the
   smallest absolute coverage difference, breaking ties by image_key. If no unused
   prefix remains, consider all remaining images. Diversity takes precedence over
   hitting a target exactly; the last selection need not be the largest mask.
2. Include three previously identified visual-review candidates without landslide
   labels: train/ohr_010_patch_2689, train/ohr_000_patch_0259, and
   valid/ohr_009_patch_3100. Their earlier visual concerns are unresolved, not
   diagnoses or completed domain reviews.
3. Include train/ohr_007_patch_0257 as one representative of the known exact
   duplicate group. Its uniform dark appearance requires an interpretability
   assessment. This selection neither excludes nor reassigns any group member.
4. From remaining images without landslide labels, excluding known duplicate-group
   members, select in order: the training image with lowest grayscale standard
   deviation among images with mean intensity 20-230 inclusive; the training image
   with highest mean intensity; the training image with highest grayscale standard
   deviation; and the validation image with highest fraction of pixels below
   grayscale intensity 16. Break ties by image_key and never repeat an image.
   Measurements use cv2.imread(..., IMREAD_GRAYSCALE) at original resolution.

These brightness/contrast measures are review-priority heuristics, not geological
classifiers. The exact approved image list and per-image reasons appear below.

## Human review instructions

- Before drawing any geological conclusions, agree on the target definition and
  inclusion/exclusion rules with a qualified lunar-domain reviewer.
- Inspect each image for interpretability and available source context. Request
  original-resolution imagery where shadow, blur, brightness, or cropping limits
  interpretation. Record unresolved uncertainty rather than guessing.
- Assess whether existing labels are supported, whether omissions may be present,
  and whether any boundary could be defensibly delineated. Existing rectangles
  are reference aids, not verified geological boundaries.
- Do not designate empty masks as confirmed negatives. Missing labels and dark
  pixels do not establish absence. No candidate is a confirmed landslide merely
  because it was selected for this review.
- Record actual reviewer identity, date, evidence, and decision only after review
  is performed and its recording is authorized. Require independent review of
  proposed conclusions and adjudicate disagreements. Until then all records
  remain unreviewed and all decision fields remain null.
- Keep validation review independent of model predictions. Preserve all original
  assets and splits. Creating new annotations, masks, or a training dataset is
  outside this review-plan task and requires separate authorization.

## Approved shortlist

All entries below have review status unreviewed. Short names are COCO original
filenames; the JSON contains the exact exported file paths.

| ID | Split | Original filename | Existing landslide annotations | Existing mask foreground (%) |
|---|---|---|---:|---:|
| R01 | train | ohr_009_patch_1041.png | 1 | 1.27 |
| R02 | train | ohr_016_patch_3262.png | 2 | 4.58 |
| R03 | train | ohr_018_patch_0258.png | 2 | 8.09 |
| R04 | train | ohr_011_patch_4306.png | 5 | 10.48 |
| R05 | train | ohr_019_patch_0568.png | 4 | 14.84 |
| R06 | train | ohr_015_patch_1802.png | 3 | 19.40 |
| R07 | train | ohr_017_patch_0154.png | 6 | 27.94 |
| R08 | train | ohr_006_patch_1816.png | 6 | 52.97 |
| R09 | valid | ohr_013_patch_2704.png | 1 | 1.26 |
| R10 | valid | ohr_008_patch_0235.png | 2 | 5.24 |
| R11 | valid | ohr_004_patch_0901.png | 3 | 13.04 |
| R12 | valid | ohr_012_patch_4008.png | 1 | 25.36 |
| R13 | train | ohr_010_patch_2689.png | 0 | 0.00 |
| R14 | train | ohr_000_patch_0259.png | 0 | 0.00 |
| R15 | valid | ohr_009_patch_3100.png | 0 | 0.00 |
| R16 | train | ohr_007_patch_0257.png | 0 | 0.00 |
| R17 | train | ohr_000_patch_1347.png | 0 | 0.00 |
| R18 | train | ohr_019_patch_0467.png | 0 | 0.00 |
| R19 | train | ohr_021_patch_2240.png | 0 | 0.00 |
| R20 | valid | ohr_000_patch_3852.png | 0 | 0.00 |

Related guidance: annotation_limitations.md and manual_annotation_protocol.md.
