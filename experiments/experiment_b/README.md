# Experiment B: agreement with unverified rectangular reference masks

Infrastructure only. Training requires separate approval and an explicit
`--train` argument. No geological segmentation accuracy is established by this
experiment. Rectangular reference masks are unverified; empty masks are not
confirmed geological negatives. Pixels outside rectangles may also be mislabeled.

## Non-training commands (PowerShell, project root)

```powershell
.\.venv311\Scripts\python.exe -B experiments/experiment_b/train_experiment_b.py
.\.venv311\Scripts\python.exe -B experiments/experiment_b/train_experiment_b.py --validate-only
.\.venv311\Scripts\python.exe -B experiments/experiment_b/test_experiment_b.py
```

The first command displays help. Validation checks train/valid inputs and the
protected checkpoint hash, without constructing a model or optimizer, writing
output files, or creating a run directory. Tests use synthetic tensors, mocked
filesystem writes, and fake batch iterators; they perform no optimizer updates.
`-B` and the runner's bytecode setting avoid writing caches in the original code.

After a separate training approval, the command would be:

```powershell
.\.venv311\Scripts\python.exe -B experiments/experiment_b/train_experiment_b.py --train
```

Do not run it as part of infrastructure validation.

## Fixed design and safe selection

- Reuse root `UNet` and `LunarLandslideDataset`; never import `train_unet.py`.
- Initialize the same lightweight UNet from scratch. The old checkpoint is read
  for SHA-256 protection only, not loaded as training initialization.
- Scan train and valid only. Select the 73 training images whose original mask
  contains nonzero pixels out of exactly 279 pairs. Selection is an in-memory
  subset, not a split reassignment or a modification to the original dataset.
- Require validation counts of 80 total, 19 nonempty, 61 empty. Stop on count,
  COCO pairing, decoding, mask-value, shape, or resize-presence discrepancies.
- Preserve loader preprocessing: BGR to RGB, area image resize, float32 / 255,
  nearest-neighbor mask resize, and binary float32 masks. Image size is 256 x 256.
- Batch size 2, no dropped final batch, zero loader workers, two PyTorch CPU
  threads. Adam learning rate 0.001, otherwise Adam defaults; unweighted
  BCEWithLogitsLoss. No Dice loss, weighting, augmentation, or threshold tuning.
- Seed 42 for Python, NumPy, PyTorch, and a dedicated training shuffle generator.
  Request deterministic PyTorch algorithms. Record package versions and source
  hashes; bitwise identity across different versions/hardware is not guaranteed.
- The configuration is checked against the approved constants; discrepancies
  cause a safe stop, never an automatic parameter change.

## Update budget

Exactly 700 optimizer updates: ceil(279 / 2) x 5, matching the old configuration's
update count. With 73 selected images and drop_last=False, a pass has 37 batches.
Stop at pass 19, batch 34: 18 complete passes plus 34 batches. No additional batch
is fetched and no additional optimizer update occurs. This is not equal unique-
image exposure: selected images repeat much more often, raising overfitting risk.
The old checkpoint is historical, not a seed-matched control.

Evaluate all 80 existing validation images at updates 140, 280, 420, 560, and 700,
in fixed order, without gradients. Do not access or evaluate the test split.

## Metrics and model selection

All results describe **agreement with unverified rectangular reference masks** at
256 x 256 and sigmoid probability >= 0.5. TP, FP, and FN refer only to recorded
mask values, not established geological truth.

- Dice = 2TP / (2TP + FP + FN).
- Recall = TP / (TP + FN).
- Precision = TP / (TP + FP).
- Predicted foreground coverage = (TP + FP) / number of pixels, a fraction 0-1.
- Report per-image results, macro averages, valid counts for those averages,
  pooled pixel metrics, mean BCE, and fraction of images predicting foreground.
- Report the 19 nonempty references separately from `empty_unverified` (61).
  Undefined ratios are JSON null, not NaN and not perfect scores. If both masks
  are empty, Dice, recall, and precision are null. If the reference is empty but
  prediction is nonempty, Dice/precision are zero agreement and recall is null;
  these values do not establish geological false positives. Zero predictions on
  a nonempty reference give zero Dice/recall and undefined precision.
- Select the best checkpoint by highest nonempty-group macro Dice, then lower
  nonempty-group mean BCE, then earlier update. Report precision and coverage
  alongside Dice to reveal excessive foreground predictions. Do not select by
  raw overall BCE alone or by scores on empty masks.

## Isolated future run outputs

A separately authorized run creates a unique UTC timestamp + UUID directory
under `experiments/experiment_b/runs/`. Existing run directories and output files
are never reused or overwritten. Every output destination must resolve inside
that run. Redirected/escaping destinations and the original checkpoint path are
rejected. Original scripts, train/valid images, masks, COCO files, and checkpoint
hashes are verified again after execution. No masks or predictions are saved.

Each future run contains:

```text
configuration.json
data_manifest.json
environment.json
training_history.jsonl
validation_history.jsonl
validation_per_image.jsonl
summary.json
checkpoints/best.pth
checkpoints/last.pth
```

The manifest records all examined train/valid pairs and which 73 are selected,
original image keys/splits/COCO IDs and annotation IDs, absolute file paths, and
fresh image/mask/COCO hashes. It does not claim reviewed labels or new provenance.
Environment data records Python/platform/package versions and source hashes.
History contains per-update BCE, pass/batch counts, validation metrics, and elapsed
wall-clock seconds since run initialization (preflight is excluded).

The best state is retained in memory; best and last checkpoints are each written
once at successful completion. They include model/optimizer state, seed/config,
update/pass/batch position, validation results, and RNG states. This is an audit
record, not an exact mid-pass resume implementation. There is no automatic resume.
Interrupted/failed runs may contain partial histories; they are not completed runs
and must not be reused. No checkpoint is promised for interrupted training.

## Interpretation and next step

Excluding empty masks reduces potentially incorrect negative supervision but also
reduces diversity and may encourage overprediction. Rectangular labels can reward
incorrect boundaries. Source-scene independence remains unverified. Human review
and genuine boundary annotations remain necessary before scientific evaluation.
Run non-training checks first; obtain separate approval before any training run.
