# Experiment I: joint quarter-turn augmentation

Infrastructure only. Training requires explicit approval and `--train`.
All metrics measure agreement with unverified rectangular references, not geological accuracy.

## Differences from F

1. Uniform independent per-image quarter turns (0/90/180/270 degrees, each 25%) on training tensors only. Identical `torch.rot90` permutations are applied to RGB and mask tensors after the existing loader preprocessing. No interpolation, clipping, padding or other augmentation. A separate seed-42 generator leaves model initialization and shuffled sampling unchanged.
2. **Validation-based checkpoint selection**, as explicitly requested: strictly higher macro Dice on the 19 nonempty validation references; retain the earlier checkpoint on ties. F instead selected by training Dice. Thus this is not a single-change comparison of checkpoint-selection procedures.
3. Record unaugmented selected-training and validation metrics at update 0 and every 140 successful updates through 5180 (38 evaluations).

Everything else remains F: fresh existing UNet, seed 42, deterministic CPU/two threads, batch 2/drop_last=False/zero workers, Adam lr 0.001/default remaining parameters, 73 identical nonempty training image keys/order, 5180 updates (140 passes). RGB INTER_AREA 256x256 float32 /255 CHW; binary masks INTER_NEAREST then >0. No validation augmentation or gradients. Training loop never consumes validation tensors.

Loss is imported directly from F: mean BCEWithLogits plus mean per-image soft Dice loss, coefficients 1+1. Soft Dice = (2 sum(sigmoid(logits)*target)+1)/(sum(sigmoid(logits))+sum(target)+1). No class weighting, no hard threshold in loss.

## Metrics and interpretation

Binary threshold >=0.5. Per-image Dice, IoU, precision, recall, predicted/reference foreground fractions, BCE, Dice loss and total loss. Aggregate all, nonempty and empty_unverified groups include defined counts and number with predicted foreground. Multiply coverage fractions by 100 for percentages. Both-empty Dice/IoU are N/A, one-empty overlap is zero, precision N/A without predictions, recall N/A without reference foreground. Empty references are not confirmed negatives.

Compare I best to F best on the same 80 validation images, 19 nonempty/61 empty. F Dice 0.490508 and IoU 0.345484 are nonempty-reference means. Repeated validation-based selection can bias reported validation performance; it is not an independent generalization estimate. Also report the final checkpoint to expose selection effects. Repeat the saved six-image orientation protocol only as a separate diagnostic, not as augmentation of primary validation. Improvements in consistency alone, especially consistently empty predictions, are not success. Define any material-degradation tolerance before reviewing results; none is assumed automatically.

## Commands (project root, PowerShell)

```powershell
.\.venv311\Scripts\python.exe -B experiments\experiment_i_rotation\test_experiment_i.py
.\.venv311\Scripts\python.exe -B experiments\experiment_i_rotation\train_experiment_i.py --validate-only
# Only after separate training approval:
.\.venv311\Scripts\python.exe -B experiments\experiment_i_rotation\train_experiment_i.py --train
```

Default invocation prints help only. Validate-only reads train/valid, verifies F's pinned environment, exact selected records/hashes, COCO polygon reconstruction and counts 73 / 80 / 19 / 61. It creates no model, optimizer, run or checkpoints. Tests use synthetic data; no optimizer updates.

## Outputs and preservation

Only `experiments/experiment_i_rotation/runs/<UTC timestamp>_<UUID>/` receives training outputs. Existing paths cannot be overwritten. Every strict validation improvement writes an immutable `best_validation_update_NNNNN.pth`, read back and checked for exact metadata/tensor consistency. On successful completion, `best.pth` is published once as a byte-identical copy of the selected immutable checkpoint; `last.pth` is separate. During an interrupted run, use the latest selected immutable candidate recorded in `incomplete.json`; no completed summary is written. Automatic resume is not implemented: do not assume saved augmentation state alone is sufficient to resume exact sampler position.

Configuration, provenance (selected records, versions, protected hashes), per-update loss/rotation draws, all per-image evaluation metrics, final summary, hashes and best/final checkpoints share the same run directory. Sources, prior experiments/checkpoints and train/valid inputs are hash-checked before/after; discrepancies fail without repair. No test split is instantiated or read. No existing source code is edited. Existing F functions are imported without invoking its CLI or training function.

Validation doubles evaluation work relative to F's training-only monitoring; runtime depends on CPU and is not guaranteed. Immutable improvement checkpoints require extra disk space. No recovery guarantee is made.
