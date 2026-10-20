# Experiment F: BCE plus soft Dice

Controlled baseline: Experiment E, run `20261013T133640Z_402e62786bab407598214d53f521fae9` in `experiments/experiment_b_exposure/runs/`.

This measures training-image agreement with unverified rectangular reference masks, not validation performance or geological accuracy. Missing annotations are not confirmed negatives. Only training images/masks are accessed.

## Exact loss definition

For image i, let z be logits, p = sigmoid(z), y be the binary reference, and N be its pixel count:

```
BCE_i = mean over pixels of BCEWithLogits(z, y)
soft_Dice_i = (2 * sum(p * y) + 1.0) / (sum(p) + sum(y) + 1.0)
Dice_loss_i = 1 - soft_Dice_i
total_loss = mean over images(BCE_i) + mean over images(Dice_loss_i)
```

Both weights are 1.0. The smoothing constant is exactly 1.0 in numerator and denominator. Spatial/channel dimensions are reduced per image before the batch mean; this is not pooled batch Dice. Denominators use linear sums, not squared values. Computation uses float32, stable BCE-with-logits and sigmoid; no logarithm of sigmoid and no thresholding enter the loss. Finite inputs and binary targets are required.

An empty reference uses the same smoothed formula: Dice loss = `1 - 1/(sum(p)+1)`. It is finite, reaches zero for zero predicted probability, and penalizes predicted mass. This computational definition does not establish geological absence. Empty references remain excluded from the actual 73-image training/evaluation selection. Undefined binary precision and recall remain null, with defined-image counts reported; soft Dice loss does not replace these binary metrics.

## What stays controlled

Exactly the same 73 selected training records and ordering are checked against both B's manifest and E's saved provenance, including IDs, paths, hashes, annotations, resized coverage and groups. Original COCO fill rules are checked in memory without writing masks. The original dataset loader and UNet are reused unchanged.

Seed 42, fresh initialization, deterministic CPU computation with two threads, 256x256 RGB/255 preprocessing, original resize methods, shuffled batches of two, no dropped final batch, zero workers, Adam with learning rate 0.001 and default parameters, and threshold >=0.5 are unchanged. A separate evaluation generator preserves the training shuffle sequence. E's source and configuration hashes are pinned; package versions and original sources are checked against B.

The budget remains 5,180 successful updates: 37 batches per complete pass, 140 passes and 140 presentations per image. Evaluation remains update 0 and every 140 through 5,180 (38 evaluations). Binary BCE, Dice, recall, precision and predicted/reference coverage retain E's definitions. BCE is a loss, not a binary metric.

Only the training objective changes. Configuration differences from E are `experiment: F`, the combined `loss`, its explicit `loss_parameters`, and `experiment_e_sha256` provenance metadata. Training history additionally records `bce`, `dice_loss`, and `total_loss` for every pre-update batch. Evaluations record these components per image and as macro means, alongside unchanged binary metrics. Coverage groups remain <5%, 5–<15%, 15–<30%, and >=30%, based on resized reference coverage, not predictions.

## Checks and use

From the project root in VS Code PowerShell:

```powershell
& .\.venv311\Scripts\python.exe -B experiments\experiment_f_dice_loss\test_experiment_f.py
& .\.venv311\Scripts\python.exe -B experiments\experiment_f_dice_loss\train_experiment_f.py --validate-only
```

Default invocation prints help. Validate-only creates no model, optimizer, output files or run directory. Tests use synthetic loss/gradient calculations, fake optimizer steps, in-memory serialization and mocked filesystem operations. They do not train a model or write checkpoint files.

Training is not authorized during infrastructure preparation. A separately authorized run requires explicit `--train`:

```powershell
& .\.venv311\Scripts\python.exe -B experiments\experiment_f_dice_loss\train_experiment_f.py --train
```

## Outputs and safeguards

All outputs go to a new exclusive `runs/<UTC timestamp>_<UUID>/` under this experiment: configuration, provenance, environment, hashes before/after, training and evaluation histories, summary, best.pth and last.pth. Per-image and coverage-group metrics are nested in evaluation history and checkpoint evaluation metadata. Existing runs are never resumed or reused.

Best selection remains strictly improving mean binary training Dice with earliest ties retained. Update 0 initializes best. Model/optimizer tensors are deep-copied at the chosen update, serialized to exclusive staging, reloaded and verified before publication. Only the owned, hash-verified current-run best.pth may be replaced. last.pth is written separately and exclusively; summary records best/final metrics, updates and checkpoint hashes.

Optimizer pre/post hooks forbid update 5181 and count successful calls independently. Completion checks require exactly 5,180 successful steps, matching Adam counters, 140 complete passes, 10,220 presentations and all 38 evaluations. Failure/interruption produces no completed summary; partial files remain for inspection. No resume or automatic repair is implemented.

Resolved output paths must remain in the new run. Protected sources, training assets, existing checkpoints and experiment files are hashed before and after execution. Any discrepancy stops execution without repairing protected files. No existing scripts, dataset splits or masks are modified.

Compare F with E using unchanged binary overlap metrics, coverage, per-image results and groups; combined total loss is not directly comparable to E's BCE-only training loss. Equal component weights alter gradient magnitude as well as objective shape, so any change cannot isolate overlap-specific gradients from effective step-size effects. Full filesystem publication and training completion require a later authorized run; infrastructure tests cannot establish generalization or geological accuracy.
