# Experiment E: exposure budget

This is a training-image fitting diagnostic against unverified rectangular reference masks. It does not measure validation performance, generalization, or geological accuracy. Missing labels are not confirmed negatives. No validation or test images are opened.

## Controlled comparison

Baseline: `experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113`.

The only change to training hyperparameters is `max_updates: 700 -> 5180`. This is 140 complete passes: 73 images, batch size 2, no dropped final batch, 37 updates per pass. Every selected image appears exactly 140 times. B provided approximately 18.93 exposures per image.

Selection is the exact 73 nonempty training records in B's saved manifest, retaining sorted loader order. Preflight compares all 279 training records, hashes, image IDs, annotation IDs, image/mask associations and nonempty selection against B. It verifies original mask pixels using the existing COCO polygon-fill rules in memory; no mask is exported. The existing loader supplies RGB/255 images with INTER_AREA resizing and binary masks with INTER_NEAREST resizing at 256x256.

Unchanged: fresh UNet initialization, seed 42 for Python/NumPy/PyTorch, CPU with two threads, deterministic algorithms, dedicated shuffle generator seeded 42, shuffled batches of 2, zero workers, Adam at 0.001 with original defaults, unweighted BCEWithLogitsLoss, threshold >=0.5. There is no checkpoint warm start. Package versions, baseline source files and metadata are pinned and checked. Evaluation uses its own generator and cannot consume the training shuffle generator.

Monitoring changes required by this experiment (not additional training changes):

- `experiment`: B -> E.
- `evaluation_split`: training selection only; validation count requirements removed.
- Evaluation added at update 0; the original 140-update interval is retained through 5180 (38 evaluations).
- Checkpoint selection changes from validation Dice/BCE tie-breaking to strictly improving macro training Dice. Earliest ties are retained. Update 0 initializes best.pth even when Dice is zero.
- Coverage-group reporting and baseline hash metadata are added. Original description and undefined-metric policies are retained.

`config.json` is compared against the pinned B configuration with exactly these documented transformations; arbitrary changes fail preflight. Reading B metadata containing saved validation records does not open validation files; only its training records are used.

## Metrics and grouping

Each evaluation records all 73 image keys/COCO IDs with BCE, Dice, recall, precision, predicted foreground coverage and reference coverage. Aggregate metrics are arithmetic per-image means; undefined precision (no predicted positive pixels) is null and excluded from its mean, with defined counts reported. Dice and recall remain zero for an all-background prediction on these nonempty references. Coverage is recorded as a fraction, not a percentage.

Fixed reference groups use the preprocessed mask coverage, independently of predictions: <5%, 5% to <15%, 15% to <30%, and >=30%. Each group has the same metrics, image count, defined counts and number of images with nonzero Dice. An empty group has null means. These groups do not change sampling or the loss.

## Safe commands in VS Code PowerShell

From the project root, run non-training checks:

```powershell
& .\.venv311\Scripts\python.exe -B experiments\experiment_b_exposure\test_experiment_e.py
& .\.venv311\Scripts\python.exe -B experiments\experiment_b_exposure\train_experiment_e.py --validate-only
```

The default invocation prints help. `--validate-only` verifies sources, package versions, train records, preprocessing, associations and hashes without constructing a model, optimizer, run directory or output files. Tests use synthetic tensors, fake optimizer hooks, mocked filesystem writes and in-memory checkpoint serialization; they do not train.

Training needs separate approval and an explicit `--train` flag. Do not run it during infrastructure validation:

```powershell
& .\.venv311\Scripts\python.exe -B experiments\experiment_b_exposure\train_experiment_e.py --train
```

## Outputs and protection

An authorized run creates `runs/<UTC timestamp>_<UUID>/` exclusively. Output path checks reject escapes and redirected run roots. Configuration, provenance, environment, protected hashes before/after, per-update training history, evaluation history (including per-image and group metrics), summary, best.pth and last.pth are stored there only.

Successful optimizer calls are counted by a post-step hook independently of loop indices. A pre-step hook rejects update 5181. Completion requires 5180 successful calls, matching Adam counters, 140 complete passes, 10,220 presentations and all 38 evaluations. Interruptions/errors propagate and never produce a completed summary. Partial run files remain for inspection; no automatic resume or repair occurs.

Best checkpoints copy model and optimizer tensors at the selected evaluation. Each improvement is serialized to exclusive staging in the new run, reloaded and compared tensor-by-tensor with its snapshot before publication. Only the owned, hash-verified best.pth within that run can be replaced; outside changes stop the run. last.pth is separately written exclusively and verified at completion. Summary records the best update, best/final metrics and checkpoint hashes. Best selection is training fit, not validation model selection.

Training assets, original checkpoint, source files and existing experiment metadata/checkpoints are hashed before and after execution. Discrepancies stop execution without repairing or overwriting protected assets. Runtime paths are confined to the new run; no existing dataset split or code is modified.

The evidence sought is sustained improvement in training Dice/recall and the number of fitted images, including low-coverage references, with falling BCE and sensible precision/coverage. A higher best training Dice would support insufficient exposure as one contributor; it would not establish geological accuracy. Failure would not by itself prove a loss-function or architecture bug.
