# Four-image diagnostic with best and last checkpoints

Baseline: `experiments/four_image_800/runs/20261013T125653Z_a4faa0e3502241c09a076016646d9a4c`.
Infrastructure only: training requires separate approval and an explicit `--train`.

All numerical training configuration is identical to Experiment D: fresh existing
UNet, seed 42, CPU/two threads, deterministic operations, batch size 1, Adam at
0.001, unweighted BCEWithLogitsLoss, 256x256 existing loader preprocessing,
threshold 0.5, and 800 successful optimizer updates. The fixed round-robin image
order is training COCO IDs **142, 263, 245, 96**, giving 200 updates per image.
No checkpoint initialization or validation/test split access occurs.

`config.json` matches the saved baseline configuration exactly. Preflight checks
this equality, baseline source hashes, actual selected image/mask hashes, COCO
associations, and exact in-memory agreement with original polygon-fill rules.
No mask is written or changed. The loader lists training pairs but decodes only
the selected four. Original model, loader, and experiment implementations are
never modified or executed for training by this runner.

## Checkpoint selection and consistency

Evaluate all four training images at update 0 and every 20 updates through 800.
Record per-image and macro mean BCE, Dice, recall, precision, and predicted and
reference foreground coverage, plus probability statistics. Undefined precision
is JSON null; macro means exclude undefined entries and record defined counts.
Coverage is stored as a fraction in [0,1].

Save `best.pth` on the initial evaluation, then whenever aggregate macro mean
**training Dice strictly improves**. Ties keep the earlier update; there is no
BCE tie-breaker and no early stopping. The selection policy is also embedded in
each checkpoint. Selection on these four training masks is not validation-based
model selection and does not establish generalization.

At each improvement, deep-copy model weights, optimizer state, configuration,
update number, and the full corresponding evaluation record. Serialize immediately
to an exclusively created staging file within the current unique run, reload it,
and compare every tensor and metadata field with that snapshot before publication.
This avoids state_dict references drifting as later optimizer updates occur.

The first `best.pth` publication refuses an existing destination on Windows.
Later replacements are allowed only for the best file created by this active run,
after checking its SHA-256 still matches the owned file. Replacement uses a
same-directory atomic rename. Previous runs and existing checkpoints are never
overwritten. Interrupted staging files may remain for inspection; a previously
published best stays complete. No automatic cleanup or resume is performed.

At update 800, save `last.pth` separately using exclusive creation and reload it
to verify its weights/metadata. The completion summary records best update,
best metrics, final metrics, both checkpoint hashes, successful update count,
and protected-file status. Best checkpoint metadata is verified against the
tracked evaluation before completion. The completion marker is written last,
after final integrity checks; interrupted/failed runs are not marked completed.

## Non-training checks

From the project root in PowerShell:

```powershell
.\.venv311\Scripts\python.exe -B experiments\four_image_best_checkpoint\test_four_image.py
.\.venv311\Scripts\python.exe -B experiments\four_image_best_checkpoint\train_four_image.py --validate-only
.\.venv311\Scripts\python.exe -B experiments\four_image_best_checkpoint\train_four_image.py
```

Default invocation prints help. Validation writes nothing and creates no model
or optimizer. Tests use mocks, synthetic tensors, and in-memory checkpoint
serialization; they perform no optimizer updates or disk checkpoint writes.
Bytecode writes are disabled. Tests cover selection and ties, cloned weights,
serialization consistency, publication failure, interruption, update limits,
path escape, external changes, and preservation of numerical training logic.

## Training only after separate approval

```powershell
.\.venv311\Scripts\python.exe -B experiments\four_image_best_checkpoint\train_four_image.py --train
```

Outputs are confined to a new `runs/<UTC timestamp>_<UUID>/`: configuration,
provenance, environment, before/after hashes, history, best and last checkpoints,
and summary. Run collisions fail. The optimizer post-hook counts successful step
returns independently of the loop, and the pre-hook blocks update 801. Completion
requires 800 successes, matching Adam counters, 200 updates per image, and all
41 evaluation records. Protected source files, selected data, training annotations,
prior experiment records and checkpoints are hashed before and after the run.
Discrepancies are reported, never repaired; keep protected files unchanged in
other applications while the run is active.

Matching these **unverified rectangular reference masks** measures training-image
fit, not geological accuracy or generalization. Empty masks are not confirmed
geological negatives. Source-scene IDs and patch coordinates remain unknown.
