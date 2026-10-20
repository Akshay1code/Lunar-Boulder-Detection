# Single-image overfit diagnostic

This experiment measures **training-image fit against an unverified rectangular
reference mask**. It is not validation performance or geological accuracy.
Fitting one image does not establish generalization. Failure within 200 updates
does not prove that the architecture cannot fit the image. Missing labels are
not confirmed negatives. No validation or test split is accessed.

## Fixed design

- Image: `train/ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg`.
- Prior manifest records COCO image ID 142 and landslide annotation ID 743;
  preflight verifies the actual files, hashes, paths, and COCO associations.
- Existing `UNet`, initialized from scratch; seed 42 for Python, NumPy and PyTorch.
- CPU, two threads, deterministic PyTorch operations, batch size 1.
- Existing training dataset loader: RGB, image resize with INTER_AREA to 256x256,
  float32 divided by 255; mask resize with INTER_NEAREST, binary mask > 0.
- Adam, learning rate 0.001; unweighted BCEWithLogitsLoss; threshold 0.5.
- Exactly 200 optimizer updates on the same image, without augmentation,
  early stopping, automatic extension, or loading an existing checkpoint.

The unchanged loader enumerates training pairs, but the runner decodes only the
selected pair. No validation/test dataset is instantiated. The Experiment B
manifest is read for the selected training record; its other split records are
not used. Existing checkpoints are hashed for protection, not loaded.

## Non-training commands

Run from the project root in the VS Code PowerShell terminal:

```powershell
.\.venv311\Scripts\python.exe -B experiments\single_image_overfit\train_single_image.py
.\.venv311\Scripts\python.exe -B experiments\single_image_overfit\test_single_image.py
.\.venv311\Scripts\python.exe -B experiments\single_image_overfit\train_single_image.py --validate-only
```

The default command prints help. Tests use synthetic tensors and mocks, never
real optimizer updates. Preflight checks the selected data and protected files;
it creates no model, optimizer, output directory, or checkpoint. `-B` and the
runner's bytecode setting prevent project bytecode cache writes.

## Training requires separate approval

Do not run this command until the 200-update training run is separately approved:

```powershell
.\.venv311\Scripts\python.exe -B experiments\single_image_overfit\train_single_image.py --train
```

Default and `--validate-only` never train. Configuration changes are rejected.
The script does not itself record conversational approval: `--train` is the
explicit operational gate.

## Update counting and completion

A loop requests updates 1 through 200. An optimizer pre-hook blocks any call
after 200 successful updates; a post-hook independently counts successful
`optimizer.step()` returns. Failed calls do not increment that counter. After
the loop, require exactly 200 successful calls, Adam step counters of 200 for
every parameter, and history entries 0 through 200. Nonfinite losses/gradients
and mismatched counters stop the run. An incomplete run is never resumed.

`summary.json` with status `completed` is written only after successful checks
and output saving. Missing, partial, or unparsable summary means incomplete,
including a process killed before completion. Partial run artifacts may remain
for inspection; they are never overwritten, deleted, or automatically repaired.

## Measurements and outputs

Evaluate the same training image in `model.eval()` and inference mode at update
0 and after every update. History labels pre-update training BCE separately
from post-update evaluation BCE. Record probability min/max/mean/median,
foreground coverage at 0.5, Dice, recall, reference foreground coverage,
successful step count, and elapsed seconds. Update 0 has no training loss.
Coverage values are fractions in [0,1]. Dice/recall measure agreement with the
supplied mask. No lower-threshold tuning or geological classification occurs.

Only an actual authorized run creates a new
`runs/<UTC timestamp>_<UUID>/` directory with:

- `configuration.json`, `selected_image.json`, `environment.json`;
- `protected_hashes_before.json`, `protected_hashes_after.json`;
- `history.jsonl`, final `last.pth`, and completion `summary.json`.

No reference masks, boundary annotations, image copies, or prediction PNGs are
generated. All writes use exclusive creation inside the unique run. Path
escapes, redirected run roots, existing files, and run collisions fail safely.

Selected image/mask, training COCO annotations, prior manifest, relevant source
files, and existing project/experiment `.pth` checkpoints are SHA-256 hashed
before and after execution. Hash discrepancies or missing files are reported
and never repaired. Source files and old checkpoints are only opened for reads.
Before/after hashes detect changes; they cannot prevent another program from
editing a protected file concurrently, so keep these files unchanged during a run.

## Interpretation

Compare the baseline and final training-image fit, not validation performance.
If it fits, the pipeline has demonstrated fitting this one supplied target;
annotation reliability and performance on independent scenes remain unresolved.
If it does not fit, inspect the loss and probability history before deciding on
another separately approved diagnostic. Neither outcome validates lunar
landslide boundaries or geological accuracy.
