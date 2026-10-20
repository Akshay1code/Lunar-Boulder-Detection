# Four-image overfit diagnostic

Phase 1 creates infrastructure only. Training requires separate authorization.
All metrics describe **training-image fit against unverified rectangular reference
masks**, not validation performance or geological accuracy. Missing labels are
not confirmed geological negatives. Validation/test splits are never instantiated
or read. This is a fitting diagnostic, not a representative dataset benchmark.

## Fixed selection

The four training images are deliberately chosen to retain the successful
single-image example and add small, intermediate, and large reference foreground
fractions. Selection uses existing mask coverage, not model predictions. Different
filename prefixes do not verify independent source scenes. Source-scene IDs and
patch coordinates remain unknown; none are invented.

| Training COCO ID | Filename prefix (full filename in config) | Original foreground | 256x256 foreground |
|---|---|---:|---:|
| 142 | ohr_000_patch_0154 | 33.325195% | 33.325195% |
| 263 | ohr_019_patch_2428 | 2.753174% | 2.728271% |
| 245 | ohr_001_patch_1715 | 13.309570% | 13.336182% |
| 96 | ohr_006_patch_1816 | 52.968994% | 53.013611% |

`config.json` records exact IDs, filenames, paths, COCO annotation IDs, SHA-256
hashes calculated from actual image/mask bytes, and both coverage values.
Preflight compares these records with Experiment B's saved training manifest and
the actual training COCO annotations. No fallback selection is allowed.

Preflight checks nonempty 0/255 masks, image/mask dimensions, distinct image
hashes, and the original mask naming rule `<image stem>_mask.png`. It reproduces
`create_masks.py` polygon filling **only in memory for equality checking**:
landslide categories only, valid list polygons, float32 coordinates, NumPy rounding
to int32, and OpenCV fillPoly value 255 on zero background. It never imports or
executes `create_masks.py`, which otherwise processes additional splits and writes
masks. No reference mask is created on disk or altered. A discrepancy stops the run.

## Fixed experiment

Fresh existing UNet; existing dataset-loader preprocessing; RGB float32/255,
256x256 INTER_AREA image resize, INTER_NEAREST mask resize and mask > 0.
Seed 42 for Python/NumPy/PyTorch, CPU with two threads, deterministic PyTorch
operations, batch size 1, Adam learning rate 0.001, unweighted BCEWithLogitsLoss.
No checkpoint initialization, augmentation, threshold tuning, or early stopping.

The fixed round-robin order is 142, 263, 245, 96, repeated 100 times. Exactly
400 successful optimizer updates give each image exactly 100 updates. This is
less per-image exposure than the prior 200-update single-image run; failure at
this budget alone does not establish inability to fit multiple images.

The unchanged loader lists training pairs; only the four selected pairs are
decoded. The prior manifest is read for training records only. No validation or
test images, masks, or COCO annotation files are opened.

## Infrastructure checks (no training)

From the project root in PowerShell:

```powershell
.\.venv311\Scripts\python.exe -B experiments\four_image_overfit\test_four_image.py
.\.venv311\Scripts\python.exe -B experiments\four_image_overfit\train_four_image.py --validate-only
.\.venv311\Scripts\python.exe -B experiments\four_image_overfit\train_four_image.py
```

Default invocation prints help. `--validate-only` checks actual records, hashes,
associations and preprocessing without constructing a model or optimizer or
creating output directories. Tests use synthetic tensors and mocked optimizer
calls/output writes, never real training. Bytecode writes are disabled.

## Separately authorized training only

Only after training is separately approved:

```powershell
.\.venv311\Scripts\python.exe -B experiments\four_image_overfit\train_four_image.py --train
```

The flag is the operational gate; the program does not store conversational
approval. All fixed configuration fields and selected IDs/order are checked.

## Metrics and outputs

At update 0 and every 20 successful updates through 400, evaluate all four
training images in eval/inference mode, always threshold >= 0.5. Save per-image
BCE, Dice, recall, precision, predicted foreground coverage, reference foreground
coverage, and probability min/max/mean. Aggregate metrics are arithmetic image
means. Precision is JSON null when no positives are predicted. Macro means exclude
undefined values and report the number of defined images; all undefined means
remain null. Coverage values in JSON are fractions, not percentages.

An actual training run creates only a new `runs/<UTC timestamp>_<UUID>/` with:

- `configuration.json`, `provenance.json`, `environment.json`;
- `protected_hashes_before.json`, `protected_hashes_after.json`;
- `history.jsonl` (21 evaluations, each containing four per-image rows);
- final `last.pth`, and completion `summary.json`.

Source files, selected data, training annotations, previous experiment records
and checkpoints are hashed before/after execution. Changes or missing files are
reported, never repaired. No existing files are overwritten; exclusive creation
is used for all outputs. Run collisions and resolved output path escapes fail.
Keep protected files unchanged in other applications while a run is active.

An optimizer pre-hook blocks update 401; a post-hook independently counts
successful optimizer.step returns. A failed call is not counted. Completion
requires 400 successes, Adam step counters of 400 for every parameter, 100
updates per image, all 21 evaluations, and passing final file-integrity checks.
The completion summary is written last. Missing/unparseable completion summary
means incomplete, including abrupt interruption. Partial run outputs are left
for inspection and never resumed or automatically deleted.

## Interpretation

Compare per-image trajectories as well as averages. Success establishes only that
the pipeline can fit these four supplied rectangular targets. It does not verify
geological boundaries, source-scene independence, or generalization. Failure
within this budget leaves optimization, exposure, imbalance and annotation
limitations unresolved; it is not evidence by itself of a software bug.
