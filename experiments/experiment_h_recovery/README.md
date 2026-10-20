# Experiment H recovery infrastructure

No training has been authorized for this implementation task. Default invocation
shows help; --validate-only reads training data and writes no outputs.

References are unverified rectangular polygons. Scores measure training-reference
agreement, not validation performance or geological accuracy. Empty references
are not established negatives. No validation or test data is loaded.

## Fixed experiment

Unchanged H: same 73 selected training images, fresh UNet, seed 42, CPU two
threads, 256x256 preprocessing, shuffled batch 2 with final batch 1, Adam 0.001,
unweighted BCE + per-image soft Dice (smoothing 1, coefficients 1+1), threshold
0.5, 5,180 successful updates / 140 passes. Paired horizontal flip p=0.5 uses a
separate generator. Evaluation on unaugmented training images at update 0 and
every 140 updates. Best selection remains strictly improving macro training
Dice, retaining earlier ties. No gradient clipping or loss changes.

## Recovery

Each scheduled evaluation publishes recovery_update_XXXXXX.pth containing model,
Adam state, successful update number, current pass order and next-batch position,
Python/NumPy/PyTorch RNG states, shuffle/augmentation/evaluation generator states,
configuration, selected-image provenance, history prefix, latest evaluation,
and the best weights and metrics so far. Recovery 005180 is an archival completed
state; resuming it is rejected. best.pth and last.pth are separate.

A private index-only DataLoader materializes one pass using precisely H's sampler
and base-seed draws. Its final RNG state and the unconsumed order are saved.
This changes when the private generator advances, not sample order; tests compare
all 140 passes against the original loader and recovery across pass boundaries.
Images themselves are loaded only for the next batch, keeping memory low.

Recovery uses exclusive staging, load-and-compare verification and Windows
no-overwrite rename. Files remain within a unique runs/UTC_timestamp_UUID folder.
Resuming creates another unique run with its parent checkpoint path/hash and
copied history prefix. Original runs are never appended to or replaced. Outputs
from after the chosen recovery point must be replayed; partial updates are not
claimed completed. The old H update-0 best.pth is NOT a recovery checkpoint.

All existing experiment/source/checkpoint files and training assets are hashed
before and after execution. Configuration/environment/baseline checks fail on
discrepancies. A crash before checkpoint publication leaves the earlier complete
recovery intact. Only a verified 5,180-update run receives a completed summary.

## Diagnostics

Each evaluation logs probability min/max/mean/population standard deviation for
each image and all pixels combined, in the same inference pass. Each optimizer
step logs global L2 gradient norm and per-parameter L2 min/max/mean, before step().
These do not alter probabilities, gradients or training RNGs. Logs do not turn
uncertain references into verified labels.

## Non-training commands (project root, PowerShell)

```powershell
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_recovery/test_experiment_h_recovery.py
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_recovery/train_experiment_h_recovery.py --validate-only
```

After separate training approval only:

```powershell
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_recovery/train_experiment_h_recovery.py --train
# Future recovery, also requires training approval:
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_recovery/train_experiment_h_recovery.py --train --resume "experiments/experiment_h_recovery/runs/<run>/recovery_update_001400.pth"
```

Tests use synthetic tensors, fake optimizer counters and in-memory serialization.
No real optimizer.step() or model-training run is performed. They verify exact
batch/flip/RNG continuation, restored Adam state and identical subsequent forward
outputs; bitwise equivalence of full resumed training is not empirically tested
without training. Reproducibility requires the same pinned software/hardware.
