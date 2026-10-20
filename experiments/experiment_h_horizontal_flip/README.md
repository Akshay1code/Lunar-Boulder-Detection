# Experiment H: horizontal flips

Infrastructure only; training requires separate user approval and explicit --train.
Baseline: Experiment F. References are unverified rectangular polygons. Metrics
measure reference-mask agreement, not geological accuracy. Empty references do
not establish negatives.

Same fresh UNet, seed 42, CPU 2 threads, 73 selected training images in F order,
256x256 preprocessing, shuffled batch size 2 (last batch 1), Adam lr 0.001,
unweighted BCE + F soft Dice (weights 1+1, smoothing 1, mean per image).
Exactly 5,180 successful optimizer steps = 140 complete passes. Evaluation at
0 and every 140 updates through 5,180, always on unaugmented training images.
No validation/test data is loaded by this runner. Compare validation separately
using the existing evaluator, eval mode and no gradients, at threshold 0.5.

Only training treatment change: independent horizontal flip probability 0.5
for each image presentation, jointly applied to image and mask after existing
preprocessing. Dedicated CPU torch.Generator(seed=42), separate from global RNG
and unchanged shuffle generator. No interpolation, cropping, brightness change,
or vertical flip. Per-batch flip flags are logged in training_history.jsonl.

Best checkpoint retains F's strict macro training Dice improvement rule, with
earlier ties retained; NOT validation-based selection. Deep-copied weights and
metadata are serialized and verified. last.pth is saved separately at completion.

All outputs go to a new runs/UTC-timestamp_UUID directory: configuration,
provenance, package versions, protected hashes before/after, training history,
evaluation history (per-image, aggregate and coverage groups), summary,
best.pth and last.pth. Default invocation only shows help. --validate-only reads
training assets, verifies count/order/COCO masks/hashes and writes no outputs.
Existing run/checkpoint/source hashes are checked before and after execution.
Interrupted runs are reported INCOMPLETE and have no completed summary.

From the project root in PowerShell:

```powershell
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_horizontal_flip/test_experiment_h.py
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_horizontal_flip/train_experiment_h.py --validate-only
```

Only after separate training approval:

```powershell
& .\.venv311\Scripts\python.exe -B experiments/experiment_h_horizontal_flip/train_experiment_h.py --train
```
