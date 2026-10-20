# Controlled exploratory Mars-to-Moon transfer

Existing Experiment I is the historical lunar-only arm A; its best checkpoint is
re-evaluated and matched to saved metrics before training. B uses the identical
fresh seed42 U-Net, 2330 fixed-budget Mars updates (10 passes), then 5180 lunar
updates (140 passes). Lunar settings and evaluation reuse I's implementation.
No existing training code, checkpoints, or datasets are edited.

Mars uses train=465, val=66, RGB only, no augmentation. Lunar uses I's exact 73
nonempty training images and 80 validation images, joint quarter turns only on
training. Both stages use CPU two threads, batch2, Adam .001, existing unweighted
BCE plus smoothed soft Dice. Lunar Adam and sampling/augmentation RNG reset.
Validation is inference-only, without gradients. Threshold remains .5.

Mars is monitored every 233 updates and its final predefined weights are used.
Lunar is evaluated at zero and every140, choosing strict improvement in macro
Dice on the19 nonempty references, retaining earlier ties. This is a fixed budget
with validation monitoring, not an open-ended extension or automatic early stop.
Other coverage/precision/recall metrics are recorded to expose overprediction.

Each evaluation saves a uniquely named immutable state verified by reloading.
best.pth and last.pth are created only at successful completion. A failed run is
marked incomplete. States contain optimizer/model and some RNG states but no
automatic exact mid-epoch resume is implemented; do not claim it is supported.
Outputs remain in a UUID timestamp run. Existing assets are hashed before/after.

Data passed technical pairing, range and interpolation checks. Six saved overlays
show no obvious global flip/transpose/shift, but do NOT establish geology. Image
GeoTIFF tags are absent; mask tags remain. No geographic transform comparison can
be certified. No safe evidence supports altering any alignment; originals remain
unchanged. Published Mars mask semantics also differ from the unverified lunar
rectangles. License records: dataset CC BY4.0, repository MIT; retain attribution.

This comparison has one seed and extra compute in B. Historical A is not a fresh
concurrent run. Reused validation and uncertain annotations limit conclusions.
It can test reference-mask agreement, not geological accuracy or terrain safety.

Commands from project root:

```
.\.venv311\Scripts\python.exe -B experiments/mmlsv2_transfer/test_transfer.py
.\.venv311\Scripts\python.exe -B experiments/mmlsv2_transfer/run_transfer.py
.\.venv311\Scripts\python.exe -B experiments/mmlsv2_transfer/run_transfer.py --train
```

Default performs preflight only; training requires --train. No test split is used.
