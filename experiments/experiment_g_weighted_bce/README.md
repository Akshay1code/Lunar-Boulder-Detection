# Experiment G — infrastructure only until separately approved

Training objective: agreement with UNVERIFIED rectangular references, not geological accuracy.
Empty references are not confirmed geological negatives.

The runner imports F without editing it. It reuses its validated 73-image selection,
seed 42, fresh UNet initialization, shuffled batch-2 loader, CPU threads=2,
Adam lr=0.001, 256x256 preprocessing, and soft Dice implementation.
The only optimization change is weighted BCE replacing ordinary BCE in the loss:

`loss = mean(weighted_BCEWithLogits(logits, target, pos_weight=5.521724)) + mean(soft_Dice_loss)`.

The positive weight is round(4050560/733568, 6), checked again using training masks.
F's Dice formula remains `1-(2*sum(sigmoid(z)*y)+1)/(sum(sigmoid(z))+sum(y)+1)`,
computed per image, then averaged across the batch. Ordinary BCE, weighted BCE,
Dice loss, and total loss are logged separately. Weighting also changes BCE's
scale relative to Dice; both component coefficients remain 1.

Evaluation is added on 80 validation images, including 19 nonempty and 61
empty UNVERIFIED references. It cannot backpropagate: inference mode, separate
loader and RNG, no optimizer calls. Evaluation runs at 0 and every 140 updates
through 5180 (38 evaluations). Both-empty overlap is undefined; defined-only
macro means and defined counts are recorded. Threshold stays 0.5.

Each run exclusively creates `runs/<UTC timestamp>_<uuid>/`. It saves
`best_training.pth`, `best_validation.pth`, `last.pth`, configuration, provenance,
environment, histories containing per-image evaluation results, hashes, and summary.
Best training uses macro training Dice; best validation uses nonempty validation
macro Dice. Strict improvements only; earliest ties retained. Deep-copied weights
are saved with the matching evaluation and verified after serialization.
Only checkpoints owned by that newly created run may be replaced on improvement.
Existing experiments/checkpoints are hashed and never overwritten.

The update counter hooks successful optimizer calls directly and blocks step 5181.
Interrupted/failed runs get an incomplete record, never a completed summary.
Completion requires 5180 updates, 140 full passes and all 38 evaluations.

From project root, non-training commands:

```powershell
& .\.venv311\Scripts\python.exe -B experiments\experiment_g_weighted_bce\test_experiment_g.py
& .\.venv311\Scripts\python.exe -B experiments\experiment_g_weighted_bce\train_experiment_g.py --validate-only
```

Default invocation only prints help. After separate training approval, append
`--train` to the runner command. Do not launch training before that approval.
