# Model Weights

This directory holds starting backbone weights and trained checkpoints for Lunar crater, rille, and boulder detection.

## Available Starting Backbones
- `yolo11n.pt` - YOLO11 Nano (5.6 MB), optimized for lightweight inference.
- `yolo11s.pt` - YOLO11 Small (19.3 MB), default recommended balance of speed and recall.

## Trained Checkpoints
When you train a model using `python scripts/train.py --name v3`:
Checkpoints are saved under `runs/lunar_detection/<run_name>/weights/best.pt` and `last.pt`.
You can copy top-performing checkpoints here (e.g. `weights/best.pt`) for evaluation and deployment.
