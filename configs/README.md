# Dataset Configurations

This directory contains YAML configurations for training and evaluating YOLO models on Lunar surface imagery.

| Configuration | Classes | Purpose | Usage Command |
|---|---|---|---|
| `data.yaml` | `crater`, `rille` (2 classes) | Baseline ISRO dataset split | `python scripts/train.py --data configs/data.yaml` |
| `data_merged.yaml` | `crater`, `rille`, `boulder` (3 classes) | Merged crater/rille + Roboflow boulder dataset | `python scripts/train.py --data configs/data_merged.yaml --name v3` |
| `data_boulder_only.yaml` | `boulder` (1 class) | Specialized single-class boulder model | `python scripts/train.py --data configs/data_boulder_only.yaml` |
| `data_eval_craters_rilles.yaml` | `crater`, `rille` (evaluated) | Benchmark crater/rille evaluation on test split | `python scripts/evaluate.py --data configs/data_eval_craters_rilles.yaml` |
| `data_eval_boulders.yaml` | `boulder` (evaluated) | Benchmark boulder detection evaluation | `python scripts/evaluate.py --data configs/data_eval_boulders.yaml` |

> **Note:** `data.yaml` is also symlinked/mirrored at the project root for seamless standard YOLO compatibility.
