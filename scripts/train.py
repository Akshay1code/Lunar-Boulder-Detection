"""
Training pipeline for Lunar Crater/Rille Detection with YOLO11 (n/s/m selectable via --model).

Features:
- Selectable YOLO11 backbone (--model, default yolo11s.pt)
- Custom epoch callbacks for validation metric tracking
- Automated epoch_metrics.csv logging (Precision, Recall, mAP50, mAP50-95)
- Automated recall_per_epoch.png visualization update
- Per-epoch confusion matrix generation (confusion_matrices/confusion_matrix_epoch_XX.png)
- Dedicated epoch result folders (epoch1, epoch2, ...) for modular reporting
- Preserves checkpointing (weights/best.pt, weights/last.pt), early stopping, and standard logs.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for reliable server/batch execution
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def extract_validation_metrics(trainer) -> tuple[float, float, float, float]:
    """
    Extract precision, recall, mAP50, and mAP50-95 from validator or trainer state
    without triggering a redundant second validation pass.
    """
    validator = getattr(trainer, "validator", None)
    val_metrics = getattr(validator, "metrics", None)
    box = getattr(val_metrics, "box", None)

    if box is not None:
        try:
            p = float(getattr(box, "mp", 0.0))
            r = float(getattr(box, "mr", 0.0))
            map50 = float(getattr(box, "map50", 0.0))
            map50_95 = float(getattr(box, "map", 0.0))
            return p, r, map50, map50_95
        except Exception as e:
            print(f"[Callback] Note: Box metrics retrieval notice: {e}")

    # Fallback to trainer.metrics dictionary if box object is unavailable
    trainer_metrics = getattr(trainer, "metrics", {}) or {}
    p = float(trainer_metrics.get("metrics/precision(B)", 0.0))
    r = float(trainer_metrics.get("metrics/recall(B)", 0.0))
    map50 = float(trainer_metrics.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(trainer_metrics.get("metrics/mAP50-95(B)", 0.0))
    return p, r, map50, map50_95


def update_recall_chart(csv_path: Path, chart_path: Path) -> None:
    """
    Generate or update the recall_per_epoch.png chart showing validation recall vs epoch.
    """
    if not csv_path.exists():
        return

    epochs = []
    recalls = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                recalls.append(float(row["recall"]))
            except (ValueError, KeyError):
                continue

    if not epochs:
        return

    # Deduplicate epochs keeping the latest metric value in case of restarts
    metric_map = {}
    for ep, rec in zip(epochs, recalls):
        metric_map[ep] = rec
    epochs = sorted(metric_map.keys())
    recalls = [metric_map[ep] for ep in epochs]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(epochs, recalls, marker="o", color="#005f9c", linewidth=2.2, label="Validation Recall")
    if len(epochs) <= 30:  # per-point labels only stay readable on short runs
        for ep, rec in zip(epochs, recalls):
            plt.annotate(
                f"{rec:.3f}",
                (ep, rec),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                fontweight="bold"
            )
    plt.title("Lunar Detection - Validation Recall vs Epoch", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Recall (all classes)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    if len(epochs) <= 30:
        plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend(loc="lower right")
    plt.tight_layout()

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=200)
    plt.close()


def save_epoch_confusion_matrix(trainer, epoch: int, cm_dir: Path) -> Path:
    """
    Extract and save the current epoch's confusion matrix from validator state.
    """
    cm_dir.mkdir(parents=True, exist_ok=True)
    epoch_cm_path = cm_dir / f"confusion_matrix_epoch_{epoch:02d}.png"

    validator = getattr(trainer, "validator", None)
    cm = getattr(validator, "confusion_matrix", None)
    if cm is None and validator is not None:
        cm = getattr(getattr(validator, "metrics", None), "confusion_matrix", None)

    generated = False
    if cm is not None:
        try:
            # Use Ultralytics ConfusionMatrix plot method
            temp_dir = cm_dir / f"_tmp_epoch_{epoch}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            cm.plot(normalize=False, save_dir=temp_dir)
            tmp_cm = temp_dir / "confusion_matrix.png"
            if tmp_cm.exists() and tmp_cm.stat().st_size > 0:
                shutil.copy2(tmp_cm, epoch_cm_path)
                # Also save normalized version if available
                cm.plot(normalize=True, save_dir=temp_dir)
                tmp_norm = temp_dir / "confusion_matrix_normalized.png"
                if tmp_norm.exists():
                    shutil.copy2(tmp_norm, cm_dir / f"confusion_matrix_normalized_epoch_{epoch:02d}.png")
                generated = True
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[Callback] Note: cm.plot attempt ({e}), using direct matplotlib fallback.")

    # Matplotlib fallback directly rendering cm.matrix
    if not generated and cm is not None and getattr(cm, "matrix", None) is not None:
        matrix = cm.matrix
        names = getattr(cm, "names", None) or getattr(validator, "names", None)
        if isinstance(names, dict):
            class_names = list(names.values())
        elif isinstance(names, list):
            class_names = list(names)
        else:
            class_names = [f"class_{i}" for i in range(matrix.shape[0] - 1)]
        labels = class_names + ["background"] if len(class_names) < matrix.shape[0] else class_names

        plt.figure(figsize=(7, 6))
        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix (Epoch {epoch})", fontsize=13, fontweight="bold")
        plt.colorbar()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)

        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                plt.text(
                    j, i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black"
                )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(epoch_cm_path, dpi=200)
        plt.close()

    return epoch_cm_path


def save_epoch_result_folders(
    run_dir: Path,
    epoch: int,
    metrics_payload: dict,
    epoch_cm_path: Path
) -> None:
    """
    Save epoch results into dedicated folders (epoch1, epoch2, ...) in:
    1. Training run directory: run_dir / "epoch_results" / f"epoch{epoch}"
    2. Reports directory: ROOT / "reports" / "ultralytics_results" / run_dir.name / f"epoch{epoch}"
    """
    target_folders = [
        run_dir / "epoch_results" / f"epoch{epoch}",
        ROOT / "reports" / "ultralytics_results" / run_dir.name / f"epoch{epoch}"
    ]

    for folder in target_folders:
        folder.mkdir(parents=True, exist_ok=True)

        # Copy confusion matrix
        if epoch_cm_path and epoch_cm_path.exists():
            shutil.copy2(epoch_cm_path, folder / epoch_cm_path.name)
            shutil.copy2(epoch_cm_path, folder / "confusion_matrix.png")
            norm_cm = epoch_cm_path.parent / f"confusion_matrix_normalized_epoch_{epoch:02d}.png"
            if norm_cm.exists():
                shutil.copy2(norm_cm, folder / norm_cm.name)
                shutil.copy2(norm_cm, folder / "confusion_matrix_normalized.png")

        # Copy recall chart
        recall_chart = run_dir / "recall_per_epoch.png"
        if recall_chart.exists():
            shutil.copy2(recall_chart, folder / "recall_per_epoch.png")

        # Save metrics JSON
        with open(folder / "metrics.json", "w", encoding="utf-8") as jf:
            json.dump(metrics_payload, jf, indent=2)


def on_val_start(validator) -> None:
    """
    Ensure validator.args.plots is enabled so that the confusion matrix
    processes batches on every validation epoch.
    """
    if hasattr(validator, "args"):
        validator.args.plots = True


def on_train_epoch_end(trainer) -> None:
    """
    Ensure validator.args.plots is enabled prior to running validation each epoch.
    """
    if hasattr(trainer, "validator") and hasattr(trainer.validator, "args"):
        trainer.validator.args.plots = True


def on_fit_epoch_end(trainer) -> None:
    """
    Ultralytics training callback executed at the end of every epoch (after validation).
    Extracts metrics, appends to CSV, updates recall curve, saves confusion matrix,
    and creates epoch-specific result folders without a second validation pass.
    """
    epoch = trainer.epoch + 1  # 1-indexed epoch

    # Ultralytics calls on_fit_epoch_end in final_eval with epoch > epochs. Skip duplicate final eval trigger.
    if epoch > trainer.epochs:
        return

    run_dir = Path(trainer.save_dir)
    cm_dir = run_dir / "confusion_matrices"

    # 1. Extract validation metrics
    p, r, map50, map50_95 = extract_validation_metrics(trainer)

    # 2. Record metrics in epoch_metrics.csv (clean reset on epoch 1, append otherwise)
    csv_path = run_dir / "epoch_metrics.csv"
    mode = "w" if epoch == 1 else "a"
    write_header = (epoch == 1) or (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with open(csv_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writerow([epoch, f"{p:.5f}", f"{r:.5f}", f"{map50:.5f}", f"{map50_95:.5f}"])

    # 3. Update recall_per_epoch.png
    update_recall_chart(csv_path, run_dir / "recall_per_epoch.png")

    # 4. Save per-epoch confusion matrix
    epoch_cm_path = save_epoch_confusion_matrix(trainer, epoch, cm_dir)

    # 5. Populate separate epoch folders (epoch1, epoch2, ...)
    metrics_payload = {
        "epoch": epoch,
        "precision": p,
        "recall": r,
        "mAP50": map50,
        "mAP50-95": map50_95
    }

    # On epoch 1 of a fresh run, clean any stale leftover epoch folders from previous runs
    if epoch == 1:
        epoch_results_folder = run_dir / "epoch_results"
        if epoch_results_folder.exists():
            for old_dir in epoch_results_folder.glob("epoch*"):
                if old_dir.is_dir() and old_dir.name != "epoch1":
                    shutil.rmtree(old_dir, ignore_errors=True)

    save_epoch_result_folders(run_dir, epoch, metrics_payload, epoch_cm_path)

    print(
        f"\n[Epoch {epoch} Callback Complete] "
        f"Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}"
    )


def train_model(
    epochs: int = 10,
    batch: int = 8,
    imgsz: int = 640,
    device_arg: str | None = None,
    resume: bool = False,
    name: str = "v2",
    model_name: str = "yolo11s.pt",
    patience: int = 30,
    data: str = "data.yaml",
):
    """
    Execute YOLO11n training with custom validation epoch tracking.
    """
    # Hardware & Precision Configuration
    if device_arg is not None:
        device = int(device_arg) if device_arg.isdigit() else device_arg
        amp = (device == 0 or device == "0" or "cuda" in str(device)) and torch.cuda.is_available()
    elif torch.cuda.is_available():
        device = 0
        amp = True
        print(f"[Hardware] CUDA GPU available: {torch.cuda.get_device_name(0)}. Using device=0, AMP=True.")
    else:
        device = "cpu"
        amp = False
        print(
            f"[Hardware] Notice: Current PyTorch environment is CPU-only (torch=={torch.__version__}). "
            f"Using device='cpu', AMP=False."
        )

    print(f"\n{'='*70}")
    print(f"Starting YOLO Training ({model_name}): {epochs} Epochs, Batch={batch}, ImgSz={imgsz}, Device={device}, AMP={amp}, Run Name={name}")
    print(f"{'='*70}\n")

    # Initialize model (or load checkpoint if resuming)
    last_ckpt = ROOT / f"runs/lunar_detection/{name}/weights/last.pt"
    if resume and last_ckpt.exists():
        print(f"[Resume] Resuming training from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        weights_candidate = ROOT / "weights" / model_name
        if weights_candidate.exists():
            model = YOLO(str(weights_candidate))
        else:
            model = YOLO(model_name)

    # Register callbacks
    model.add_callback("on_val_start", on_val_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Resolve dataset yaml path (check root and configs/)
    data_path = Path(data) if Path(data).is_absolute() else (ROOT / data)
    if not data_path.exists() and (ROOT / "configs" / data).exists():
        data_path = ROOT / "configs" / data

    # Launch training
    train_kwargs = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "amp": amp,
        "project": str(ROOT / "runs/lunar_detection"),
        "name": name,
        "exist_ok": True,
        "save": True,      # Preserves weights/best.pt and weights/last.pt
        "plots": True,     # Preserves standard training curves and logs
        # --- Accuracy improvements ---
        "patience": patience,   # early stopping on val fitness
        "cos_lr": True,         # cosine LR decay -> smoother, more stable late epochs
        "close_mosaic": 15,     # last N epochs on clean (non-mosaic) images
        "seed": 0,              # reproducible runs for fair comparison
        # Lunar imagery has no fixed "up", so vertical flips are valid.
        # No rotation: rotating thin diagonal rille boxes inflates axis-aligned boxes.
        "flipud": 0.5,
        "fliplr": 0.5,
        "scale": 0.5,
        "hsv_v": 0.4,           # illumination variation
    }
    if resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11n on Lunar Crater/Rille Dataset")
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum epochs to train (default: 150; early stopping via --patience)"
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0' or 'cpu')")
    parser.add_argument("--name", type=str, default="v2", help="Run name / output subfolder (default: 'v2'; 'baseline' is your old run, don't reuse it)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Starting weights, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt (default: yolo11s.pt)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Dataset YAML relative to project root (e.g. data_merged.yaml for crater+rille+boulder)")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience in epochs (default: 30)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device_arg=args.device,
        resume=args.resume,
        name=args.name,
        model_name=args.model,
        patience=args.patience,
        data=args.data,
    )