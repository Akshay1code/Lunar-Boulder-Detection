"""Compare three checkpoints on six training images; inference only.

Metrics measure agreement with unverified rectangular references, not geological
accuracy. --check-only verifies inference and PNG rendering without writing files.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_unet_predictions import (
    ROOT, THRESHOLD, comparison, digest, load_model, select_images,
    training_dataset,
)
import cv2
import numpy as np
import torch
from PIL import Image

CHECKPOINTS = {
    "Original": ROOT / "checkpoints/unet_best.pth",
    "Experiment E": ROOT / "experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9/best.pth",
    "Experiment F": ROOT / "experiments/experiment_f_dice_loss/runs/20261014T102910Z_262e760875cd4383a83f9d45a975d403/best.pth",
}
DISCLAIMER = "Training-image agreement with unverified rectangular reference masks; not proof of geological accuracy."


def measure(probabilities, reference):
    """Binary Dice without smoothing; all references must be nonempty."""
    if probabilities.shape != reference.shape or not reference.any():
        raise ValueError("Expected matching 256x256 prediction and nonempty reference")
    predicted = probabilities >= THRESHOLD
    dice = 2 * np.count_nonzero(predicted & reference) / (
        np.count_nonzero(predicted) + np.count_nonzero(reference))
    return {
        "predicted_foreground_percent": float(predicted.mean() * 100),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "reference_foreground_percent": float(reference.mean() * 100),
        "dice_unverified_reference": float(dice),
    }, predicted.astype(np.uint8) * 255


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    before, run = {}, None
    try:
        torch.set_num_threads(2)
        dataset = training_dataset()  # Only train; uses the existing loader unchanged.
        indices = select_images(dataset, 6)  # Same selection as preview_unet_predictions.py.
        if len(indices) != 6:
            raise ValueError(f"Expected exactly six nonempty pairs; found {len(indices)}")
        protected = list(CHECKPOINTS.values()) + [
            ROOT / "unet_model.py", ROOT / "dataset_loader.py",
            ROOT / "predict_image.py", ROOT / "tools/preview_unet_predictions.py",
        ]
        protected += [path for index in indices for path in dataset.pairs[index]]
        for path in protected:
            if not path.is_file():
                raise FileNotFoundError(f"Missing required file: {path}")
            before[path] = digest(path)
        rows, predictions, tags = [], {}, {}
        for label, checkpoint in CHECKPOINTS.items():
            model, tag = load_model(checkpoint)  # Strict state loading, CPU, eval().
            tags[label] = tag
            for index in indices:
                image_path, mask_path = dataset.pairs[index]
                tensor, target = dataset[index]
                with torch.no_grad():
                    logits = model(tensor.unsqueeze(0))
                    if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                        raise ValueError(f"Invalid model output: {label}, {image_path}")
                    probabilities = logits.sigmoid()[0, 0].numpy()
                metrics, binary = measure(probabilities, target[0].numpy() > 0)
                predictions[label, index] = binary
                rows.append({
                    "checkpoint_label": label, "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": before[checkpoint], "checkpoint_position": tag,
                    "split": "train", "filename": image_path.name,
                    "image_path": str(image_path), "mask_path": str(mask_path),
                    "image_sha256": before[image_path], "mask_sha256": before[mask_path],
                    "threshold": THRESHOLD, "metric_resolution": "256x256",
                    **metrics, "interpretation": DISCLAIMER,
                })
                print(f"{label} | {image_path.name} | foreground {metrics['predicted_foreground_percent']:.3f}%"
                      f" | probability min/max/mean {metrics['probability_min']:.6f}/"
                      f"{metrics['probability_max']:.6f}/{metrics['probability_mean']:.6f}"
                      f" | reference {metrics['reference_foreground_percent']:.3f}%"
                      f" | Dice (unverified) {metrics['dice_unverified_reference']:.6f}")
            del model
        if not args.check_only:
            parent = ROOT / "checkpoint_comparisons"
            if parent.resolve() != parent:
                raise ValueError("Redirected output directory refused")
            parent.mkdir(exist_ok=True)
            run = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex)
            run.mkdir(exist_ok=False)
        for number, index in enumerate(indices, 1):
            image_path, mask_path = dataset.pairs[index]
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            reference = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if original is None or reference is None or original.shape[:2] != reference.shape:
                raise ValueError(f"Unreadable or incompatible pair: {image_path}")
            canvas = Image.new("RGB", (1330, 465 * 3), "white")
            for row_number, label in enumerate(CHECKPOINTS):
                panel = comparison(original, reference, predictions[label, index],
                                   image_path.name, label + " / " + CHECKPOINTS[label].name, tags[label])
                canvas.paste(panel, (0, 465 * row_number))
                panel.close()
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG")
            canvas.close()
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.load()
                if decoded.size != (1330, 1395):
                    raise ValueError("Incorrect comparison PNG size")
            if run:
                with (run / f"{number:02d}_{image_path.stem}_comparison.png").open("xb") as stream:
                    stream.write(buffer.getvalue())
        if len(rows) != 18:
            raise ValueError("Expected 18 metric records")
        if run:
            with (run / "comparison_metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved 6 comparison PNGs and 18 CSV records: {run}")
        else:
            print("Checks passed: 18 inferences, 6 PNGs rendered in memory; no outputs written.")
        print(DISCLAIMER)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        if run:
            print(f"Incomplete outputs may remain in {run}; no overwrite or repair attempted.", file=sys.stderr)
        return 1
    finally:
        changed = [str(path) for path, expected in before.items()
                   if not path.is_file() or digest(path) != expected]
        if changed:
            raise RuntimeError(f"Protected file hash discrepancy; no repair attempted: {changed}")


if __name__ == "__main__":
    raise SystemExit(main())
