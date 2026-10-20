"""CPU-only validation inference against unverified rectangular reference masks.

No training or test-split access. --check-only evaluates without writing reports.
Both-empty Dice/IoU are undefined (N/A), never counted as perfect agreement.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_unet_predictions import ROOT, digest, load_model
from dataset_loader import LunarLandslideDataset
import cv2
import numpy as np
import torch

RUN = ROOT / "experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9"
CHECKPOINT = RUN / "best.pth"
THRESHOLD = 0.5
DISCLAIMER = "Agreement with unverified rectangular reference masks; NOT scientifically verified landslide accuracy."


def validation_dataset():
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        dataset = LunarLandslideDataset("valid", image_size=256)
        dataset.pairs = [(a.resolve(), b.resolve()) for a, b in dataset.pairs]
    finally:
        os.chdir(previous)
    stems = [a.stem.casefold() for a, _ in dataset.pairs]
    if len(stems) != len(set(stems)):
        raise ValueError("Ambiguous validation image stems")
    if len(dataset) != 80:
        raise ValueError(f"Expected 80 validation pairs; found {len(dataset)}")
    return dataset


def overlap(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference shape mismatch")
    p, r = int(prediction.sum()), int(reference.sum())
    intersection = int(np.count_nonzero(prediction & reference))
    union = p + r - intersection
    return {
        "reference_foreground_percent": 100 * r / reference.size,
        "predicted_foreground_percent": 100 * p / prediction.size,
        "dice_unverified": 2 * intersection / (p + r) if p + r else None,
        "iou_unverified": intersection / union if union else None,
        "reference_pixels": r, "predicted_pixels": p,
        "intersection_pixels": intersection, "union_pixels": union,
        "reference_status": "nonempty_unverified" if r else "empty_unverified",
        "metric_status": "defined" if union else "undefined_both_empty",
    }


def aggregate(title, rows):
    defined = [r for r in rows if r["dice_unverified"] is not None]
    lines = [f"{title}: {len(rows)} images",
             f"Defined overlap scores: {len(defined)}; both-empty excluded: {len(rows)-len(defined)}"]
    for field in ("dice_unverified", "iou_unverified"):
        value = f"{np.mean([r[field] for r in defined]):.8f}" if defined else "N/A"
        lines.append(f"Macro mean {field} (equal weight per defined image): {value}")
    for field in ("reference_foreground_percent", "predicted_foreground_percent"):
        value = f"{np.mean([r[field] for r in rows]):.6f}%" if rows else "N/A"
        lines.append(f"Mean {field}: {value}")
    return "\n".join(lines)


def verify_hashes(before):
    changed = [str(path) for path, expected in before.items()
               if not path.is_file() or digest(path) != expected]
    if changed:
        raise RuntimeError(f"Protected file hash discrepancy; no repair attempted: {changed}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Run all validation inference without writing reports")
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        protected = [CHECKPOINT, RUN / "configuration.json"] + [ROOT / name for name in (
            "unet_model.py", "dataset_loader.py", "tools/preview_unet_predictions.py",
            "tools/compare_unet_checkpoints.py",
        )]
        for path in protected:
            before[path] = digest(path)
        config = json.loads((RUN / "configuration.json").read_text(encoding="utf-8"))
        if config["image_size"] != 256 or config["threshold"] != THRESHOLD or config["model"] != "UNet":
            raise ValueError("Experiment E configuration is incompatible with this evaluation")
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)  # Existing UNet, strict CPU load, eval().
        dataset = validation_dataset()  # Never instantiate train or test datasets.
        for image_path, mask_path in dataset.pairs:
            for path in (image_path, mask_path):
                before[path] = digest(path)
        rows = []
        for index, (image_path, mask_path) in enumerate(dataset.pairs):
            raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or not set(np.unique(raw)) <= {0, 255}:
                raise ValueError(f"Invalid grayscale 0/255 reference mask: {mask_path}")
            image, target = dataset[index]  # Includes original dimension checks.
            reference = target[0].numpy() > 0
            if bool(raw.any()) != bool(reference.any()):
                raise ValueError(f"Mask emptiness changed during existing preprocessing: {mask_path}")
            with torch.no_grad():
                logits = model(image.unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f"Invalid output for {image_path}")
                prediction = logits.sigmoid()[0, 0].numpy() >= THRESHOLD
            rows.append({
                "filename": image_path.name, "split": "valid",
                "image_path": str(image_path), "mask_path": str(mask_path),
                "image_sha256": before[image_path], "mask_sha256": before[mask_path],
                "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": before[CHECKPOINT],
                "checkpoint_position": tag, "threshold": THRESHOLD,
                "metric_resolution": "256x256", **overlap(prediction, reference),
                "interpretation": DISCLAIMER,
            })
            if (index + 1) % 20 == 0:
                print(f"Evaluated {index+1}/{len(dataset)} validation images", flush=True)
        nonempty = [r for r in rows if r["reference_pixels"] > 0]
        empty = [r for r in rows if r["reference_pixels"] == 0]
        if (len(nonempty), len(empty)) != (19, 61):
            raise ValueError(f"Expected 19 nonempty/61 empty references; found {len(nonempty)}/{len(empty)}")
        summary = "\n\n".join([
            DISCLAIMER,
            f"Checkpoint: {CHECKPOINT}\nSHA-256: {before[CHECKPOINT]}\nPosition: {tag}\n"
            f"CPU inference only; split=valid; threshold >= {THRESHOLD}; metrics at 256x256.\n"
            "Existing loader: RGB, INTER_AREA resize, float32 / 255; masks INTER_NEAREST then >0.\n"
            "All pairs evaluated in sorted filename order; no training or test data accessed.",
            "Dice = 2*intersection/(predicted+reference pixels); IoU = intersection/union; no smoothing.\n"
            "Both masks empty: N/A, excluded from macro overlap means (not scored as 1).\n"
            "Empty reference with nonempty prediction: Dice/IoU=0 (reference disagreement only).\n"
            "Empty references are NOT confirmed geological negatives. Missing masks cause failure.\n"
            "All-image means include defined empty-reference disagreements and can differ greatly\n"
            "from nonempty-reference means. Coverage averages include every image in each group.",
            aggregate("All validation images", rows),
            aggregate("Nonempty unverified references", nonempty),
            aggregate("Empty unverified references", empty),
            f"Empty-reference images with nonempty predictions: {sum(r['predicted_pixels'] > 0 for r in empty)}/{len(empty)}",
        ]) + "\n"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "N/A" if v is None else v for k, v in row.items()} for row in rows)
        csv_text = buffer.getvalue()
        if len(list(csv.DictReader(io.StringIO(csv_text)))) != 80:
            raise ValueError("CSV serialization failed")
        verify_hashes(before)
        if not args.check_only:
            parent = ROOT / "validation_evaluations"
            if parent.resolve() != parent:
                raise ValueError("Redirected output directory refused")
            parent.mkdir(exist_ok=True)
            output = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output / "per_image_metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                stream.write(csv_text)
            with (output / "summary.txt").open("x", encoding="utf-8") as stream:
                stream.write(summary)
        print(summary)
        print(f"Reports saved: {output}" if output else "Checks passed; no reports written.")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        if output:
            print(f"Partial reports may remain in {output}; no overwrite or repair attempted.", file=sys.stderr)
        return 1
    finally:
        verify_hashes(before)


if __name__ == "__main__":
    raise SystemExit(main())
