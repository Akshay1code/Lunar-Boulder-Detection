"""Isolated Experiment B. No training unless --train is explicitly supplied."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import uuid

# Do not create bytecode caches in the original project when importing its modules.
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset_loader import LunarLandslideDataset
from unet_model import UNet

LABEL = "Agreement with unverified rectangular reference masks"
ORIGINAL_CHECKPOINT = PROJECT_ROOT / "checkpoints/unet_best.pth"
FIXED = {
    "seed": 42, "device": "cpu", "cpu_threads": 2, "image_size": 256,
    "batch_size": 2, "drop_last": False, "num_workers": 0,
    "model": "UNet", "initialization": "from_scratch", "optimizer": "Adam",
    "learning_rate": 0.001, "loss": "BCEWithLogitsLoss", "pos_weight": None,
    "threshold": 0.5, "max_updates": 700, "validate_every": 140,
    "expected_train_total": 279, "expected_train_nonempty": 73,
    "expected_valid_total": 80, "expected_valid_nonempty": 19,
    "expected_valid_empty": 61,
    "checkpoint_selection": "nonempty_macro_dice_then_lower_nonempty_bce_then_earlier_update",
    "undefined_metrics": None, "empty_empty_dice": None,
    "empty_masks_are_confirmed_negatives": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config():
    config = json.loads((EXPERIMENT_ROOT / "config.json").read_text(encoding="utf-8"))
    for key, value in FIXED.items():
        require(key in config and config[key] == value, f"Unapproved configuration: {key}")
    require(config.get("description") == LABEL, "Required metric disclaimer missing")
    return config


def make_dataset(split, image_size):
    require(split in ("train", "valid"), "Only train and valid may be accessed")
    # The existing loader uses a relative mask root. Resolve its pairs while in
    # the project directory, then restore cwd so callers can run from anywhere.
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        dataset = LunarLandslideDataset(split, image_size=image_size)
        dataset.pairs = [(a.resolve(), b.resolve()) for a, b in dataset.pairs]
        dataset.image_dir = dataset.image_dir.resolve()
    finally:
        os.chdir(previous)
    return dataset


def checked_selection(nonempty, expected_total, expected_nonempty, split):
    require(len(nonempty) == expected_total,
            f"{split}: expected {expected_total} images, found {len(nonempty)}")
    selected = [i for i, value in enumerate(nonempty) if value]
    require(len(selected) == expected_nonempty,
            f"{split}: expected {expected_nonempty} nonempty masks, found {len(selected)}")
    return selected


def inspect_dataset(dataset, split, config):
    require(split in ("train", "valid"), "Disallowed dataset split")
    annotation_path = dataset.image_dir / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_info = {i["file_name"]: i for i in data["images"]}
    require(len(image_info) == len(data["images"]), "Duplicate COCO filenames")
    require(len({i["id"] for i in data["images"]}) == len(image_info), "Duplicate COCO image IDs")
    require(len({a["id"] for a in data["annotations"]}) == len(data["annotations"]), "Duplicate annotation IDs")
    land_ids = {c["id"] for c in data["categories"] if c["name"].lower() == "landslide"}
    require(bool(land_ids), "Missing landslide category")
    require({p.name for p, _ in dataset.pairs} == set(image_info), "COCO/loader image list differs")
    by_id = {i["id"]: [] for i in data["images"]}
    for a in data["annotations"]:
        require(a["image_id"] in by_id, "Orphan COCO annotation")
        by_id[a["image_id"]].append(a)
    records, nonempty = [], []
    for index, (image_path, mask_path) in enumerate(dataset.pairs):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(mask is not None, f"Unreadable mask: {mask_path}")
        require(set(np.unique(mask).tolist()) <= {0, 255}, f"Unexpected mask values: {mask_path}")
        info = image_info[image_path.name]
        require(mask.shape == (info["height"], info["width"]), "COCO/mask dimensions differ")
        image_tensor, target = dataset[index]  # Exercise unchanged preprocessing.
        size = config["image_size"]
        require(tuple(image_tensor.shape) == (3, size, size) and tuple(target.shape) == (1, size, size), "Tensor shape mismatch")
        require(image_tensor.dtype == target.dtype == torch.float32, "Tensor dtype mismatch")
        require(bool(torch.isfinite(image_tensor).all()) and 0 <= image_tensor.min() <= image_tensor.max() <= 1, "Invalid normalized image")
        require(set(torch.unique(target).tolist()) <= {0.0, 1.0}, "Nonbinary target")
        present = bool(np.any(mask > 0))
        require(present == bool(torch.any(target > 0)), "Resize changes empty/nonempty status")
        annotations = by_id[info["id"]]
        lands = [a["id"] for a in annotations if a["category_id"] in land_ids]
        require(present == bool(lands), "Mask presence disagrees with COCO labels")
        nonempty.append(present)
        records.append({
            "image_key": f"{split}/{image_path.name}", "original_split": split,
            "coco_image_id": info["id"], "annotation_ids": [a["id"] for a in annotations],
            "landslide_annotation_ids": lands, "image_path": str(image_path),
            "mask_path": str(mask_path), "image_sha256": sha256(image_path),
            "mask_sha256": sha256(mask_path), "reference_nonempty": present,
            "selected_for_training": split == "train" and present,
            "label_status": "unverified_rectangular_reference" if present else "no_landslide_label_not_confirmed_negative",
        })
    selected = checked_selection(nonempty, config[f"expected_{split}_total"],
                                 config[f"expected_{split}_nonempty"], split)
    if split == "valid":
        require(len(nonempty) - len(selected) == config["expected_valid_empty"], "Validation empty count differs")
    return records, selected, {str(annotation_path): sha256(annotation_path)}


def prepare(config):
    protected = {str(ORIGINAL_CHECKPOINT): sha256(ORIGINAL_CHECKPOINT)}
    for path in [PROJECT_ROOT / n for n in ("dataset_loader.py", "unet_model.py", "train_unet.py", "predict_image.py")]:
        protected[str(path)] = sha256(path)
    datasets, manifest, selections = {}, {}, {}
    for split in ("train", "valid"):
        datasets[split] = make_dataset(split, config["image_size"])
        records, selections[split], annotation_hash = inspect_dataset(datasets[split], split, config)
        manifest[split] = records
        protected.update(annotation_hash)
        for r in records:
            protected[r["image_path"]] = r["image_sha256"]
            protected[r["mask_path"]] = r["mask_sha256"]
    keys = [r["image_key"] for rows in manifest.values() for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate image keys")
    return datasets, manifest, selections, protected


def verify_unchanged(protected):
    for path, digest in protected.items():
        require(sha256(path) == digest, f"Protected file changed: {path}")


def metrics(tp, fp, fn, pixels):
    require(pixels > 0, "Metric pixel count must be positive")
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "predicted_foreground_coverage": (tp + fp) / pixels,
    }


def aggregate(rows):
    require(bool(rows), "Cannot aggregate an empty validation group")
    tp, fp, fn, pixels = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "pixels"))
    means, counts = {}, {}
    for key in ("dice", "recall", "precision", "predicted_foreground_coverage"):
        values = [r[key] for r in rows if r[key] is not None]
        counts[key] = len(values)
        means[key] = sum(values) / len(values) if values else None
    return {"images": len(rows), "macro": means, "macro_valid_counts": counts,
            "pooled": metrics(tp, fp, fn, pixels), "mean_bce": sum(r["bce"] for r in rows) / len(rows),
            "fraction_images_predicting_foreground": sum(r["tp"] + r["fp"] > 0 for r in rows) / len(rows)}


def evaluate(model, loader, records, threshold):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            require(logits.shape == targets.shape and bool(torch.isfinite(logits).all()), "Invalid validation logits")
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            predictions = torch.sigmoid(logits) >= threshold
            for prediction, target, loss in zip(predictions, targets > 0, losses):
                record = records[len(rows)]
                tp = int((prediction & target).sum())
                fp = int((prediction & ~target).sum())
                fn = int((~prediction & target).sum())
                pixels = target.numel()
                rows.append({"image_key": record["image_key"], "reference_nonempty": record["reference_nonempty"],
                             "tp": tp, "fp": fp, "fn": fn, "pixels": pixels, "bce": float(loss.mean()),
                             **metrics(tp, fp, fn, pixels)})
    require(len(rows) == len(records), "Validation loader/manifest mismatch")
    return {"nonempty": aggregate([r for r in rows if r["reference_nonempty"]]),
            "empty_unverified": aggregate([r for r in rows if not r["reference_nonempty"]])}, rows


def bounded_batches(loader, budget):
    """No extra batch or optimizer update beyond budget; independent of training."""
    require(budget > 0 and len(loader) > 0, "Invalid update budget or empty loader")
    step, pass_number = 0, 0
    while step < budget:
        pass_number += 1
        for batch_number, batch in enumerate(loader, 1):
            step += 1
            yield step, pass_number, batch_number, batch
            if step == budget:
                return


def output_path(run_dir, relative):
    root = (EXPERIMENT_ROOT / "runs").resolve()
    run = Path(run_dir).resolve()
    require(run.parent == root and run != root, "Run must be a direct child of Experiment B/runs")
    destination = (run / relative).resolve()
    require(destination.is_relative_to(run) and destination != run, "Output escapes run directory")
    require(destination != ORIGINAL_CHECKPOINT.resolve(), "Original checkpoint is protected")
    return destination


def create_run():
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    runs = EXPERIMENT_ROOT / "runs"
    require(runs.resolve().parent == EXPERIMENT_ROOT.resolve(), "Redirected runs directory")
    runs.mkdir(exist_ok=True)
    run = runs / name
    run.mkdir(exist_ok=False)  # Never reuse or resume an existing directory.
    return run


def write_json(run, name, value):
    with output_path(run, name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_checkpoint(run, name, state):
    destination = output_path(run, "checkpoints/" + name)
    destination.parent.mkdir(exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(state, stream)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "training_shuffle_generator": generator.get_state()}


def train(config, datasets, manifest, selections, protected):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config["cpu_threads"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(Subset(datasets["train"], selections["train"]),
                              batch_size=2, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=2, shuffle=False, num_workers=0,
                              generator=torch.Generator().manual_seed(config["seed"]))
    model = UNet().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    run = create_run()
    start = time.perf_counter()
    write_json(run, "configuration.json", config)
    write_json(run, "data_manifest.json", {"label": LABEL, "splits": manifest, "protected_file_hashes": protected})
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "opencv-python")}
    source_hashes = {str(p): sha256(p) for p in [Path(__file__), EXPERIMENT_ROOT / "config.json",
                     PROJECT_ROOT / "unet_model.py", PROJECT_ROOT / "dataset_loader.py"]}
    write_json(run, "environment.json", {"python": sys.version, "platform": platform.platform(),
               "packages": versions, "source_sha256": source_hashes, "cpu_threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
    best_key, best_state = None, None
    with output_path(run, "training_history.jsonl").open("x", encoding="utf-8") as th, \
         output_path(run, "validation_history.jsonl").open("x", encoding="utf-8") as vh, \
         output_path(run, "validation_per_image.jsonl").open("x", encoding="utf-8") as ph:
        for step, pass_number, batch_number, (images, targets) in bounded_batches(train_loader, config["max_updates"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            require(logits.shape == targets.shape, "Training output shape mismatch")
            loss = criterion(logits, targets)
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            optimizer.step()
            entry = {"label": LABEL, "update": step, "pass": pass_number, "batch_in_pass": batch_number,
                     "batch_size": len(images), "bce": float(loss.detach()), "elapsed_seconds": time.perf_counter() - start}
            th.write(json.dumps(entry, allow_nan=False) + "\n"); th.flush()
            if step % config["validate_every"] == 0 or step == config["max_updates"]:
                summary, per_image = evaluate(model, valid_loader, manifest["valid"], config["threshold"])
                entry = {"label": LABEL, "update": step, "threshold": config["threshold"],
                         "elapsed_seconds": time.perf_counter() - start, "groups": summary}
                vh.write(json.dumps(entry, allow_nan=False) + "\n"); vh.flush()
                for row in per_image:
                    ph.write(json.dumps({"label": LABEL, "update": step, **row}, allow_nan=False) + "\n")
                ph.flush()
                key = (summary["nonempty"]["macro"]["dice"], -summary["nonempty"]["mean_bce"], -step)
                state = {"label": LABEL, "update": step, "pass": pass_number, "batch_in_pass": batch_number,
                         "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                         "rng_state": rng_state(generator), "configuration": config, "validation": summary}
                if best_key is None or key > best_key:
                    best_key, best_state = key, copy.deepcopy(state)
                print(json.dumps(entry, allow_nan=False), flush=True)
    verify_unchanged(protected)
    # Save exactly once each, exclusively; never replace any checkpoint file.
    save_checkpoint(run, "best.pth", best_state)
    save_checkpoint(run, "last.pth", state)
    write_json(run, "summary.json", {"label": LABEL, "completed_updates": step,
               "best_update": best_state["update"], "elapsed_seconds": time.perf_counter() - start,
               "best_validation": best_state["validation"], "last_validation": summary,
               "protected_files_unchanged": True,
               "limitations": ["Empty masks are not confirmed negatives", "Boundaries are unverified rectangles",
                               "Source-scene independence is unverified", "Not geological segmentation accuracy"]})
    print(f"Completed isolated run: {run}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Read-only data checks; no model, optimizer, outputs, or checkpoints")
    modes.add_argument("--train", action="store_true", help="Explicitly start the separately authorized 700-update experiment")
    args = parser.parse_args(argv)
    if not args.train and not args.validate_only:
        parser.print_help()
        return 0
    try:
        config = load_config()
        # Avoid CPU thread overhead during read-only tensor checks too.
        torch.set_num_threads(config["cpu_threads"])
        datasets, manifest, selections, protected = prepare(config)
        try:
            if args.validate_only:
                print(json.dumps({"label": LABEL, "status": "validation_passed_no_training",
                    "train_total": len(manifest["train"]), "train_selected": len(selections["train"]),
                    "valid_total": len(manifest["valid"]), "valid_nonempty": len(selections["valid"]),
                    "valid_empty": len(manifest["valid"]) - len(selections["valid"]),
                    "original_checkpoint_sha256": protected[str(ORIGINAL_CHECKPOINT)],
                    "outputs_written": False}, indent=2))
            else:
                train(config, datasets, manifest, selections, protected)
        finally:
            verify_unchanged(protected)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"SAFE STOP: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
