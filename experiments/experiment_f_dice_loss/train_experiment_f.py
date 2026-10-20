"""Experiment F: training-mask fit only. Default and --validate-only never train."""
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
sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
EXPERIMENT_ROOT, PROJECT_ROOT = EXPERIMENT, ROOT
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset_loader import LunarLandslideDataset
from unet_model import UNet
LABEL = "Training-image fit against unverified rectangular reference masks; not validation performance or geological accuracy"
ORIGINAL_CHECKPOINT = ROOT / "checkpoints/unet_best.pth"
BASELINE = ROOT / "experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113"
BUDGET = 5180
E_BASELINE = ROOT / 'experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9'
E_PINNED = {'experiments/experiment_b_exposure/train_experiment_e.py': '8f55e20986cf0fd70baa03b42114029baef69663e2831b59708de24a42d37b7e', 'experiments/experiment_b_exposure/config.json': '005f786f6697fab4daac744b50faf4d7698cf5c326a54f5aaaf8738c105afd0f', 'experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9/configuration.json': '005f786f6697fab4daac744b50faf4d7698cf5c326a54f5aaaf8738c105afd0f', 'experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9/provenance.json': '1e7f39512dd1ad2b2b0ce54b05082cfe5ac1883915c8ed7093d1432ee6b6a353'}
LOSS_PARAMETERS = {'bce_weight': 1.0, 'dice_weight': 1.0, 'smoothing': 1.0, 'dice_reduction': 'mean_per_image', 'dice_denominator': 'sum_probabilities_plus_sum_targets', 'empty_target_policy': 'include_smoothed_formula_not_confirmed_negative'}

PINNED_BASELINE = {'experiments/experiment_b/train_experiment_b.py': 'cbe86cf761218af9adb4ff174a4ffc0cdba51fabd472d04cc0d7e840a48d3e0e', 'experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113/configuration.json': '4407aad9262c139eeb02a85cb72597a66c5a64092c179c464227d9ccc8d17bdb', 'experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113/data_manifest.json': '3fcb1c2e43888258a3d3bde662ed0a795a7272f60a531725a2127896efad82fb', 'experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113/environment.json': 'd07e61b7711b68947346809a31577f0ba01b4be9ccc72cdad78b7d3635140a69'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked_selection(nonempty, expected_total, expected_nonempty, split):
    require(len(nonempty) == expected_total,
            f"{split}: expected {expected_total} images, found {len(nonempty)}")
    selected = [i for i, value in enumerate(nonempty) if value]
    require(len(selected) == expected_nonempty,
            f"{split}: expected {expected_nonempty} nonempty masks, found {len(selected)}")
    return selected

def metrics(tp, fp, fn, pixels):
    require(pixels > 0, "Metric pixel count must be positive")
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "predicted_foreground_coverage": (tp + fp) / pixels,
    }

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

digest = sha256

def verify_unchanged(before):
    after, errors = {}, []
    for p, expected in before.items():
        try:
            after[p] = digest(p)
        except OSError as exc:
            after[p] = None
            errors.append(f"{p}: {exc}")
        if after[p] != expected:
            errors.append(f"Hash discrepancy: {p}")
    require(not errors, "\n".join(errors))
    return after

def verify_mask_rules(mask, info, annotations, landslide_categories):
    """In-memory consistency check, mirroring create_masks.py; never writes PNGs."""
    require(mask is not None and mask.shape == (info["height"], info["width"]), "Mask/COCO dimension mismatch")
    require(set(np.unique(mask).tolist()) <= {0, 255}, "Expected 0/255 reference mask")
    expected = np.zeros(mask.shape, dtype=np.uint8)
    for annotation in annotations:
        if annotation["category_id"] not in landslide_categories:
            continue
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            points = np.round(points).astype(np.int32)
            cv2.fillPoly(expected, [points], 255)
    require(np.array_equal(mask, expected), "Existing mask differs from original COCO polygon-fill rules")

def output_path(run, name):
    root = (EXPERIMENT / "runs").resolve()
    run = Path(run).resolve()
    require(root.parent == EXPERIMENT.resolve() and run.parent == root, "Run escapes experiment")
    path = (run / name).resolve()
    require(path.is_relative_to(run) and path != run, "Output escapes run")
    return path

def create_run():
    root = EXPERIMENT / "runs"
    require(root.resolve().parent == EXPERIMENT.resolve(), "Redirected run root")
    root.mkdir(exist_ok=True)
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex)
    run.mkdir(exist_ok=False)
    return run

def write_json(run, name, value):
    with output_path(run, name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def save_checkpoint(run, value):
    with output_path(run, "last.pth").open("xb") as stream:
        torch.save(value, stream)

def checkpoint_state(model, optimizer, config, evaluation):
    # Clone now: state_dict alone shares storage with live model parameters.
    return copy.deepcopy(dict(label=LABEL, update=evaluation["update"],
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        configuration=config, training_image_fit=evaluation,
        selection="strictly_higher_macro_training_dice_earliest_tie"))

def verify_checkpoint(actual, expected):
    def same(a, b):
        if isinstance(b, torch.Tensor):
            return isinstance(a, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(b, dict):
            return isinstance(a, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
        if isinstance(b, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
        return a == b
    require(same(actual, expected), "Serialized checkpoint weights/metadata mismatch")

class BestCheckpoint:
    def __init__(self, run):
        self.run = run
        self.score = None
        self.evaluation = None
        self.file_hash = None

    def publish(self, state):
        destination = output_path(self.run, "best.pth")
        if self.file_hash is None:
            require(not destination.exists(), "Refusing pre-existing best checkpoint")
        else:
            require(destination.is_file() and digest(destination) == self.file_hash,
                    "Best checkpoint changed externally; refusing replacement")
        # Exclusive staging inside this unique run. Interrupted staging is left
        # for inspection; an earlier complete best remains intact.
        staged = output_path(self.run, f"best_update_{state['update']}_{uuid.uuid4().hex}.tmp")
        with staged.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        verify_checkpoint(torch.load(staged, map_location="cpu", weights_only=True), state)
        if self.file_hash is None:
            # Windows rename refuses an existing destination (no first-write overwrite).
            staged.rename(destination)
        else:
            require(digest(destination) == self.file_hash, "Best changed during staging")
            os.replace(staged, destination)
        return digest(destination)

    def consider(self, model, optimizer, config, evaluation):
        score = evaluation["aggregate"]["macro_mean"]["dice"]
        require(score is not None and math.isfinite(score), "Invalid checkpoint selection score")
        if self.score is not None and score <= self.score:
            return False
        state = checkpoint_state(model, optimizer, config, evaluation)
        file_hash = self.publish(state)
        # Advance selection metadata only after successful serialization/publication.
        self.score = score
        self.evaluation = copy.deepcopy(evaluation)
        self.file_hash = file_hash
        return True

    def verify(self):
        path = output_path(self.run, "best.pth")
        require(self.file_hash is not None and digest(path) == self.file_hash, "Best checkpoint missing/changed")
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state["update"] == self.evaluation["update"] and state["training_image_fit"] == self.evaluation,
                "Best checkpoint update/metrics mismatch")
        return state

def make_dataset(split, image_size):
    require(split == "train", "Only training may be accessed")
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

def inspect_dataset(dataset, split, config):
    require(split == "train", "Disallowed dataset split")
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
        verify_mask_rules(mask, info, annotations, land_ids)
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
    return records, selected, {str(annotation_path): sha256(annotation_path)}

def expected_configuration():
    for name, expected in {**PINNED_BASELINE, **E_PINNED}.items():
        require(digest(ROOT / name) == expected, f"Baseline hash discrepancy: {name}")
    baseline = json.loads((E_BASELINE / "configuration.json").read_text(encoding="utf-8"))
    require(baseline == json.loads((E_BASELINE.parents[1] / "config.json").read_text(encoding="utf-8")), "E configuration discrepancy")
    config = dict(baseline)
    config.update(experiment="F", loss="BCEWithLogitsLoss+SoftDiceLoss",
                  loss_parameters=LOSS_PARAMETERS, experiment_e_sha256=E_PINNED)
    return config


def load_config():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    require(config == expected_configuration(), "Configuration differs from approved Experiment F")
    environment = json.loads((BASELINE / "environment.json").read_text(encoding="utf-8"))
    for name, expected in environment["source_sha256"].items():
        require(digest(name) == expected, f"Baseline source changed: {name}")
    for name, expected in environment["packages"].items():
        require(importlib.metadata.version(name) == expected, f"Package differs from baseline: {name}")
    require(platform.python_version() == environment["python"].split()[0], "Python differs from baseline")
    return config


def coverage_group(coverage):
    require(0 < coverage <= 1, "Nonempty reference coverage required")
    if coverage < .05:
        return "lt_5_percent"
    if coverage < .15:
        return "5_to_lt_15_percent"
    if coverage < .30:
        return "15_to_lt_30_percent"
    return "ge_30_percent"


def verify_records(records, prior, selected):
    require(records == prior, "Current training records differ from saved B manifest")
    require(len(selected) == 73 and selected == [i for i,r in enumerate(prior) if r["selected_for_training"]], "Selection/order differs from B")
    require(len({records[i]["image_key"] for i in selected}) == 73, "Duplicate selected keys")
    require(len({records[i]["coco_image_id"] for i in selected}) == 73, "Duplicate selected COCO IDs")
    require(all(r["original_split"] == "train" for r in records), "Non-training record forbidden")


def prepare(config):
    ds = make_dataset("train", config["image_size"])
    paths = list(ROOT.glob("*.py"))
    paths += [p for p in (ROOT / "experiments").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md", ".pth")]
    paths += list((ROOT / "checkpoints").rglob("*.pth"))
    paths += [ds.image_dir / "_annotations.coco.json"] + [p for pair in ds.pairs for p in pair]
    require(ORIGINAL_CHECKPOINT in paths, "Original checkpoint missing")
    before = {str(p.resolve()): digest(p) for p in paths}
    try:
        records, selected, _ = inspect_dataset(ds, "train", config)
        prior = json.loads((BASELINE / "data_manifest.json").read_text(encoding="utf-8"))["splits"]["train"]
        verify_records(records, prior, selected)
        chosen = []
        for i in selected:
            _, target = ds[i]
            coverage = float(target.mean())
            chosen.append(dict(records[i], dataset_index=i, reference_foreground_coverage=coverage,
                               coverage_group=coverage_group(coverage), source_scene_id=None,
                               patch_coordinates=None, label=LABEL))
        require(chosen == json.loads((E_BASELINE / "provenance.json").read_text(encoding="utf-8"))["selection"], "Selection/order differs from E")
        return ds, selected, chosen, before
    finally:
        verify_unchanged(before)


def make_loaders(ds, selected, seed):
    generator = torch.Generator().manual_seed(seed)
    subset = Subset(ds, selected)
    training = DataLoader(subset, batch_size=2, shuffle=True, generator=generator,
                          num_workers=0, drop_last=False)
    evaluation = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
    return training, evaluation, generator


def loss_components(logits, targets):
    """Per-image float32 BCE and soft Dice, followed by batch-mean reduction.

    D_i = (2 sum(sigmoid(z_i)*y_i) + 1)/(sum(sigmoid(z_i)) + sum(y_i) + 1).
    Total = mean(BCE_i) + mean(1-D_i). No binary threshold in the loss.
    Empty targets use the same formula; they are not confirmed negatives.
    """
    require(logits.shape == targets.shape and logits.ndim == 4 and logits.shape[1] == 1
            and logits.shape[0] > 0 and logits.numel() > 0, "Invalid loss shapes")
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(targets).all()), "Nonfinite loss inputs")
    require(bool(((targets == 0) | (targets == 1)).all()), "Binary reference targets required")
    z, y = logits.float(), targets.float()
    bce_per_image = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none").flatten(1).mean(1)
    p, y = torch.sigmoid(z).flatten(1), y.flatten(1)
    dice_per_image = 1.0 - (2.0*(p*y).sum(1) + 1.0)/(p.sum(1) + y.sum(1) + 1.0)
    bce, dice_loss = bce_per_image.mean(), dice_per_image.mean()
    return dict(bce=bce, dice_loss=dice_loss, total_loss=bce+dice_loss,
                bce_per_image=bce_per_image, dice_loss_per_image=dice_per_image)


def aggregate(rows):
    names = ("bce", "dice_loss", "total_loss", "dice", "recall", "precision", "predicted_foreground_coverage", "reference_foreground_coverage")
    means, counts = {}, {}
    for name in names:
        values = [r[name] for r in rows if r[name] is not None]
        counts[name] = len(values)
        means[name] = sum(values)/len(values) if values else None
    return dict(images=len(rows), macro_mean=means, defined_image_counts=counts,
                images_nonzero_dice=sum(r["dice"] is not None and r["dice"] > 0 for r in rows))


def evaluate(model, loader, records):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            require(logits.shape == targets.shape and bool(torch.isfinite(logits).all()), "Invalid evaluation logits")
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            components = loss_components(logits, targets)
            predictions = torch.sigmoid(logits) >= .5
            for prediction, target, loss, dice_loss in zip(predictions, targets > 0, losses, components["dice_loss_per_image"]):
                r = records[len(rows)]
                tp, fp, fn = int((prediction & target).sum()), int((prediction & ~target).sum()), int((~prediction & target).sum())
                coverage = float(target.float().mean())
                require(coverage == r["reference_foreground_coverage"], "Evaluation association/coverage mismatch")
                rows.append(dict(image_key=r["image_key"], coco_image_id=r["coco_image_id"],
                    coverage_group=r["coverage_group"], bce=float(loss.mean()), dice_loss=float(dice_loss),
                    total_loss=float(loss.mean()+dice_loss), reference_foreground_coverage=coverage,
                    **metrics(tp, fp, fn, target.numel())))
    require(len(rows) == len(records) == 73, "Exactly 73 training evaluation records required")
    groups = {name: aggregate([r for r in rows if r["coverage_group"] == name])
              for name in ("lt_5_percent", "5_to_lt_15_percent", "15_to_lt_30_percent", "ge_30_percent")}
    return dict(label=LABEL, threshold=.5, per_image=rows, aggregate=aggregate(rows), coverage_groups=groups)


class StepBudget:
    def __init__(self, optimizer):
        self.successful = 0
        self.handles = [optimizer.register_step_pre_hook(self.pre), optimizer.register_step_post_hook(self.post)]

    def pre(self, optimizer, args, kwargs):
        require(self.successful < BUDGET, "Update 5181 forbidden")

    def post(self, optimizer, args, kwargs):
        self.successful += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


def check_completion(counter, optimizer, evaluations, presentations, pass_number, batch_number):
    require(counter.successful == BUDGET, "Incomplete successful update count")
    require(evaluations == list(range(0, BUDGET+1, 140)), "Incomplete evaluation schedule")
    require(presentations == 73*140 and pass_number == 140 and batch_number == 37, "Exposure budget differs")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    require(bool(params) and all(p in optimizer.state and int(optimizer.state[p]["step"]) == BUDGET for p in params), "Adam step counters differ")


def train(config, ds, selected, records, before):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config["cpu_threads"])
    training, evaluation, generator = make_loaders(ds, selected, config["seed"])
    model = UNet().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    counter = StepBudget(optimizer)
    run = create_run()
    best = BestCheckpoint(run)
    start = time.perf_counter()
    evaluations, presentations = [], 0
    try:
        write_json(run, "configuration.json", config)
        write_json(run, "provenance.json", dict(label=LABEL, selection=records, baseline=str(E_BASELINE), selection_baseline=str(BASELINE)))
        write_json(run, "protected_hashes_before.json", before)
        write_json(run, "environment.json", dict(python=sys.version, platform=platform.platform(),
            packages={n: importlib.metadata.version(n) for n in ("torch", "numpy", "opencv-python")},
            cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        with output_path(run, "training_history.jsonl").open("x", encoding="utf-8") as history, \
             output_path(run, "evaluation_history.jsonl").open("x", encoding="utf-8") as eh:
            final = None
            def record(step):
                nonlocal final
                final = dict(update=step, successful_optimizer_steps=counter.successful,
                             elapsed_seconds=time.perf_counter()-start, **evaluate(model, evaluation, records))
                eh.write(json.dumps(final, allow_nan=False)+"\n"); eh.flush()
                evaluations.append(step)
                best.consider(model, optimizer, config, final)
                print(json.dumps(dict(update=step, aggregate=final["aggregate"]), allow_nan=False), flush=True)
            record(0)
            for step, pass_number, batch_number, (images, targets) in bounded_batches(training, BUDGET):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                require(logits.shape == targets.shape, "Training output shape mismatch")
                components = loss_components(logits, targets)
                loss = components["total_loss"]
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Missing/nonfinite gradient")
                optimizer.step()
                require(counter.successful == step, "Loop/optimizer count mismatch")
                presentations += len(images)
                history.write(json.dumps(dict(label=LABEL, update=step, successful_optimizer_steps=counter.successful,
                    **{"pass":pass_number}, batch_in_pass=batch_number, batch_size=len(images), bce=float(components["bce"].detach()),
                    dice_loss=float(components["dice_loss"].detach()), total_loss=float(loss.detach()),
                    elapsed_seconds=time.perf_counter()-start), allow_nan=False)+"\n"); history.flush()
                if step % 140 == 0:
                    record(step)
        check_completion(counter, optimizer, evaluations, presentations, pass_number, batch_number)
        after = verify_unchanged(before)
        best.verify()
        last_state = checkpoint_state(model, optimizer, config, final)
        save_checkpoint(run, last_state)
        verify_checkpoint(torch.load(output_path(run, "last.pth"), map_location="cpu", weights_only=True), last_state)
        write_json(run, "protected_hashes_after.json", after)
    except BaseException:
        print(f"INCOMPLETE run: {run}; successful updates: {counter.successful}", file=sys.stderr)
        raise
    finally:
        counter.close()
        verify_unchanged(before)
    write_json(run, "summary.json", dict(status="completed", label=LABEL, successful_optimizer_steps=counter.successful,
        complete_passes=140, presentations=presentations, exposures_per_image=140,
        elapsed_seconds=time.perf_counter()-start, protected_files_unchanged=True,
        best_update=best.evaluation["update"], best_training_image_fit=best.evaluation,
        final_training_image_fit=final, best_checkpoint_sha256=best.file_hash,
        last_checkpoint_sha256=digest(output_path(run, "last.pth"))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Read-only preflight, no model or outputs")
    modes.add_argument("--train", action="store_true", help="Explicitly authorize 5,180 optimizer updates")
    args = parser.parse_args(argv)
    if not args.train and not args.validate_only:
        parser.print_help()
        return 0
    try:
        config = load_config()
        torch.set_num_threads(config["cpu_threads"])
        ds, selected, records, before = prepare(config)
        if args.train:
            train(config, ds, selected, records, before)
        else:
            verify_unchanged(before)
            print(json.dumps(dict(status="preflight_passed_no_training", selected_images=len(records),
                selected_image_ids=[r["coco_image_id"] for r in records],
                coverage_groups={g: sum(r["coverage_group"] == g for r in records) for g in sorted({r["coverage_group"] for r in records})},
                mean_reference_foreground_coverage=sum(r["reference_foreground_coverage"] for r in records)/73,
                protected_files=len(before), outputs_written=False), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"SAFE STOP: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
