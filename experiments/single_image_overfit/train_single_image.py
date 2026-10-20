"""Single training-image fit diagnostic. No training without --train."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from dataset_loader import LunarLandslideDataset
from unet_model import UNet

LABEL = "Training-image fit against an unverified rectangular reference mask; not validation performance or geological accuracy"
IMAGE_KEY = "train/ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg"
PRIOR_RUN = ROOT / "experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113"
FIXED = dict(label=LABEL, image_key=IMAGE_KEY, seed=42, device="cpu", cpu_threads=2,
             image_size=256, batch_size=1, initialization="from_scratch", model="UNet",
             optimizer="Adam", learning_rate=0.001, loss="BCEWithLogitsLoss",
             pos_weight=None, threshold=0.5, max_updates=200)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def verify_unchanged(before):
    after, discrepancies = {}, []
    for path, expected in before.items():
        try:
            after[path] = digest(path)
        except OSError as exc:
            after[path] = None
            discrepancies.append(f"{path}: {exc}")
        if after[path] != expected:
            discrepancies.append(f"Hash discrepancy: {path}")
    require(not discrepancies, "\n".join(discrepancies))
    return after


def load_config():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    require(config == FIXED, "Configuration differs from approved fixed design")
    return config


def make_dataset(split):
    require(split == "train", "Only the training split is permitted")
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        ds = LunarLandslideDataset("train", image_size=256)
        ds.pairs = [(a.resolve(), b.resolve()) for a, b in ds.pairs]
        ds.image_dir = ds.image_dir.resolve()
    finally:
        os.chdir(previous)
    return ds


def selected_index(pairs):
    matches = [i for i, (p, _) in enumerate(pairs) if p.name == IMAGE_KEY.split("/", 1)[1]]
    require(len(matches) == 1, "Expected exactly one selected training image")
    return matches[0]


def prepare():
    # Only train is instantiated; other training pairs are listed by the existing
    # loader, but only the selected image and mask are decoded.
    ds = make_dataset("train")
    index = selected_index(ds.pairs)
    image_path, mask_path = ds.pairs[index]
    annotation_path = ds.image_dir / "_annotations.coco.json"
    sources = [ROOT / n for n in ("unet_model.py", "dataset_loader.py", "train_unet.py", "predict_image.py")]
    sources += [EXPERIMENT / n for n in ("train_single_image.py", "config.json", "test_single_image.py", "README.md")]
    sources += [ROOT / "experiments/experiment_b/train_experiment_b.py"]
    checkpoints = list((ROOT / "checkpoints").rglob("*.pth"))
    checkpoints += list((ROOT / "experiments").rglob("*.pth"))
    require(ROOT / "checkpoints/unet_best.pth" in checkpoints, "Original checkpoint missing")
    for name in ("best.pth", "last.pth"):
        require(PRIOR_RUN / "checkpoints" / name in checkpoints, f"Experiment B {name} missing")
    manifest_path = PRIOR_RUN / "data_manifest.json"
    before = snapshot(sources + checkpoints + [image_path, mask_path, annotation_path, manifest_path])
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))["splits"]["train"]
        matches = [r for r in records if r["image_key"] == IMAGE_KEY]
        require(len(matches) == 1, "Selected key is not unique in prior manifest")
        prior = matches[0]
        require(prior["original_split"] == "train" and prior["reference_nonempty"], "Prior selection differs")
        for kind, path in (("image", image_path), ("mask", mask_path)):
            require(Path(prior[kind + "_path"]).resolve() == path, f"{kind} path differs")
            require(before[str(path)] == prior[kind + "_sha256"], f"{kind} hash differs from prior manifest")
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = [i for i in coco["images"] if i["file_name"] == image_path.name]
        require(len(images) == 1, "COCO image association is not unique")
        info = images[0]
        annotations = [a for a in coco["annotations"] if a["image_id"] == info["id"]]
        land_ids = {c["id"] for c in coco["categories"] if c["name"].lower() == "landslide"}
        lands = [a["id"] for a in annotations if a["category_id"] in land_ids]
        require(info["id"] == prior["coco_image_id"] and sorted(lands) == sorted(prior["landslide_annotation_ids"]) and bool(lands), "COCO landslide association differs")
        require(sorted(a["id"] for a in annotations) == sorted(prior["annotation_ids"]), "Annotation IDs differ")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(mask is not None, "Unreadable mask")
        require(mask.shape == (info["height"], info["width"]), "COCO dimensions differ")
        require(set(np.unique(mask).tolist()) <= {0, 255} and bool(mask.any()), "Mask must be nonempty binary 0/255")
        image, target = ds[index]
        require(image.shape == (3, 256, 256) and target.shape == (1, 256, 256), "Unexpected tensor shapes")
        require(image.dtype == target.dtype == torch.float32, "Unexpected tensor dtype")
        require(bool(torch.isfinite(image).all()) and 0 <= image.min() <= image.max() <= 1, "Invalid normalized image")
        require(set(torch.unique(target).tolist()) <= {0., 1.} and bool(target.any()), "Resized target must remain nonempty binary")
        record = dict(prior)
        record.update(label=LABEL, reference_foreground_coverage=float(target.mean()), source_scene_id=None,
                      patch_coordinates=None)
        return image.unsqueeze(0), target.unsqueeze(0), record, before
    finally:
        verify_unchanged(before)


class StepBudget:
    """Hooks count actual successful step returns, independently of loop index."""
    def __init__(self, optimizer, limit=200):
        self.successful = 0
        self.limit = limit
        self.handles = [optimizer.register_step_pre_hook(self.pre), optimizer.register_step_post_hook(self.post)]

    def pre(self, optimizer, args, kwargs):
        require(self.successful < self.limit, "Optimizer update beyond budget forbidden")

    def post(self, optimizer, args, kwargs):
        self.successful += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


def execute_updates(update):
    for step in range(1, 201):
        update(step)


def check_completion(counter, optimizer, rows):
    require(counter.successful == 200, "Run incomplete: successful optimizer steps != 200")
    require(rows == list(range(201)), "Run incomplete: history must contain updates 0 through 200")
    params = [p for g in optimizer.param_groups for p in g["params"]]
    require(bool(params), "No optimizer parameters")
    require(all(p in optimizer.state and int(optimizer.state[p]["step"]) == 200 for p in params), "Adam parameter step counters differ from 200")


def metrics(prediction, target):
    truth = target > 0
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                recall=tp/(tp+fn) if tp+fn else None,
                foreground_coverage=float(prediction.float().mean()))


def evaluate(model, image, target):
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        require(logits.shape == target.shape and bool(torch.isfinite(logits).all()), "Invalid evaluation logits")
        probabilities = torch.sigmoid(logits)
        p = probabilities.numpy()
        return dict(label=LABEL, evaluation_bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target)),
                    probability_min=float(p.min()), probability_max=float(p.max()),
                    probability_mean=float(p.mean(dtype=np.float64)), probability_median=float(np.median(p)),
                    reference_foreground_coverage=float(target.mean()), threshold=0.5,
                    **metrics(probabilities >= 0.5, target))


def output_path(run, name):
    root = (EXPERIMENT / "runs").resolve()
    run = Path(run).resolve()
    require(root.parent == EXPERIMENT.resolve() and run.parent == root, "Run path escapes experiment")
    path = (run / name).resolve()
    require(path.is_relative_to(run) and path != run, "Output path escapes unique run")
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


def train(config, image, target, record, before):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = UNet().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    counter = StepBudget(optimizer)
    run = create_run()
    start = time.perf_counter()
    rows = []
    try:
        write_json(run, "configuration.json", config)
        write_json(run, "selected_image.json", record)
        write_json(run, "protected_hashes_before.json", before)
        write_json(run, "environment.json", dict(python=sys.version, packages={n: importlib.metadata.version(n) for n in ("torch", "numpy", "opencv-python")}, cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        with output_path(run, "history.jsonl").open("x", encoding="utf-8") as stream:
            def record_row(step, training_loss):
                row = dict(update=step, successful_optimizer_steps=counter.successful,
                           training_bce_before_update=training_loss, elapsed_seconds=time.perf_counter()-start,
                           **evaluate(model, image, target))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(step)
            record_row(0, None)
            def update(step):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                require(logits.shape == target.shape, "Training shape mismatch")
                loss = criterion(logits, target)
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Missing/nonfinite gradients")
                optimizer.step()
                require(counter.successful == step, "Loop and successful-step counts disagree")
                record_row(step, float(loss.detach()))
            execute_updates(update)
        check_completion(counter, optimizer, rows)
        after = verify_unchanged(before)
        with output_path(run, "last.pth").open("xb") as stream:
            torch.save(dict(label=LABEL, update=counter.successful, model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(), configuration=config), stream)
        write_json(run, "protected_hashes_after.json", after)
        final_fit = evaluate(model, image, target)
    except BaseException:
        # Missing completion summary means incomplete, including hard termination.
        print(f"Run not completed: {run}; successful updates: {counter.successful}", file=sys.stderr)
        raise
    finally:
        counter.close()
        verify_unchanged(before)
    # No completion marker exists if any training, saving, or final hash check
    # raises. Write this only after the try/finally has completed successfully.
    write_json(run, "summary.json", dict(status="completed", label=LABEL, successful_optimizer_steps=counter.successful,
               elapsed_seconds=time.perf_counter()-start, protected_files_unchanged=True,
               final_training_image_fit=final_fit))
    print(f"Completed 200-update training-image fit diagnostic: {run}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    if not args.train and not args.validate_only:
        parser.print_help()
        return 0
    try:
        config = load_config()
        torch.set_num_threads(2)
        image, target, record, before = prepare()
        if args.train:
            train(config, image, target, record, before)
        else:
            try:
                print(json.dumps(dict(status="preflight_passed_no_training", selected_image=record,
                     protected_files=len(before), outputs_written=False), indent=2))
            finally:
                verify_unchanged(before)
    except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as exc:
        print(f"SAFE STOP: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
