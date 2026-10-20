"""Isolated four-image training fit diagnostic; no training without --train."""
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

LABEL = "Training-image fit against unverified rectangular reference masks; not validation performance or geological accuracy"
FIXED = dict(label=LABEL, seed=42, device="cpu", cpu_threads=2, image_size=256,
             batch_size=1, initialization="from_scratch", model="UNet", optimizer="Adam",
             learning_rate=0.001, loss="BCEWithLogitsLoss", pos_weight=None, threshold=0.5,
             max_updates=800, evaluate_every=20, schedule="fixed_round_robin_config_order")
IDS = [142, 263, 245, 96]
NAMES = [
    "ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg",
    "ohr_019_patch_2428_png.rf.731671e87d5157f3e99a8b859c0f78db.jpg",
    "ohr_001_patch_1715_png.rf.9723a06f2ed11acf2ec84f2df2532410.jpg",
    "ohr_006_patch_1816_png.rf.92a1a565315147ef762508fc73af4d72.jpg",
]
PRIOR = ROOT / "experiments/experiment_b/runs/20261012T175333Z_09ea39b068f7499191061dd3a0ec0113/data_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def validate_selection(records):
    require(len(records) == 4, "Exactly four selected records required")
    require([r["coco_image_id"] for r in records] == IDS, "Fixed image IDs/order differ")
    require([r["filename"] for r in records] == NAMES, "Fixed filenames/order differ")
    require(len({r["image_sha256"] for r in records}) == 4, "Selected images are exact duplicates")
    require(len({r["image_key"] for r in records}) == 4, "Duplicate image keys")
    for r in records:
        require(r["original_split"] == "train" and r["image_key"] == "train/" + r["filename"], "Only fixed training keys permitted")
        require(r["reference_nonempty"] and r["selected_for_training"], "Nonempty training selection required")


BASELINE_RUN = ROOT / "experiments/four_image_overfit/runs/20261013T044314Z_1465587584be4a16a43db91c35493698"


def verify_budget_only(config, baseline):
    require(baseline["max_updates"] == 400 and config["max_updates"] == 800,
            "Expected update budget change 400 -> 800")
    require(config == dict(baseline, max_updates=800),
            "Only max_updates may differ from saved baseline configuration")


def verify_previous_experiment(config):
    baseline = json.loads((BASELINE_RUN / "configuration.json").read_text(encoding="utf-8"))
    verify_budget_only(config, baseline)
    protected = json.loads((BASELINE_RUN / "protected_hashes_before.json").read_text(encoding="utf-8"))
    for name in ("unet_model.py", "dataset_loader.py", "create_masks.py",
                 "experiments/four_image_overfit/train_four_image.py"):
        path = (ROOT / name).resolve()
        require(digest(path) == protected[str(path)], f"Baseline source changed: {path}")


def load_config():
    c = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    require({k: v for k, v in c.items() if k != "selected"} == FIXED, "Configuration differs from fixed design")
    validate_selection(c["selected"])
    verify_previous_experiment(c)
    return c


def make_dataset(split):
    require(split == "train", "Validation and test split access forbidden")
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        ds = LunarLandslideDataset("train", image_size=256)
        ds.pairs = [(a.resolve(), b.resolve()) for a, b in ds.pairs]
        ds.image_dir = ds.image_dir.resolve()
    finally:
        os.chdir(previous)
    return ds


def verify_mask_rules(mask, info, annotations, landslide_categories):
    """In-memory consistency check, mirroring create_masks.py; never writes PNGs."""
    require(mask is not None and mask.shape == (info["height"], info["width"]), "Mask/COCO dimension mismatch")
    require(set(np.unique(mask).tolist()) <= {0, 255} and bool(mask.any()), "Expected nonempty 0/255 mask")
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


def verify_association(record, info, annotations, categories):
    require(info["id"] == record["coco_image_id"] and info["file_name"] == record["filename"], "COCO image association differs")
    require(all(a["image_id"] == info["id"] for a in annotations), "Wrong annotation image ID")
    require(sorted(a["id"] for a in annotations) == sorted(record["annotation_ids"]), "Annotation IDs differ")
    lands = sorted(a["id"] for a in annotations if a["category_id"] in categories)
    require(bool(lands) and lands == sorted(record["landslide_annotation_ids"]), "Landslide annotation IDs differ")


def prepare(config):
    validate_selection(config["selected"])
    ds = make_dataset("train")
    annotation_path = ds.image_dir / "_annotations.coco.json"
    pairs = []
    for r in config["selected"]:
        matches = [(i, p, m) for i, (p, m) in enumerate(ds.pairs) if p.name == r["filename"]]
        require(len(matches) == 1, "Selected filename must occur exactly once")
        i, p, m = matches[0]
        require(p.resolve() == Path(r["image_path"]).resolve() and m.resolve() == Path(r["mask_path"]).resolve(), "Saved path association differs")
        require(m.name == p.stem + "_mask.png", "Mask naming rule differs")
        pairs.append((i, p, m))
    sources = [p for p in ROOT.iterdir() if p.suffix == ".py"]
    sources += [p for p in (ROOT / "experiments").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md", ".pth")]
    sources += list((ROOT / "checkpoints").rglob("*.pth"))
    require(ROOT / "checkpoints/unet_best.pth" in sources, "Protected original checkpoint missing")
    paths = sources + [annotation_path, PRIOR] + [p for _, a, b in pairs for p in (a, b)]
    before = {str(p.resolve()): digest(p) for p in paths}
    try:
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))["splits"]["train"]
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        require(len({i["id"] for i in coco["images"]}) == len(coco["images"]), "Duplicate COCO image IDs")
        require(len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"]), "Duplicate annotation IDs")
        categories = {c["id"] for c in coco["categories"] if c["name"].lower() == "landslide"}
        tensors, provenance = [], []
        for r, (i, p, m) in zip(config["selected"], pairs):
            match = [x for x in prior if x["image_key"] == r["image_key"]]
            require(len(match) == 1, "Prior training record is not unique")
            for key, value in match[0].items():
                require(r.get(key) == value, f"Selection differs from prior manifest: {key}")
            require(digest(p) == r["image_sha256"] and digest(m) == r["mask_sha256"], "Selected input hash discrepancy")
            infos = [x for x in coco["images"] if x["id"] == r["coco_image_id"]]
            require(len(infos) == 1, "Missing COCO image")
            info = infos[0]
            annotations = [a for a in coco["annotations"] if a["image_id"] == info["id"]]
            verify_association(r, info, annotations, categories)
            mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            verify_mask_rules(mask, info, annotations, categories)
            image, target = ds[i]
            require(image.shape == (3, 256, 256) and target.shape == (1, 256, 256), "Tensor shape mismatch")
            require(image.dtype == target.dtype == torch.float32 and bool(torch.isfinite(image).all()), "Tensor dtype/nonfinite image")
            require(0 <= image.min() <= image.max() <= 1, "Image normalization invalid")
            require(set(torch.unique(target).tolist()) <= {0., 1.} and bool(target.any()), "Nonempty binary target required")
            require(float(np.mean(mask > 0)) == r["reference_foreground_coverage_original"], "Original coverage differs")
            require(float(target.mean()) == r["reference_foreground_coverage_256"], "Resized coverage differs")
            tensors.append((image.unsqueeze(0), target.unsqueeze(0)))
            provenance.append(dict(r, label=LABEL, width=info["width"], height=info["height"], source_scene_id=None, patch_coordinates=None, mask_rules_verified=True))
        return tensors, provenance, before
    finally:
        verify_unchanged(before)


class StepBudget:
    def __init__(self, optimizer):
        self.successful = 0
        self.handles = [optimizer.register_step_pre_hook(self.pre), optimizer.register_step_post_hook(self.post)]

    def pre(self, optimizer, args, kwargs):
        require(self.successful < 800, "Update 801 forbidden")

    def post(self, optimizer, args, kwargs):
        self.successful += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


def execute_updates(update):
    for step in range(1, 801):
        update(step, (step - 1) % 4)


def check_completion(counter, optimizer, evaluations, counts):
    require(counter.successful == 800, "Incomplete successful update count")
    require(evaluations == list(range(0, 801, 20)), "Incomplete evaluation history")
    require(counts == [200]*4, "Each image must receive exactly 200 updates")
    params = [p for g in optimizer.param_groups for p in g["params"]]
    require(bool(params) and all(p in optimizer.state and int(optimizer.state[p]["step"]) == 800 for p in params), "Adam step counters differ")


def metrics(prediction, target):
    truth = target > 0
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                predicted_foreground_coverage=float(prediction.float().mean()),
                reference_foreground_coverage=float(truth.float().mean()))


def aggregate(rows):
    require(len(rows) == 4, "Exactly four evaluation rows required")
    means, counts = {}, {}
    for key in ("bce", "dice", "recall", "precision", "predicted_foreground_coverage", "reference_foreground_coverage"):
        values = [r[key] for r in rows if r[key] is not None]
        means[key] = sum(values)/len(values) if values else None
        counts[key] = len(values)
    return dict(macro_mean=means, defined_image_counts=counts)


def evaluate(model, tensors, records):
    require(len(tensors) == len(records) == 4, "Four evaluation images required")
    model.eval()
    rows = []
    with torch.inference_mode():
        for (image, target), record in zip(tensors, records):
            logits = model(image)
            require(logits.shape == target.shape and bool(torch.isfinite(logits).all()), "Invalid evaluation output")
            p = torch.sigmoid(logits)
            rows.append(dict(image_key=record["image_key"], coco_image_id=record["coco_image_id"],
                bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target)),
                probability_min=float(p.min()), probability_max=float(p.max()), probability_mean=float(p.mean()),
                **metrics(p >= .5, target)))
    return dict(label=LABEL, threshold=.5, per_image=rows, aggregate=aggregate(rows))


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


def train(config, tensors, records, before):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = UNet().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    counter = StepBudget(optimizer)
    run = create_run()
    start = time.perf_counter()
    evaluations, counts = [], [0]*4
    try:
        write_json(run, "configuration.json", config)
        write_json(run, "provenance.json", records)
        write_json(run, "protected_hashes_before.json", before)
        write_json(run, "environment.json", dict(python=sys.version, packages={n: importlib.metadata.version(n) for n in ("torch", "numpy", "opencv-python")}, cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        with output_path(run, "history.jsonl").open("x", encoding="utf-8") as history:
            final = None
            def record(step):
                nonlocal final
                final = dict(update=step, successful_optimizer_steps=counter.successful,
                             updates_per_image=list(counts), elapsed_seconds=time.perf_counter()-start,
                             **evaluate(model, tensors, records))
                history.write(json.dumps(final, allow_nan=False) + "\n")
                history.flush()
                evaluations.append(step)
            record(0)
            def update(step, index):
                image, target = tensors[index]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                require(logits.shape == target.shape, "Training shape mismatch")
                loss = criterion(logits, target)
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Missing/nonfinite gradients")
                optimizer.step()
                counts[index] += 1
                require(counter.successful == step, "Loop/optimizer count mismatch")
                if step % 20 == 0:
                    record(step)
            execute_updates(update)
        check_completion(counter, optimizer, evaluations, counts)
        after = verify_unchanged(before)
        save_checkpoint(run, dict(label=LABEL, update=counter.successful, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), configuration=config))
        write_json(run, "protected_hashes_after.json", after)
    except BaseException:
        print(f"INCOMPLETE run: {run}; successful updates: {counter.successful}", file=sys.stderr)
        raise
    finally:
        counter.close()
        verify_unchanged(before)
    # Completion is recorded only after all work and final preservation checks.
    write_json(run, "summary.json", dict(status="completed", label=LABEL, successful_optimizer_steps=counter.successful,
               updates_per_image=counts, elapsed_seconds=time.perf_counter()-start, protected_files_unchanged=True,
               final_training_image_fit=final))
    print(f"Completed isolated diagnostic: {run}")


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
        tensors, records, before = prepare(config)
        if args.train:
            train(config, tensors, records, before)
        else:
            try:
                print(json.dumps(dict(status="preflight_passed_no_training", records=records,
                                     protected_files=len(before), outputs_written=False), indent=2))
            finally:
                verify_unchanged(before)
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"SAFE STOP: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
