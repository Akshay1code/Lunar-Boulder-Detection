"""
Experiment J: Fine-tuning from Experiment I checkpoint with verified negative space supervision.

Key differences from Experiment I:
- Loads Experiment I best.pth as starting weights (fine-tuning, not training from scratch)
- Dataset includes ALL 279 train images (not only the 73 with landslide labels)
- Loss uses per-pixel ignore masks: only fg (landslide) and verified dark space pixels contribute
- Lower learning rate (2e-4) to preserve learned features
- Budget: 400 updates (~6-8 minutes on this CPU)
- Validation metric: nonempty landslide Dice + FP rate in space regions

IMPORTANT CONSTRAINTS:
- Does NOT modify test split (no access)
- Does NOT modify Experiment I checkpoint or any existing files
- Does NOT trust unlabelled pixels as confirmed negatives
- Does NOT trust mixed-brightness space annotations
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet

LABEL = "Experiment J: fine-tuned from I with verified space negative supervision; not geological accuracy"
DATASET_ROOT = Path(r"C:\Users\Dylan Noronha\Downloads\moon_ohrc_detection.v2i.coco-segmentation")
I_CHECKPOINT = ROOT / "experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth"

IMAGE_SIZE = 256
DARK_THRESHOLD = 15     # brightness below this is considered genuine space
CLEAN_FRACTION = 0.90   # >90% of annotation pixels must be dark to supervise as background
SPACE_CAT = 4
LANDSLIDE_CAT = 3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rasterize_polygons(anns, cat_id, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for a in anns:
        if a["category_id"] != cat_id:
            continue
        for poly in a.get("segmentation", []):
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def build_supervision_masks(im_gray, anns, h, w, H=IMAGE_SIZE, W=IMAGE_SIZE):
    """
    Returns (fg256, bg256, ignore256) each uint8 at H x W.
    fg:     255 = landslide (positive supervision)
    bg:     255 = verified dark space pixel (negative supervision)
    ignore: 255 = uncertain/unlabelled (excluded from loss)

    Rules:
    - All pixels in landslide annotations → fg
    - Pixels inside space annotations with >=90% dark content AND pixel brightness <15 → bg
    - Space annotation pixels that are too bright, or annotations that are too mixed → ignore
    - All other pixels (unlabelled terrain, shadows, etc.) → ignore
    """
    fg_orig = rasterize_polygons(anns, LANDSLIDE_CAT, h, w)

    space_bg_orig = np.zeros((h, w), dtype=np.uint8)
    space_ign_orig = np.zeros((h, w), dtype=np.uint8)

    for a in anns:
        if a["category_id"] != SPACE_CAT:
            continue
        for poly in a.get("segmentation", []):
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            ann_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(ann_mask, [pts], 255)
            ann_pixels = im_gray[ann_mask > 0]
            if len(ann_pixels) == 0:
                continue
            dark_frac = float((ann_pixels < DARK_THRESHOLD).mean())
            if dark_frac >= CLEAN_FRACTION:
                dark_mask = ((im_gray < DARK_THRESHOLD) & (ann_mask > 0)).astype(np.uint8) * 255
                space_bg_orig = np.maximum(space_bg_orig, dark_mask)
                not_dark = ((im_gray >= DARK_THRESHOLD) & (ann_mask > 0)).astype(np.uint8) * 255
                space_ign_orig = np.maximum(space_ign_orig, not_dark)
            else:
                # Mixed annotation → ignore entirely
                space_ign_orig = np.maximum(space_ign_orig, ann_mask)

    fg256 = cv2.resize(fg_orig, (W, H), interpolation=cv2.INTER_NEAREST)
    bg256 = cv2.resize(space_bg_orig, (W, H), interpolation=cv2.INTER_NEAREST)
    sig256 = cv2.resize(space_ign_orig, (W, H), interpolation=cv2.INTER_NEAREST)

    # Build ignore: everything not fg and not bg and not in a space annotation
    # All unlabelled = ignore too
    base_ignore = np.ones((H, W), dtype=np.uint8) * 255
    base_ignore[fg256 > 0] = 0
    base_ignore[bg256 > 0] = 0
    base_ignore[sig256 > 0] = 0
    ignore256 = np.maximum(sig256, base_ignore)

    # fg wins over bg (should not overlap, but just in case)
    bg256[fg256 > 0] = 0
    ignore256[fg256 > 0] = 0
    ignore256[bg256 > 0] = 0

    return fg256, bg256, ignore256


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpaceNegativeDataset(Dataset):
    """
    Returns (image_tensor, fg_tensor, bg_tensor, ignore_tensor) for each image.

    fg_tensor:     1xHxW float32 {0, 1}
    bg_tensor:     1xHxW float32 {0, 1}
    ignore_tensor: 1xHxW float32 {0, 1}  (1 = ignored in loss)
    """
    def __init__(self, split, image_size=IMAGE_SIZE):
        require(split in {"train", "valid"}, f"Disallowed split: {split}")
        ann_file = DATASET_ROOT / split / "_annotations.coco.json"
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        img_info = {img["id"]: img for img in coco["images"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)

        self.records = []
        for img_id, info in img_info.items():
            img_path = DATASET_ROOT / split / info["file_name"]
            self.records.append(dict(
                image_id=img_id,
                image_path=img_path,
                info=info,
                anns=anns_by_img[img_id],
            ))
        self.records.sort(key=lambda r: str(r["image_path"]))
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        info = rec["info"]
        h, w = info["height"], info["width"]

        im_bgr = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
        require(im_bgr is not None, f"Cannot read: {rec['image_path']}")
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        fg256, bg256, ignore256 = build_supervision_masks(im_gray, rec["anns"], h, w, self.image_size, self.image_size)

        # Image preprocessing: same as Experiment I
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image_t = torch.from_numpy(im_resized.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        H = W = self.image_size
        fg_t = torch.from_numpy((fg256 > 0).astype(np.float32)).unsqueeze(0)
        bg_t = torch.from_numpy((bg256 > 0).astype(np.float32)).unsqueeze(0)
        ign_t = torch.from_numpy((ignore256 > 0).astype(np.float32)).unsqueeze(0)

        return image_t, fg_t, bg_t, ign_t


# ---------------------------------------------------------------------------
# Loss with ignore mask
# ---------------------------------------------------------------------------

def masked_bce_dice_loss(logits, fg, bg, ignore, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
    """
    Compute BCE + soft Dice loss, ignoring uncertain pixels.

    supervised_mask = (fg or bg) and not ignore
    target = fg where supervised
    """
    # Supervision mask: 1 where we have a label (fg or bg), 0 where we ignore
    supervised = ((fg + bg) > 0).float() * (1.0 - ignore)
    # Target: 1 for landslide foreground, 0 for space background
    target = fg.clone()

    # --- Masked BCE ---
    bce_per_pixel = F_torch.binary_cross_entropy_with_logits(logits, target, reduction='none')
    supervised_sum = supervised.sum().clamp(min=1.0)
    bce = (bce_per_pixel * supervised).sum() / supervised_sum

    # --- Masked Soft Dice (per image) ---
    probs = torch.sigmoid(logits)
    dice_loss_sum = 0.0
    batch_size = logits.size(0)
    for i in range(batch_size):
        sup_i = supervised[i]
        p_i = probs[i] * sup_i
        t_i = target[i] * sup_i
        numerator = 2.0 * (p_i * t_i).sum() + smooth
        denominator = p_i.sum() + t_i.sum() + smooth
        dice_loss_sum += 1.0 - numerator / denominator
    dice_loss = dice_loss_sum / batch_size

    total = bce_weight * bce + dice_weight * dice_loss
    return total, bce.detach(), dice_loss.detach()


# ---------------------------------------------------------------------------
# Evaluation helpers (standard binary metrics vs landslide ground truth)
# ---------------------------------------------------------------------------

def metrics_for_batch(logits, fg, threshold=0.5):
    """Returns per-image dicts with tp/fp/fn/pixel counts."""
    preds = (torch.sigmoid(logits) >= threshold)
    targets = (fg > 0)
    results = []
    for p, t in zip(preds, targets):
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        results.append(dict(tp=tp, fp=fp, fn=fn, pixels=int(t.numel()),
                            predicted_coverage=float(p.float().mean()),
                            reference_coverage=float(t.float().mean())))
    return results


def aggregate_metrics(per_image):
    """Aggregate tp/fp/fn across images and compute Dice, IoU, precision, recall."""
    tp = sum(r["tp"] for r in per_image)
    fp = sum(r["fp"] for r in per_image)
    fn = sum(r["fn"] for r in per_image)
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    denom_prec = tp + fp
    denom_rec = tp + fn
    return dict(
        n=len(per_image),
        macro_dice=float(2 * tp / denom_dice) if denom_dice else None,
        macro_iou=float(tp / denom_iou) if denom_iou else None,
        macro_precision=float(tp / denom_prec) if denom_prec else None,
        macro_recall=float(tp / denom_rec) if denom_rec else None,
        mean_pred_coverage=float(np.mean([r["predicted_coverage"] for r in per_image])),
        mean_ref_coverage=float(np.mean([r["reference_coverage"] for r in per_image])),
        tp=tp, fp=fp, fn=fn,
    )


def evaluate_validation(model, valid_loader, threshold=0.5):
    """
    Run eval on all 80 validation images; split into:
    - nonempty: 19 images with landslide annotations
    - space_only: 33 images with space annotations only
    - empty: 28 images with no annotations
    """
    model.eval()
    all_rows = []
    with torch.no_grad():
        for images, fg, bg, ignore in valid_loader:
            logits = model(images)
            batch_results = metrics_for_batch(logits, fg, threshold)
            has_space = [(b > 0).any().item() for b in bg]
            for i, r in enumerate(batch_results):
                r["has_space"] = has_space[i]
            all_rows.extend(batch_results)

    nonempty = [r for r in all_rows if r["reference_coverage"] > 0]
    space_only = [r for r in all_rows if r["reference_coverage"] == 0 and r["has_space"]]
    empty = [r for r in all_rows if r["reference_coverage"] == 0]

    return dict(
        all=aggregate_metrics(all_rows),
        nonempty=aggregate_metrics(nonempty) if nonempty else None,
        space_only=aggregate_metrics(space_only) if space_only else None,
        empty_unverified=aggregate_metrics(empty) if empty else None,
    )


# ---------------------------------------------------------------------------
# Joint rotation augmentation (same as Experiment I)
# ---------------------------------------------------------------------------

def joint_rotate(images, fg, bg, ign, generator):
    n = images.shape[0]
    turns = torch.randint(0, 4, (n,), generator=generator).tolist()
    def rot(t, k):
        return torch.rot90(t, k, (-2, -1)).contiguous()
    ri = torch.stack([rot(images[i], k) for i, k in enumerate(turns)])
    rf = torch.stack([rot(fg[i], k) for i, k in enumerate(turns)])
    rb = torch.stack([rot(bg[i], k) for i, k in enumerate(turns)])
    rg = torch.stack([rot(ign[i], k) for i, k in enumerate(turns)])
    return ri, rf, rb, rg, turns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_run():
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    run = runs / stamp
    run.mkdir(exist_ok=False)
    return run


def output_path(run, name):
    p = (run / name).resolve()
    require(p.parent == run.resolve() and p != run.resolve(), f"Invalid output path: {name}")
    require(not p.exists(), f"Refusing overwrite: {p}")
    return p


def save_checkpoint(run, name, model, optimizer, update, extra_meta):
    state = copy.deepcopy(dict(
        label=LABEL,
        update=update,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        **extra_meta,
    ))
    p = output_path(run, name)
    torch.save(state, p)
    # Verify round-trip
    loaded = torch.load(p, map_location="cpu", weights_only=True)
    require(loaded["update"] == update, "Checkpoint update mismatch")
    return p


def train(cfg, args):
    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])

    # Load datasets
    train_ds = SpaceNegativeDataset("train", image_size=cfg["image_size"])
    valid_ds = SpaceNegativeDataset("valid", image_size=cfg["image_size"])

    require(len(train_ds) == cfg["expected_train_total"],
            f"Expected {cfg['expected_train_total']} train images, got {len(train_ds)}")
    require(len(valid_ds) == cfg["expected_valid_total"],
            f"Expected {cfg['expected_valid_total']} valid images, got {len(valid_ds)}")

    gen_aug = torch.Generator()
    gen_aug.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )


    # Load model from Experiment I checkpoint
    print(f"Loading Experiment I checkpoint: {I_CHECKPOINT}")
    ckpt = torch.load(I_CHECKPOINT, map_location="cpu", weights_only=True)
    model = UNet().cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from update: {ckpt.get('update', '?')}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_score = None
    best_path = None
    run = create_run()

    # Save config
    cfg_copy = dict(cfg, base_checkpoint=str(I_CHECKPOINT),
                    base_checkpoint_sha256=sha256(I_CHECKPOINT))
    with output_path(run, "config.json").open("x", encoding="utf-8") as f:
        json.dump(cfg_copy, f, indent=2)

    start = time.perf_counter()
    history = []
    eval_rows = []

    def do_eval(step):
        nonlocal best_score, best_path
        result = evaluate_validation(model, valid_loader)
        land_dice = result["nonempty"]["macro_dice"] if result["nonempty"] else None
        space_cov = result["space_only"]["mean_pred_coverage"] if result["space_only"] else None
        all_cov = result["all"]["mean_pred_coverage"]
        elapsed = time.perf_counter() - start

        row = dict(update=step, elapsed_seconds=elapsed, **{
            f"val_{k}": v for k, v in result["nonempty"].items()
        } if result["nonempty"] else {}, space_fp_coverage=space_cov, all_coverage=all_cov)
        eval_rows.append(row)

        land_dice_str = f"{land_dice:.4f}" if land_dice is not None else "N/A"
        space_cov_str = f"{space_cov:.4f}" if space_cov is not None else "N/A"
        print(f"[U={step:4d}] Val nonempty Dice={land_dice_str} | "
              f"Val all coverage={all_cov:.4f} | Space-only cov={space_cov_str} "
              f"| {elapsed:.0f}s", flush=True)

        # Save best by nonempty Dice (landslide detection preserved)
        if land_dice is not None and math.isfinite(land_dice):
            if best_score is None or land_dice > best_score:
                best_score = land_dice
                name = f"best_validation_update_{step:05d}.pth"
                best_path = save_checkpoint(run, name, model, optimizer, step,
                                             dict(evaluation=result, configuration=cfg_copy))
                print(f"  *** New best checkpoint: {name} (Dice={land_dice:.4f})")

        return result

    # Initial evaluation before any fine-tuning
    print(f"\nEvaluating Experiment I baseline on validation (before fine-tuning)...")
    do_eval(0)

    # Training loop
    print(f"\nStarting fine-tuning: {cfg['max_updates']} updates, lr={cfg['learning_rate']}")
    step = 0
    while step < cfg["max_updates"]:
        model.train()
        for batch in train_loader:
            if step >= cfg["max_updates"]:
                break
            images, fg, bg, ign = batch

            # Apply joint rotation augmentation (same as Experiment I)
            images, fg, bg, ign, turns = joint_rotate(images, fg, bg, ign, gen_aug)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, bce_val, dice_val = masked_bce_dice_loss(
                logits, fg, bg, ign,
                bce_weight=cfg["bce_weight"],
                dice_weight=cfg["dice_weight"],
            )
            loss.backward()
            optimizer.step()

            step += 1
            elapsed = time.perf_counter() - start
            history.append(dict(update=step, bce=float(bce_val), dice_loss=float(dice_val),
                                 total_loss=float(loss.detach()), elapsed_seconds=elapsed))

            if step % cfg["validate_every"] == 0:
                do_eval(step)
                model.train()

    # Final evaluation
    print("\nFinal evaluation...")
    do_eval(step)

    # Save last checkpoint
    save_checkpoint(run, "last.pth", model, optimizer, step,
                    dict(configuration=cfg_copy))

    # Publish best.pth as alias
    if best_path is not None:
        best_bytes = best_path.read_bytes()
        dest = run / "best.pth"
        require(not dest.exists(), "best.pth already exists")
        dest.write_bytes(best_bytes)
        print(f"\nBest checkpoint: best.pth (from {best_path.name})")

    elapsed_total = time.perf_counter() - start
    summary = dict(
        status="completed",
        successful_updates=step,
        best_score_nonempty_dice=best_score,
        best_checkpoint=str(best_path) if best_path else None,
        elapsed_seconds=elapsed_total,
        evaluation_history=eval_rows,
        training_history_length=len(history),
    )
    with (run / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (run / "training_history.jsonl").open("w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    print(f"\nRun complete in {elapsed_total:.0f}s")
    print(f"Run directory: {run}")
    print(f"Best nonempty Dice: {best_score:.4f}" if best_score else "No best checkpoint saved")


def smoke_test():
    """
    Run 5 updates with batch_size=1, verify shapes and loss is finite.
    No checkpoint saved.
    """
    print("=== SMOKE TEST ===")
    torch.set_num_threads(2)

    ds = SpaceNegativeDataset("train", image_size=256)
    print(f"Dataset OK: {len(ds)} images")

    # Check one item
    img, fg, bg, ign = ds[0]
    require(img.shape == (3, 256, 256), f"Image shape: {img.shape}")
    require(fg.shape == (1, 256, 256), f"FG shape: {fg.shape}")
    require(bg.shape == (1, 256, 256), f"BG shape: {bg.shape}")
    require(ign.shape == (1, 256, 256), f"Ign shape: {ign.shape}")
    require(set(fg.unique().tolist()) <= {0.0, 1.0}, "FG not binary")
    require(set(bg.unique().tolist()) <= {0.0, 1.0}, "BG not binary")
    require(set(ign.unique().tolist()) <= {0.0, 1.0}, "IGN not binary")
    require(bool(torch.isfinite(img).all()), "Non-finite image")
    require(0 <= float(img.min()) and float(img.max()) <= 1.0, "Image out of [0,1]")
    print(f"  Sample 0: fg_px={(fg>0).sum().item()}, bg_px={(bg>0).sum().item()}, ign_px={(ign>0).sum().item()}")

    # Load model
    ckpt = torch.load(I_CHECKPOINT, map_location="cpu", weights_only=True)
    model = UNet().cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    gen = torch.Generator().manual_seed(42)
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        images, fg2, bg2, ign2 = batch
        images, fg2, bg2, ign2, turns = joint_rotate(images, fg2, bg2, ign2, gen)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, bce_v, dice_v = masked_bce_dice_loss(logits, fg2, bg2, ign2)
        require(bool(torch.isfinite(loss)), f"Non-finite loss at step {i}: {loss}")
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - start
        print(f"  Step {i+1}: loss={loss.item():.4f} bce={bce_v.item():.4f} dice={dice_v.item():.4f} t={elapsed:.1f}s")
        if i >= 4:
            break

    sec_per_step = (time.perf_counter() - start) / 5
    print(f"\n  Time per step: {sec_per_step:.1f}s")
    print(f"  Estimated time for 400 updates: {400 * sec_per_step / 60:.1f} minutes")
    print("=== SMOKE TEST PASSED ===\n")
    return sec_per_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true", help="Run 5-step smoke test only")
    group.add_argument("--train", action="store_true", help="Run full training")
    args = parser.parse_args(argv)

    if not (args.smoke_test or args.train):
        parser.print_help()
        return

    with open(HERE / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.smoke_test:
        smoke_test()
    elif args.train:
        train(cfg, args)


if __name__ == "__main__":
    main()
