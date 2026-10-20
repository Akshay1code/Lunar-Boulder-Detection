"""
Experiment K: Balanced Fine-Tuning with Explicit Terrain and Space Negative Supervision.

Key design principles:
1. Landslide Foreground (target = 1, valid = 1):
   - Exact polygon regions annotated as landslide (category_id = 3).
2. Boundary Uncertainty (target = 0, valid = 0, IGNORE):
   - A narrow 2-pixel dilation border around landslide polygons.
3. Confirmed Lunar Terrain Background (target = 0, valid = 1):
   - Lunar terrain surrounding landslides in positive images.
   - Entire patches of lunar terrain in confirmed non-landslide images (craters/boulders).
4. Verified Space Background (target = 0, valid = 1):
   - Pitch-black pixels (< 15 brightness) inside clean space bounding boxes.
   - Ambiguous/bright pixels inside space boxes marked as IGNORE (valid = 0).
5. Balanced Batching:
   - Every batch of 4 contains 2 landslide-positive images, 1 confirmed terrain image,
     and 1 reliable space image.
6. Checkpoint Selection:
   - Rejects any model that overpredicts lunar terrain (coverage > 40%) or destroys landslide Dice (< 0.45).
   - Optimizes composite score: landslide_macro_dice - 0.5 * space_fp_coverage.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
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
from torch.utils.data import Dataset, DataLoader, Sampler

from unet_model import UNet

LABEL = "Experiment K: Balanced supervision with verified terrain & space negatives"
DATASET_ROOT = Path(r"C:\Users\Dylan Noronha\Downloads\moon_ohrc_detection.v2i.coco-segmentation")
I_CHECKPOINT = ROOT / "experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth"

IMAGE_SIZE = 256
DARK_THRESHOLD = 15
CLEAN_FRACTION = 0.90
LANDSLIDE_CAT = 3
SPACE_CAT = 4


def require(cond, msg):
    if not cond:
        raise RuntimeError(f"Assertion failed: {msg}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Mask Building
# ---------------------------------------------------------------------------

def build_supervision_tensors(im_bgr, anns, h, w, size=256, border_ignore_px=2):
    """
    Returns:
    - target: 1xHxW float32 {0, 1}
    - valid_mask: 1xHxW float32 {0, 1} (1 = supervised, 0 = ignore)
    """
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    
    cats = {a["category_id"] for a in anns}
    has_landslide = (LANDSLIDE_CAT in cats)
    has_space = (SPACE_CAT in cats)
    
    target_orig = np.zeros((h, w), dtype=np.uint8)
    valid_orig = np.ones((h, w), dtype=np.uint8)  # default supervised
    
    if has_landslide:
        # Build landslide mask
        land_mask = np.zeros((h, w), dtype=np.uint8)
        for a in anns:
            if a["category_id"] == LANDSLIDE_CAT:
                for poly in a.get("segmentation", []):
                    pts = np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)
                    cv2.fillPoly(land_mask, [pts], 255)
        
        target_orig[land_mask > 0] = 1
        
        # Add boundary ignore zone
        if border_ignore_px > 0 and land_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_ignore_px*2+1, border_ignore_px*2+1))
            dilated = cv2.dilate(land_mask, kernel)
            border = (dilated > 0) & (land_mask == 0)
            valid_orig[border] = 0
            
    elif has_space:
        # Clean space annotations
        space_bg = np.zeros((h, w), dtype=np.uint8)
        space_ign = np.zeros((h, w), dtype=np.uint8)
        
        for a in anns:
            if a["category_id"] == SPACE_CAT:
                bx, by, bw, bh = [int(v) for v in a["bbox"]]
                patch = im_gray[by:by+bh, bx:bx+bw]
                if patch.size == 0:
                    continue
                dark_frac = float((patch < DARK_THRESHOLD).mean())
                if dark_frac >= CLEAN_FRACTION:
                    dark_px = (im_gray < DARK_THRESHOLD)
                    space_bg[by:by+bh, bx:bx+bw] = dark_px[by:by+bh, bx:bx+bw].astype(np.uint8)
                    not_dark_px = (im_gray >= DARK_THRESHOLD)
                    space_ign[by:by+bh, bx:bx+bw] = not_dark_px[by:by+bh, bx:bx+bw].astype(np.uint8)
                else:
                    space_ign[by:by+bh, bx:bx+bw] = 1
                    
        target_orig.fill(0)
        valid_orig.fill(1)
        valid_orig[space_ign > 0] = 0
        
    else:
        # Confirmed non-landslide terrain (craters, boulders, lunar regolith)
        target_orig.fill(0)
        valid_orig.fill(1)

    # Resize to target training size
    target_256 = cv2.resize(target_orig, (size, size), interpolation=cv2.INTER_NEAREST)
    valid_256 = cv2.resize(valid_orig, (size, size), interpolation=cv2.INTER_NEAREST)
    
    target_t = torch.from_numpy(target_256.astype(np.float32)).unsqueeze(0)
    valid_t = torch.from_numpy(valid_256.astype(np.float32)).unsqueeze(0)
    
    return target_t, valid_t


# ---------------------------------------------------------------------------
# Dataset and Balanced Batch Sampler
# ---------------------------------------------------------------------------

class BalancedDataset(Dataset):
    def __init__(self, split, image_size=IMAGE_SIZE, border_ignore_px=2):
        self.split = split
        self.image_size = image_size
        self.border_ignore_px = border_ignore_px
        
        ann_file = DATASET_ROOT / split / "_annotations.coco.json"
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
            
        img_info = {img["id"]: img for img in coco["images"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
            
        self.records = []
        for img_id, info in sorted(img_info.items(), key=lambda x: x[1]["file_name"]):
            img_path = DATASET_ROOT / split / info["file_name"]
            anns = anns_by_img[img_id]
            cats = {a["category_id"] for a in anns}
            
            if LANDSLIDE_CAT in cats:
                group = "landslide"
            elif SPACE_CAT in cats:
                group = "space"
            else:
                group = "terrain"
                
            self.records.append(dict(
                image_id=img_id,
                image_path=img_path,
                info=info,
                anns=anns,
                group=group,
            ))
            
        self.landslide_indices = [i for i, r in enumerate(self.records) if r["group"] == "landslide"]
        self.terrain_indices = [i for i, r in enumerate(self.records) if r["group"] == "terrain"]
        self.space_indices = [i for i, r in enumerate(self.records) if r["group"] == "space"]
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        info = rec["info"]
        h, w = info["height"], info["width"]
        
        im_bgr = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
        require(im_bgr is not None, f"Could not read: {rec['image_path']}")
        
        target_t, valid_t = build_supervision_tensors(
            im_bgr, rec["anns"], h, w, size=self.image_size, border_ignore_px=self.border_ignore_px
        )
        
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        im_t = torch.from_numpy(im_resized.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        
        return im_t, target_t, valid_t


class BalancedBatchSampler(Sampler):
    """
    Yields batches composed of:
    - 2 Landslide images
    - 1 Confirmed terrain image
    - 1 Reliable space image
    """
    def __init__(self, dataset, total_batches, seed=42):
        self.dataset = dataset
        self.total_batches = total_batches
        self.rng = random.Random(seed)
        
    def __iter__(self):
        land = list(self.dataset.landslide_indices)
        terr = list(self.dataset.terrain_indices)
        spc = list(self.dataset.space_indices)
        
        for _ in range(self.total_batches):
            batch = [
                self.rng.choice(land),
                self.rng.choice(land),
                self.rng.choice(terr),
                self.rng.choice(spc),
            ]
            yield batch
            
    def __len__(self):
        return self.total_batches


# ---------------------------------------------------------------------------
# Masked Loss Function
# ---------------------------------------------------------------------------

def masked_bce_dice_loss(logits, targets, valid_mask, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
    """
    targets: 1xHxW float32 {0, 1}
    valid_mask: 1xHxW float32 {0, 1} (1 = supervised, 0 = ignore)
    """
    # Masked BCE
    bce_raw = F_torch.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    valid_sum = valid_mask.sum().clamp(min=1.0)
    bce = (bce_raw * valid_mask).sum() / valid_sum
    
    # Soft Dice per image
    probs = torch.sigmoid(logits)
    dice_loss_sum = 0.0
    batch_size = logits.size(0)
    for i in range(batch_size):
        m = valid_mask[i]
        p = probs[i] * m
        t = targets[i] * m
        num = 2.0 * (p * t).sum() + smooth
        den = p.sum() + t.sum() + smooth
        dice_loss_sum += 1.0 - (num / den)
        
    dice_loss = dice_loss_sum / batch_size
    total = bce_weight * bce + dice_weight * dice_loss
    return total, bce.detach(), dice_loss.detach()


# ---------------------------------------------------------------------------
# Joint Rotation Augmentation
# ---------------------------------------------------------------------------

def joint_rotate(images, targets, valid_masks, generator):
    n = images.shape[0]
    turns = torch.randint(0, 4, (n,), generator=generator).tolist()
    def rot(t, k):
        return torch.rot90(t, k, (-2, -1)).contiguous()
    ri = torch.stack([rot(images[i], k) for i, k in enumerate(turns)])
    rt = torch.stack([rot(targets[i], k) for i, k in enumerate(turns)])
    rv = torch.stack([rot(valid_masks[i], k) for i, k in enumerate(turns)])
    return ri, rt, rv


# ---------------------------------------------------------------------------
# Rigorous Validation Evaluation
# ---------------------------------------------------------------------------

def evaluate_validation(model, valid_ds, threshold=0.5):
    """
    Evaluates model across the 3 distinct validation groups:
    - 19 Landslide images: macro & micro Dice, IoU, Precision, Recall, Mean Pred Cov
    - 33 Space images: verified-space false-positive coverage
    - 28 Terrain images: verified-terrain false-positive coverage
    - Overall: mean pred coverage across all 80 images
    """
    model.eval()
    
    land_tp, land_fp, land_fn = 0, 0, 0
    land_dices, land_ious, land_precs, land_recs, land_covs = [], [], [], [], []
    
    space_fp_px, space_total_px = 0, 0
    terrain_fp_px, terrain_total_px = 0, 0
    all_covs = []
    
    with torch.no_grad():
        for i in range(len(valid_ds)):
            rec = valid_ds.records[i]
            img_t, target_t, valid_t = valid_ds[i]
            
            prob = torch.sigmoid(model(img_t.unsqueeze(0))).squeeze().numpy()
            pred = (prob >= threshold)
            all_covs.append(float(pred.mean()))
            
            group = rec["group"]
            h, w = rec["info"]["height"], rec["info"]["width"]
            
            if group == "landslide":
                # Build ground truth landslide mask at 256x256
                land_mask = np.zeros((h, w), dtype=np.uint8)
                for a in rec["anns"]:
                    if a["category_id"] == LANDSLIDE_CAT:
                        for poly in a.get("segmentation", []):
                            pts = np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)
                            cv2.fillPoly(land_mask, [pts], 255)
                land_256 = cv2.resize(land_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
                
                tp = int((pred & land_256).sum())
                fp = int((pred & ~land_256).sum())
                fn = int((~pred & land_256).sum())
                land_tp += tp
                land_fp += fp
                land_fn += fn
                
                d = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
                iou = tp/(tp+fp+fn) if (tp+fp+fn) else 0.0
                p = tp/(tp+fp) if (tp+fp) else 0.0
                r = tp/(tp+fn) if (tp+fn) else 0.0
                land_dices.append(d)
                land_ious.append(iou)
                land_precs.append(p)
                land_recs.append(r)
                land_covs.append(float(pred.mean()))
                
            elif group == "space":
                # Verified dark space pixels inside clean space annotations
                im_bgr = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
                im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
                s_mask = np.zeros((h, w), dtype=np.uint8)
                for a in rec["anns"]:
                    if a["category_id"] == SPACE_CAT:
                        bx, by, bw, bh = [int(v) for v in a["bbox"]]
                        patch = im_gray[by:by+bh, bx:bx+bw]
                        if patch.size > 0 and (patch < DARK_THRESHOLD).mean() >= CLEAN_FRACTION:
                            dark_px = (im_gray < DARK_THRESHOLD)
                            s_mask[by:by+bh, bx:bx+bw] = dark_px[by:by+bh, bx:bx+bw].astype(np.uint8) * 255
                            
                s_256 = cv2.resize(s_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
                if s_256.any():
                    space_fp_px += int((pred & s_256).sum())
                    space_total_px += int(s_256.sum())
                    
            else:
                # Confirmed non-landslide terrain
                terrain_fp_px += int(pred.sum())
                terrain_total_px += pred.size
                
    macro_dice = float(np.mean(land_dices)) if land_dices else 0.0
    micro_dice = float(2 * land_tp / (2 * land_tp + land_fp + land_fn)) if (2 * land_tp + land_fp + land_fn) else 0.0
    macro_iou = float(np.mean(land_ious)) if land_ious else 0.0
    macro_prec = float(np.mean(land_precs)) if land_precs else 0.0
    macro_rec = float(np.mean(land_recs)) if land_recs else 0.0
    mean_pred_cov = float(np.mean(land_covs)) if land_covs else 0.0
    space_fp_cov = float(space_fp_px / space_total_px) if space_total_px else 0.0
    terrain_fp_cov = float(terrain_fp_px / terrain_total_px) if terrain_total_px else 0.0
    all_mean_cov = float(np.mean(all_covs))
    
    return dict(
        landslide_images=len(land_dices),
        macro_dice=macro_dice,
        micro_dice=micro_dice,
        macro_iou=macro_iou,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        landslide_mean_pred_coverage=mean_pred_cov,
        space_fp_coverage=space_fp_cov,
        terrain_fp_coverage=terrain_fp_cov,
        all_mean_coverage=all_mean_cov,
        tp=land_tp, fp=land_fp, fn=land_fn,
        space_fp_px=space_fp_px, space_total_px=space_total_px,
        terrain_fp_px=terrain_fp_px, terrain_total_px=terrain_total_px,
    )


# ---------------------------------------------------------------------------
# Run Infrastructure
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
    return p


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg, max_updates_override=None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    
    max_updates = max_updates_override or cfg["max_updates"]
    
    print(f"Loading datasets (image_size={cfg['image_size']})...")
    train_ds = BalancedDataset("train", image_size=cfg["image_size"], border_ignore_px=cfg["border_ignore_pixels"])
    valid_ds = BalancedDataset("valid", image_size=cfg["image_size"], border_ignore_px=0)
    
    print(f"Train images: {len(train_ds)} (Landslide: {len(train_ds.landslide_indices)}, "
          f"Terrain: {len(train_ds.terrain_indices)}, Space: {len(train_ds.space_indices)})")
    print(f"Valid images: {len(valid_ds)} (Landslide: {len(valid_ds.landslide_indices)}, "
          f"Terrain: {len(valid_ds.terrain_indices)}, Space: {len(valid_ds.space_indices)})")
    
    sampler = BalancedBatchSampler(train_ds, total_batches=max_updates, seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    
    # Load model from Experiment I checkpoint
    print(f"\nLoading baseline checkpoint: {I_CHECKPOINT}")
    ckpt = torch.load(I_CHECKPOINT, map_location="cpu", weights_only=True)
    model = UNet().cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    gen_aug = torch.Generator().manual_seed(cfg["seed"])
    
    run = create_run()
    cfg_copy = dict(cfg, base_checkpoint=str(I_CHECKPOINT), base_sha256=sha256(I_CHECKPOINT))
    with output_path(run, "config.json").open("x", encoding="utf-8") as f:
        json.dump(cfg_copy, f, indent=2)
        
    start_time = time.perf_counter()
    eval_history = []
    train_history = []
    
    best_score = -float("inf")
    best_path = None
    best_eval = None
    
    def evaluate_and_record(step):
        nonlocal best_score, best_path, best_eval
        metrics = evaluate_validation(model, valid_ds)
        elapsed = time.perf_counter() - start_time
        metrics["update"] = step
        metrics["elapsed_seconds"] = elapsed
        eval_history.append(metrics)
        
        # Calculate composite selection score:
        # We want high landslide dice, low space fp, without blowing up terrain fp or coverage
        macro_dice = metrics["macro_dice"]
        space_fp = metrics["space_fp_coverage"]
        terrain_fp = metrics["terrain_fp_coverage"]
        mean_cov = metrics["landslide_mean_pred_coverage"]
        
        composite = macro_dice - 0.5 * space_fp
        metrics["composite_score"] = composite
        
        # Check acceptance criteria
        crit = cfg["acceptance_criteria"]
        acceptable = (
            macro_dice >= crit["min_landslide_macro_dice"] and
            terrain_fp <= crit["max_terrain_fp_coverage"] and
            mean_cov <= crit["max_landslide_mean_pred_coverage"]
        )
        
        print(f"[U={step:3d} | {elapsed:.0f}s] Macro Dice: {macro_dice:.4f} | Space FP: {space_fp:.4f} | "
              f"Terrain FP: {terrain_fp:.4f} | Landslide Cov: {mean_cov:.4f} | Score: {composite:.4f}", flush=True)
        
        if acceptable and composite > best_score:
            best_score = composite
            best_eval = metrics
            name = f"best_update_{step:05d}.pth"
            best_path = save_checkpoint(run, name, model, optimizer, step,
                                        dict(evaluation=metrics, configuration=cfg_copy))
            print(f"  *** NEW BEST CHECKPOINT: {name} (Score={composite:.4f}, Dice={macro_dice:.4f}, Space FP={space_fp:.4f})", flush=True)
            
        return metrics

    # Baseline evaluation at update 0
    print("\n--- Evaluating Baseline (Experiment I) before fine-tuning ---", flush=True)
    evaluate_and_record(0)
    
    print(f"\n--- Starting fine-tuning: {max_updates} updates, lr={cfg['learning_rate']} ---", flush=True)
    step = 0
    model.train()
    for batch in train_loader:
        if step >= max_updates:
            break
            
        images, targets, valid_masks = batch
        images, targets, valid_masks = joint_rotate(images, targets, valid_masks, gen_aug)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, bce, dice_loss = masked_bce_dice_loss(
            logits, targets, valid_masks,
            bce_weight=cfg["loss_parameters"]["bce_weight"],
            dice_weight=cfg["loss_parameters"]["dice_weight"],
            smooth=cfg["loss_parameters"]["smoothing"],
        )
        loss.backward()
        optimizer.step()
        
        step += 1
        elapsed = time.perf_counter() - start_time
        train_history.append(dict(
            update=step, bce=float(bce), dice_loss=float(dice_loss),
            total_loss=float(loss.detach()), elapsed_seconds=elapsed
        ))
        
        if step % 10 == 0:
            print(f"  [Step {step:3d}/{max_updates}] Total Loss: {float(loss):.4f} (BCE: {float(bce):.4f}, Dice: {float(dice_loss):.4f}) | {elapsed:.0f}s", flush=True)

        if step % cfg["validate_every"] == 0 or step == max_updates:
            evaluate_and_record(step)
            model.train()
            
    # Finalize run
    save_checkpoint(run, "last.pth", model, optimizer, step, dict(configuration=cfg_copy))
    
    if best_path is not None:
        (run / "best.pth").write_bytes(best_path.read_bytes())
        print(f"\nPublished best.pth from {best_path.name}")
    else:
        print("\nWARNING: No checkpoint met acceptance criteria!")
        
    total_time = time.perf_counter() - start_time
    summary = dict(
        status="completed",
        successful_updates=step,
        elapsed_seconds=total_time,
        best_checkpoint=str(best_path) if best_path else None,
        best_score=best_score if best_score != -float("inf") else None,
        best_evaluation=best_eval,
        evaluation_history=eval_history,
    )
    with output_path(run, "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    with output_path(run, "training_history.jsonl").open("x", encoding="utf-8") as f:
        for row in train_history:
            f.write(json.dumps(row) + "\n")
            
    print(f"\nExperiment K run completed in {total_time:.1f}s.")
    print(f"Run directory: {run}")
    return run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Experiment K")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 updates smoke test")
    parser.add_argument("--train", action="store_true", help="Run full training (200 updates)")
    args = parser.parse_args()
    
    with open(HERE / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    if args.smoke_test:
        print("=== RUNNING SMOKE TEST (5 updates) ===")
        train(cfg, max_updates_override=5)
    elif args.train:
        print("=== RUNNING FULL TRAINING (200 updates) ===")
        train(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
