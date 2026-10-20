"""Training-only U-Net previews; no training or scientifically validated claims.

Default checkpoint: checkpoints/unet_best.pth. Threshold is fixed at 0.5.
--check-only runs inference and renders in memory without writing previews.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from dataset_loader import LunarLandslideDataset
from unet_model import UNet

THRESHOLD = 0.5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_dataset():
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        dataset = LunarLandslideDataset("train", image_size=256)
        dataset.pairs = [(a.resolve(), b.resolve()) for a, b in dataset.pairs]
    finally:
        os.chdir(previous)
    return dataset


def select_images(dataset, count):
    selected = []
    stems = [p.stem.casefold() for p, _ in dataset.pairs]
    if len(stems) != len(set(stems)):
        raise ValueError("Ambiguous training image stems; mask associations are not unique")
    for index, (image, mask) in enumerate(dataset.pairs):
        pixels = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f"Unreadable reference mask: {mask}")
        if pixels.ndim != 2 or pixels.dtype != np.uint8 or not set(np.unique(pixels)) <= {0, 255}:
            raise ValueError(f"Expected grayscale 0/255 reference mask: {mask}")
        if np.any(pixels):
            if not image.is_file():
                raise FileNotFoundError(f"Missing training image: {image}")
            selected.append(index)
            if len(selected) == count:
                break
    if not selected:
        raise ValueError("No nonempty training reference masks found")
    if len(selected) < count:
        print(f"Only {len(selected)} nonempty training images available.")
    return selected


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dictionary containing model_state_dict")
    config = checkpoint.get("configuration", {})
    size = checkpoint.get("image_size", config.get("image_size"))
    if size != 256:
        raise ValueError(f"Checkpoint resolution is missing/incompatible: {size}; expected 256")
    if config.get("model", "UNet") != "UNet":
        raise ValueError("Checkpoint specifies a different model")
    model = UNet().cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint parameters")
    model.eval()
    tag = f"update {checkpoint['update']}" if "update" in checkpoint else f"epoch {checkpoint.get('epoch', 'unknown')}"
    return model, tag


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Invalid model output shape or nonfinite logits")
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (probabilities >= THRESHOLD).astype(np.uint8) * 255


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def comparison(original, reference, prediction, filename, checkpoint_name, tag):
    h, w = original.shape[:2]
    enlarged = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[enlarged > 0] = np.rint(.65*rgb[enlarged > 0] + .35*np.array([255, 0, 0])).astype(np.uint8)
    canvas = Image.new("RGB", (1330, 465), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"TRAIN | {filename}", font=font(16), fill="black")
    draw.text((12, 34), f"{checkpoint_name} | {tag} | threshold >= {THRESHOLD} | predicted foreground at 256x256: {np.mean(prediction > 0)*100:.3f}%", font=font(16), fill="black")
    panels = [(rgb, "Original exported image", False), (reference, "Reference mask - UNVERIFIED", True),
              (prediction, "Predicted binary mask (256x256)", True), (overlay, "Prediction overlay (red 35%)", False)]
    for index, (array, title, nearest) in enumerate(panels):
        x = 12 + index*330
        draw.text((x, 70), title, font=font(15), fill="black")
        panel = Image.fromarray(array)
        panel.thumbnail((316, 316), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas.paste(panel, (x+(316-panel.width)//2, 98+(316-panel.height)//2))
    draw.text((12, 428), "Experimental predictions, not validated landslide detections. Rectangular references are not verified geological boundaries.", font=font(15), fill="#8b3400")
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pth"),
                        help="Absolute path or path relative to the project root; original and E/F checkpoint formats supported")
    parser.add_argument("--count", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--check-only", action="store_true", help="Inference/rendering checks only; no output files")
    args = parser.parse_args(argv)
    run = None
    before = {}
    try:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
        before[str(checkpoint)] = digest(checkpoint)
        torch.set_num_threads(2)
        dataset = training_dataset()
        indices = select_images(dataset, args.count)
        for index in indices:
            for path in dataset.pairs[index]:
                before[str(path)] = digest(path)
        model, tag = load_model(checkpoint)
        print(f"Checkpoint: {checkpoint} ({tag})\nThreshold: {THRESHOLD}; CPU; training images only")
        if not args.check_only:
            parent = ROOT / "prediction_previews"
            if parent.resolve() != parent:
                raise ValueError("Redirected output directory refused")
            parent.mkdir(exist_ok=True)
            run = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex)
            run.mkdir(exist_ok=False)
        for number, index in enumerate(indices, 1):
            image_path, mask_path = dataset.pairs[index]
            tensor, target = dataset[index]  # Exact existing preprocessing, including dimension checks.
            if tensor.shape != (3, 256, 256) or not bool(target.any()):
                raise ValueError(f"Invalid preprocessed input or empty reference: {image_path}")
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            reference = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if original is None or reference is None or original.shape[:2] != reference.shape:
                raise ValueError(f"Unreadable or dimension-mismatched pair: {image_path}, {mask_path}")
            prediction = predict(model, tensor)
            figure = comparison(original, reference, prediction, image_path.name, checkpoint.name, tag)
            # Verify PNG serialization in memory in both modes.
            buffer = io.BytesIO()
            figure.save(buffer, format="PNG")
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.load()
                if decoded.size != (1330, 465):
                    raise ValueError("Preview dimensions invalid")
            if run:
                with (run/f"{number:02d}_{image_path.stem}_comparison.png").open("xb") as stream:
                    stream.write(buffer.getvalue())
            figure.close()
            print(f"{number}/{len(indices)} {image_path.name}: predicted foreground {np.mean(prediction > 0)*100:.3f}%")
        print(f"Saved previews: {run}" if run else "Checks passed: no preview files written.")
        print("Training-image visualization only; no scientific accuracy or test-performance claim.")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        if run:
            print(f"Partial previews remain in {run}; no overwrite or automatic repair.", file=sys.stderr)
        return 1
    finally:
        changed = [path for path, expected in before.items()
                   if not Path(path).is_file() or digest(Path(path)) != expected]
        if changed:
            raise RuntimeError(f"Source/checkpoint hash discrepancy; no repairs attempted: {changed}")


if __name__ == "__main__":
    raise SystemExit(main())
