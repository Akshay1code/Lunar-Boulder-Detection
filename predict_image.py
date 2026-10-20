import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from unet_model import UNet


# Configuration
IMAGE_SIZE = 256
ROOT = Path(__file__).resolve().parent

def get_checkpoint_path() -> Path:
    """Resolve checkpoint path from LUNAR_CHECKPOINT_PATH or standard locations."""
    env_ckpt = os.environ.get("LUNAR_CHECKPOINT_PATH")
    if env_ckpt:
        return Path(env_ckpt).resolve()

    # Default checkpoint locations
    candidates = [
        ROOT / "checkpoints/unet_best.pth",
        ROOT / "experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c

    return candidates[0]


def get_dataset_root() -> Path:
    """Resolve dataset root from LUNAR_DATASET_ROOT or standard locations."""
    env_path = os.environ.get("LUNAR_DATASET_ROOT")
    if env_path:
        return Path(env_path).resolve()

    local_rel = ROOT / "dataset"
    if local_rel.is_dir():
        return local_rel

    downloads_fallback = (
        Path.home()
        / "Downloads"
        / "moon_ohrc_detection.v2i.coco-segmentation"
    )
    if downloads_fallback.is_dir():
        return downloads_fallback

    return local_rel


CHECKPOINT_PATH = get_checkpoint_path()
DATASET_ROOT = get_dataset_root()

# Input image: can be provided as first CLI argument or defaults to standard test sample
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    IMAGE_PATH = Path(sys.argv[1]).resolve()
else:
    IMAGE_PATH = (
        DATASET_ROOT
        / "train"
        / "ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg"
    )

REFERENCE_MASK_PATH = (
    ROOT / "dataset_masks" / "train" / f"{IMAGE_PATH.stem}_mask.png"
)

OUTPUT_DIR = ROOT / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cpu")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}.\n"
            "Please configure the checkpoint path by setting LUNAR_CHECKPOINT_PATH, e.g.:\n"
            "  set LUNAR_CHECKPOINT_PATH=path\\to\\best.pth"
        )

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Input image not found: {IMAGE_PATH}.\n"
            "Please provide a path to an image as a command-line argument, e.g.:\n"
            "  python predict_image.py path\\to\\image.jpg\n"
            "or set LUNAR_DATASET_ROOT to the dataset root folder."
        )

    # Load the trained model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(model_state)
    model.eval()

    # Read the original image.
    original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

    if original is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    # Check for optional reference mask
    reference_mask = None
    if REFERENCE_MASK_PATH.is_file():
        reference_mask = cv2.imread(str(REFERENCE_MASK_PATH), cv2.IMREAD_GRAYSCALE)

    # Preprocess the image.
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    normalized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)

    # Generate predictions.
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        print(f"Probability statistics at model resolution: {probabilities.shape}")
        print("Minimum probability:", float(probabilities.min()))
        print("Maximum probability:", float(probabilities.max()))
        print("Mean probability:", float(probabilities.mean()))

        for threshold in [0.1, 0.2, 0.3, 0.5]:
            percentage = (probabilities >= threshold).mean() * 100
            print(f"Pixels at or above {threshold}: {percentage:.3f}%")

    # Fixed diagnostic threshold; lower thresholds are reported only.
    binary_mask = (probabilities >= 0.5).astype(np.uint8) * 255

    # Resize predictions to the original image dimensions.
    height, width = original.shape[:2]

    mask_original_size = cv2.resize(
        binary_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Fixed probability scale: 0 -> 0, 1 -> 255; never min-max normalize.
    probability_original_size = cv2.resize(
        probabilities, (width, height), interpolation=cv2.INTER_LINEAR
    )
    probability_heatmap = np.rint(
        np.clip(probability_original_size, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    # Create a red overlay for predicted pixels.
    overlay = original.copy()
    overlay[mask_original_size > 0] = (0, 0, 255)

    result = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)

    # Save results.
    mask_path = OUTPUT_DIR / "predicted_mask.png"
    overlay_path = OUTPUT_DIR / "prediction_overlay.png"
    original_path = OUTPUT_DIR / "original_image.png"
    heatmap_path = OUTPUT_DIR / "probability_heatmap.png"

    for path, output in [
        (mask_path, mask_original_size),
        (overlay_path, result),
        (original_path, original),
        (heatmap_path, probability_heatmap),
    ]:
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"Could not save diagnostic output: {path}")

    saved_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if saved_mask is None:
        raise RuntimeError(f"Could not read saved prediction: {mask_path}")
    print(f"Saved prediction dimensions (height, width): {saved_mask.shape}")
    if reference_mask is not None:
        print(f"Reference mask dimensions (height, width): {reference_mask.shape}")
        print(f"Dimensions match: {saved_mask.shape == reference_mask.shape}")
        print(f"Reference mask: {REFERENCE_MASK_PATH}")
        print(
            f"Unverified reference-mask foreground: "
            f"{(reference_mask > 0).mean() * 100:.3f}%"
        )
    else:
        print("Reference mask: None (standalone image inference)")

    print("Prediction completed!")
    print(f"Input image: {IMAGE_PATH}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    elif "update" in checkpoint:
        print(f"Checkpoint update: {checkpoint['update']}")
    print(f"Mask saved to: {mask_path}")
    print(f"Overlay saved to: {overlay_path}")
    print(f"Original image copy saved to: {original_path}")
    print(f"Probability heatmap saved to: {heatmap_path} (0-255 = probability 0-1)")
    print(
        f"Predicted pixel percentage: "
        f"{(saved_mask > 0).mean() * 100:.3f}%"
    )
    print(
        "Note: Predictions are experimental and are not "
        "validated lunar landslide detections."
    )
    print("Reference masks are unverified rectangular annotations, not ground truth.")


if __name__ == "__main__":
    main()
