"""
U-Net Landslide Segmentation Inference Module
==============================================
Attempts to load a trained U-Net checkpoint and exposes run_segmentation()
for per-request binary segmentation inference.

Architecture : UNet (unet_model.py) — 3 in / 1 out / base_channels=16
Input size   : 256 × 256 pixels
Output       : raw logits → sigmoid → threshold → binary mask
Preprocessing: BGR→RGB, INTER_AREA resize, /255 normalisation

IMPORTANT: No trained checkpoint exists in this repository.
The segmentation endpoint will report 'unavailable' until a valid .pth
checkpoint is placed at one of the paths listed in _CANDIDATES below and
the application is restarted.

To train a checkpoint: python train_unet.py
Required: dataset at LUNAR_DATASET_ROOT with dataset_masks/ sub-directory.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/

# ---------------------------------------------------------------------------
# Checkpoint search order
# ---------------------------------------------------------------------------
_CANDIDATES = [
    BACKEND_DIR / "weights" / "unet_best.pth",
]

IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_model: Any = None
_model_path: Path | None = None
_load_error: str | None = None


def _find_checkpoint() -> Path | None:
    for p in _CANDIDATES:
        if p.is_file():
            return p
    return None


def load_model(checkpoint: Path | None = None) -> tuple[bool, str]:
    """
    Attempt to load the U-Net model.
    Returns (success, message).
    """
    global _model, _model_path, _load_error

    ckpt = checkpoint or _find_checkpoint()
    if ckpt is None:
        msg = (
            "U-Net checkpoint not found.  "
            "Expected weights/unet_best.pth or checkpoints/unet_best.pth.  "
            "Train a checkpoint with: python backend/scripts/train_unet.py"
        )
        _load_error = msg
        return False, msg

    if not ckpt.is_file():
        msg = f"U-Net checkpoint not found: {ckpt}"
        _load_error = msg
        return False, msg

    try:
        import torch  # type: ignore[import]

        try:
            from backend.models.segmentation.unet_model import UNet
        except ImportError:
            from unet_model import UNet  # type: ignore[import]

        checkpoint_data = torch.load(str(ckpt), map_location="cpu", weights_only=True)

        if not isinstance(checkpoint_data, dict) or "model_state_dict" not in checkpoint_data:
            raise ValueError(
                "Checkpoint must be a dict with 'model_state_dict' key.  "
                "This checkpoint format is not supported."
            )

        expected_size = checkpoint_data.get("image_size", checkpoint_data.get("configuration", {}).get("image_size"))
        if expected_size is not None and expected_size != IMAGE_SIZE:
            raise ValueError(
                f"Checkpoint expects image_size={expected_size}, but inference uses {IMAGE_SIZE}."
            )

        model = UNet()
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        model.eval()

        _model = model
        _model_path = ckpt
        _load_error = None
        return True, f"Loaded {ckpt.name}"

    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load U-Net checkpoint: {type(exc).__name__}: {exc}"
        _load_error = msg
        return False, msg


def model_status() -> dict[str, Any]:
    """Return current model availability state."""
    global _model
    if _model is None:
        load_model()
    if _model is not None:
        return {
            "available": True,
            "checkpoint": _model_path.name if _model_path else "unknown",
            "checkpoint_path": str(_model_path) if _model_path else None,
            "input_size": IMAGE_SIZE,
            "note": (
                "Trained on unverified rectangular reference annotations. "
                "Predictions are experimental, not geologically validated."
            ),
        }
    return {
        "available": False,
        "error": _load_error or "U-Net model not loaded",
    }


def _preprocess(image_bgr: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """Exact preprocessing from existing dataset_loader.py and demo.py."""
    import torch  # type: ignore[import]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    return tensor


def run_segmentation(
    image_bytes: bytes,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Run U-Net segmentation on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes.
    threshold : float
        Binarisation threshold applied to sigmoid output (0–1).

    Returns
    -------
    dict with keys:
        success             : bool
        error               : str  (only when success is False)
        coverage_percent    : float  – predicted foreground pixel %
        threshold           : float
        inference_time_ms   : float
        mask_image          : str  (base64 PNG, binary 0/255)
        overlay_image       : str  (base64 PNG, original with red overlay)
        original_image      : str  (base64 PNG, display-scaled original)
        original_size       : [width, height]
        warning             : str  (broad-activation warning if coverage ≥ 40%)
    """
    global _model
    if _model is None:
        load_model()
    if _model is None:
        return {
            "success": False,
            "error": _load_error or (
                "U-Net segmentation is not yet available.  "
                "No trained checkpoint was found.  "
                "Train a model first with: python backend/scripts/train_unet.py"
            ),
        }

    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        return {"success": False, "error": "Threshold must be a number between 0 and 1."}

    # Decode
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"success": False, "error": "Could not decode image."}

    if img_bgr.shape[0] * img_bgr.shape[1] > 25_000_000:
        return {"success": False, "error": "Image exceeds 25 MP."}

    try:
        import torch  # type: ignore[import]

        tensor = _preprocess(img_bgr).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = _model(tensor)
            if logits.shape != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unexpected model output shape: {logits.shape}")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model produced non-finite output.")
            probabilities = torch.sigmoid(logits)[0, 0].numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000

    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Inference error: {type(exc).__name__}: {exc}"}

    # Binarise
    binary = (probabilities >= threshold).astype(np.uint8) * 255
    coverage = float(np.mean(binary > 0) * 100)

    # Scale to display size (cap at 1024px on longest axis)
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1024.0 / max(h, w))
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    display = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr.copy()

    # Mask resized to display dimensions
    mask_display = cv2.resize(binary, (dw, dh), interpolation=cv2.INTER_NEAREST)

    # Red overlay
    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    overlay_rgb = display_rgb.copy()
    fg = mask_display > 0
    if fg.any():
        overlay_rgb[fg] = np.rint(
            0.65 * display_rgb[fg] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

    def _to_b64(array_bgr_or_gray: np.ndarray) -> str:
        ok, enc = cv2.imencode(".png", array_bgr_or_gray)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()

    # Display images
    original_b64 = _to_b64(display)
    mask_b64 = _to_b64(mask_display)
    overlay_b64 = _to_b64(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    warning = ""
    if coverage >= 40.0:
        warning = (
            "Broad foreground activation: ≥40% of output pixels are highlighted.  "
            "This model may incorrectly mark shadowed terrain or empty space.  "
            "Do not interpret this as a confirmed landslide extent."
        )

    return {
        "success": True,
        "coverage_percent": round(coverage, 3),
        "threshold": threshold,
        "inference_time_ms": round(elapsed_ms, 1),
        "mask_image": mask_b64,
        "overlay_image": overlay_b64,
        "original_image": original_b64,
        "original_size": [w, h],
        "warning": warning,
    }
