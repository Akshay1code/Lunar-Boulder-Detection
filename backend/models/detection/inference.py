"""
YOLO Object Detection Inference Module
======================================
Loads the trained YOLO11s checkpoint (weights/yolo11s_best.pt) once at
application startup and exposes run_detection() for per-request inference.

Trained checkpoint: runs/lunar_detection/v3/weights/best.pt
Architecture:       YOLO11s
Classes (nc=3):     0=crater  1=rille  2=boulder
Training config:    configs/data_merged.yaml  (150 epochs)
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
# Default checkpoint search order
# ---------------------------------------------------------------------------
_CANDIDATES = [
    BACKEND_DIR / "weights" / "yolo11s_best.pt",
    BACKEND_DIR / "weights" / "yolo11n.pt",
    BACKEND_DIR / "weights" / "yolo11s.pt",
]


def _find_checkpoint() -> Path | None:
    for p in _CANDIDATES:
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------
_model: Any = None
_model_path: Path | None = None
_load_error: str | None = None


def load_model(checkpoint: Path | None = None) -> tuple[bool, str]:
    """
    Load the YOLO model into module-level state.
    Returns (success, message).
    Call this once at application startup.
    """
    global _model, _model_path, _load_error

    ckpt = checkpoint or _find_checkpoint()
    if ckpt is None:
        msg = (
            "No YOLO checkpoint found.  "
            "Expected weights/yolo11s_best.pt or runs/lunar_detection/v3/weights/best.pt"
        )
        _load_error = msg
        return False, msg

    if not ckpt.is_file():
        msg = f"Checkpoint not found: {ckpt}"
        _load_error = msg
        return False, msg

    try:
        from ultralytics import YOLO  # type: ignore[import]

        model = YOLO(str(ckpt))
        _model = model
        _model_path = ckpt
        _load_error = None
        return True, f"Loaded {ckpt.name}"
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load YOLO model: {type(exc).__name__}: {exc}"
        _load_error = msg
        return False, msg


def model_status() -> dict[str, Any]:
    """Return a dict describing the model loading state."""
    if _model is not None:
        names = getattr(_model, "names", {})
        return {
            "available": True,
            "checkpoint": str(_model_path.name) if _model_path else "unknown",
            "checkpoint_path": str(_model_path) if _model_path else None,
            "classes": names if isinstance(names, dict) else {},
            "num_classes": len(names) if names else 0,
        }
    return {
        "available": False,
        "error": _load_error or "Model not loaded",
    }


def run_detection(
    image_bytes: bytes,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict[str, Any]:
    """
    Run YOLO inference on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image (JPEG / PNG / BMP / TIFF).
    conf : float
        Confidence threshold (0–1).
    iou : float
        NMS IoU threshold (0–1).

    Returns
    -------
    dict with keys:
        success          : bool
        error            : str  (only when success is False)
        detections       : list[dict]  – each with class_name, class_id,
                           confidence, bbox_xyxy, bbox_xywh
        count            : int
        class_summary    : dict[str, int]
        inference_time_ms: float
        annotated_image  : str  (base64-encoded PNG, "data:image/png;base64,…")
        original_size    : [width, height]
    """
    global _model
    if _model is None:
        ok, _ = load_model()
        if not ok or _model is None:
            return {"success": False, "error": _load_error or "Model not loaded"}

    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"success": False, "error": "Could not decode image.  Upload a JPEG, PNG, BMP, or TIFF."}

    h, w = img_bgr.shape[:2]
    if w * h > 25_000_000:
        return {"success": False, "error": "Image exceeds 25 MP.  Please resize before uploading."}

    try:
        t0 = time.perf_counter()
        results = _model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            verbose=False,
            save=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Inference error: {type(exc).__name__}: {exc}"}

    res = results[0]
    names: dict = getattr(res, "names", {}) or {}

    detections = []
    class_summary: dict[str, int] = {}

    for box in res.boxes:
        cid = int(box.cls.item())
        cname = names.get(cid, str(cid))
        conf_val = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        bx, by, bw, bh = (int(v) for v in box.xywh[0].tolist())
        detections.append(
            {
                "class_id": cid,
                "class_name": cname,
                "confidence": round(conf_val, 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [bx, by, bw, bh],
            }
        )
        class_summary[cname] = class_summary.get(cname, 0) + 1

    # Sort detections by confidence (descending)
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Annotated image (BGR array → PNG bytes → base64)
    annotated_bgr = res.plot()  # returns BGR ndarray
    ok, encoded = cv2.imencode(".png", annotated_bgr)
    if not ok:
        return {"success": False, "error": "Failed to encode annotated image."}
    annotated_b64 = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()

    return {
        "success": True,
        "detections": detections,
        "count": len(detections),
        "class_summary": class_summary,
        "inference_time_ms": round(elapsed_ms, 1),
        "annotated_image": annotated_b64,
        "original_size": [w, h],
    }
