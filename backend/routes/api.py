"""
Flask API Blueprint — Lunar Boulder and Landslide Detection
=============================================================
REST endpoints providing health check, runtime config, model readiness,
lunar validation gatekeeper, YOLO crater/rille/boulder detection,
and U-Net landslide segmentation.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from backend.services.inference_service import InferenceService
from backend.models.detection.inference import model_status as yolo_status
from backend.models.segmentation.inference import model_status as unet_status
from backend.validation.file_validator import MAX_UPLOAD_BYTES

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Backward compatibility alias
routes = Blueprint("routes", __name__)

_ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}
_MAX_BYTES = MAX_UPLOAD_BYTES  # 16 MB

# Ephemeral in-memory store for validated images bound to SHA-256 fingerprint
_VALIDATION_RECORDS: dict[str, dict[str, Any]] = {}


def _compute_hash(data: bytes) -> str:
    """Return SHA-256 content hash used as secure image_id."""
    return hashlib.sha256(data).hexdigest()


def _check_upload() -> tuple[bytes | None, str | None, str]:
    """Validate multipart image upload with secure_filename sanitization."""
    if "image" not in request.files:
        return None, "No image field in the upload.", ""

    f = request.files["image"]
    raw_name = f.filename or ""
    clean_name = secure_filename(raw_name) or "upload.jpg"
    if raw_name == "":
        return None, "No file selected.", ""

    data = f.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        return None, f"File exceeds maximum upload size of {_MAX_BYTES // (1024 * 1024)} MB.", clean_name

    mime = f.mimetype or ""
    ext = Path(clean_name).suffix.lower()
    if mime not in _ALLOWED_MIME and ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        return None, f"Unsupported file type: {mime or ext or 'unknown'}. Upload a JPEG, PNG, BMP, or TIFF.", clean_name

    return data, None, clean_name


def _safe_float(value: str | None, default: float, lo: float, hi: float) -> float:
    if value is None:
        return default
    try:
        v = float(value)
        if lo <= v <= hi:
            return v
    except (TypeError, ValueError):
        pass
    return default


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@api_bp.get("/health")
def health():
    return jsonify(InferenceService.get_health())


@api_bp.get("/config")
def get_config():
    """Return runtime configuration for client-side dropzone and limits."""
    return jsonify({
        "success": True,
        "max_upload_bytes": _MAX_BYTES,
        "max_upload_mb": _MAX_BYTES // (1024 * 1024),
        "allowed_mime": sorted(list(_ALLOWED_MIME)),
        "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        "default_thresholds": {
            "acceptance": 0.70,
            "ambiguity": 0.35,
            "detection_conf": 0.25,
            "detection_iou": 0.45,
            "segmentation_threshold": 0.50,
        },
    })


@api_bp.get("/models")
def get_models():
    """Return availability, loaded checkpoints, and classes for selectable models."""
    det = yolo_status()
    seg = unet_status()

    yolo_classes = []
    if det.get("available") and isinstance(det.get("classes"), dict):
        yolo_classes = [det["classes"][k] for k in sorted(det["classes"].keys(), key=lambda x: int(x))]

    return jsonify({
        "success": True,
        "models": [
            {
                "id": "yolo11n",
                "name": "YOLO11s/n Surface Object Detection",
                "task": "Object Detection",
                "ready": bool(det.get("available")),
                "checkpoint": det.get("checkpoint"),
                "classes": yolo_classes,
                "reason": None if det.get("available") else det.get("error", "YOLO checkpoint not found"),
                "description": "Scans surface imagery for boulders, craters, and rilles.",
            },
            {
                "id": "unet",
                "name": "U-Net Landslide Segmentation",
                "task": "Binary Segmentation",
                "ready": bool(seg.get("available")),
                "checkpoint": seg.get("checkpoint"),
                "classes": ["landslide"],
                "reason": None if seg.get("available") else seg.get("error", "U-Net weights missing"),
                "description": "Generates pixel-level binary landslide hazard masks.",
            },
        ],
    })


@api_bp.post("/validate")
def validate():
    """
    Validate uploaded image through file sanity and dedicated lunar classifier.
    Stores validation decision bound to content hash (image_id).
    """
    image_bytes, error, clean_name = _check_upload()
    if error:
        return jsonify({
            "success": False,
            "error": {"code": "INVALID_UPLOAD", "message": error},
            "is_permitted": False,
            "decision": "rejected",
            "status": "invalid_file",
            "category": "invalid",
            "confidence": 0.0,
            "reason": error,
            "explanation": error,
            "checks": [
                {"id": "file_integrity", "status": "failed", "detail": error},
                {"id": "domain_classifier", "status": "skipped", "detail": "Skipped due to file error"},
            ],
        }), 400

    image_id = _compute_hash(image_bytes)
    acceptance = _safe_float(request.form.get("acceptance_threshold"), 0.70, 0.1, 1.0)
    ambiguity = _safe_float(request.form.get("ambiguity_threshold"), 0.35, 0.05, 0.95)

    gate_result = InferenceService.validate_image_gate(
        image_bytes=image_bytes,
        filename=clean_name,
        acceptance_threshold=acceptance,
        ambiguity_threshold=ambiguity,
    )
    gate_result.pop("decoded_image", None)

    status = gate_result.get("status", "rejected")
    category = gate_result.get("category", "uncertain")
    conf = gate_result.get("confidence", 0.0)
    explanation = gate_result.get("explanation", "")
    is_permitted = gate_result.get("is_permitted", False)

    if status == "accepted":
        decision = "accepted"
        checks = [
            {"id": "file_integrity", "status": "passed", "detail": f"Decodable {gate_result['file_metadata']['format']} ({gate_result['file_metadata']['dimensions']})"},
            {"id": "domain_classifier", "status": "passed", "detail": f"Lunar surface confidence {conf:.1%}"},
        ]
    elif status == "ambiguous":
        decision = "ambiguous"
        checks = [
            {"id": "file_integrity", "status": "passed", "detail": f"Decodable {gate_result['file_metadata']['format']} ({gate_result['file_metadata']['dimensions']})"},
            {"id": "domain_classifier", "status": "needs_review", "detail": f"Uncertain confidence ({conf:.1%}) between 35% and 70%"},
        ]
    else:
        decision = "rejected"
        checks = [
            {"id": "file_integrity", "status": "passed" if gate_result.get("file_metadata") else "failed", "detail": "Structural integrity verified" if gate_result.get("file_metadata") else explanation},
            {"id": "domain_classifier", "status": "failed", "detail": explanation},
        ]

    # Record validation decision server-side against SHA-256 image_id
    _VALIDATION_RECORDS[image_id] = {
        "decision": decision,
        "is_permitted": is_permitted,
        "label": category,
        "confidence": conf,
        "timestamp": time.time(),
    }

    response_data = {
        "success": True,
        "image_id": image_id,
        "decision": decision,
        "label": category,
        "confidence": conf,
        "reason": explanation if decision != "accepted" else None,
        "is_permitted": is_permitted,
        "status": status,
        "category": category,
        "explanation": explanation,
        "file_metadata": gate_result.get("file_metadata"),
        "checks": checks,
    }

    if status == "invalid_file":
        response_data["success"] = False
        response_data["error"] = explanation
        response_data["error_detail"] = {"code": "INVALID_FILE", "message": explanation}
        return jsonify(response_data), 400
    if status == "unavailable":
        response_data["success"] = False
        response_data["error"] = explanation
        response_data["error_detail"] = {"code": "CLASSIFIER_UNAVAILABLE", "message": explanation}
        return jsonify(response_data), 503

    return jsonify(response_data), 200


@api_bp.post("/detect")
def detect():
    """Run YOLO detection, enforced by the server-side validation gate."""
    image_bytes, error, clean_name = _check_upload()
    if error:
        return jsonify({
            "success": False,
            "error": error,
            "error_detail": {"code": "INVALID_UPLOAD", "message": error},
        }), 400

    image_id = _compute_hash(image_bytes)
    rec = _VALIDATION_RECORDS.get(image_id)

    if not rec or not rec.get("is_permitted"):
        gate_result = InferenceService.validate_image_gate(image_bytes=image_bytes, filename=clean_name)
        if not gate_result["is_permitted"]:
            gate_result.pop("decoded_image", None)
            err_msg = "Validation Gate Rejected: " + gate_result["explanation"]
            return jsonify({
                "success": False,
                "error": err_msg,
                "error_detail": {
                    "code": "VALIDATION_FAILED",
                    "message": err_msg,
                },
                "validation": gate_result,
            }), 403
        _VALIDATION_RECORDS[image_id] = {
            "decision": gate_result["status"],
            "is_permitted": True,
            "label": gate_result["category"],
            "confidence": gate_result["confidence"],
            "timestamp": time.time(),
        }

    conf = _safe_float(request.form.get("conf"), 0.25, 0.0, 1.0)
    iou = _safe_float(request.form.get("iou"), 0.45, 0.0, 1.0)

    result = InferenceService.detect_boulders(image_bytes, conf=conf, iou=iou)
    if not result.get("success"):
        err_msg = result.get("error", "Detection failed")
        return jsonify({
            "success": False,
            "error": err_msg,
            "error_detail": {"code": "INFERENCE_ERROR", "message": err_msg},
        }), 422

    result["image_id"] = image_id
    return jsonify(result)


@api_bp.post("/segment")
def segment():
    """Run U-Net segmentation, enforced by the server-side validation gate."""
    image_bytes, error, clean_name = _check_upload()
    if error:
        return jsonify({
            "success": False,
            "error": error,
            "error_detail": {"code": "INVALID_UPLOAD", "message": error},
        }), 400

    image_id = _compute_hash(image_bytes)
    rec = _VALIDATION_RECORDS.get(image_id)

    if not rec or not rec.get("is_permitted"):
        gate_result = InferenceService.validate_image_gate(image_bytes=image_bytes, filename=clean_name)
        if not gate_result["is_permitted"]:
            gate_result.pop("decoded_image", None)
            err_msg = "Validation Gate Rejected: " + gate_result["explanation"]
            return jsonify({
                "success": False,
                "error": err_msg,
                "error_detail": {
                    "code": "VALIDATION_FAILED",
                    "message": err_msg,
                },
                "validation": gate_result,
            }), 403
        _VALIDATION_RECORDS[image_id] = {
            "decision": gate_result["status"],
            "is_permitted": True,
            "label": gate_result["category"],
            "confidence": gate_result["confidence"],
            "timestamp": time.time(),
        }

    threshold = _safe_float(request.form.get("threshold"), 0.5, 0.0, 1.0)

    result = InferenceService.segment_landslides(image_bytes, threshold=threshold)
    if not result.get("success"):
        err_msg = result.get("error", "Segmentation failed")
        return jsonify({
            "success": False,
            "error": err_msg,
            "error_detail": {"code": "INFERENCE_ERROR", "message": err_msg},
        }), 422

    result["image_id"] = image_id
    return jsonify(result)


# Mirror endpoints on routes for backwards compatibility without prefix
@routes.get("/api/health")
def compat_health():
    return health()


@routes.get("/api/config")
def compat_config():
    return get_config()


@routes.get("/api/models")
def compat_models():
    return get_models()


@routes.post("/api/validate")
def compat_validate():
    return validate()


@routes.post("/api/detect")
def compat_detect():
    return detect()


@routes.post("/api/segment")
def compat_segment():
    return segment()
