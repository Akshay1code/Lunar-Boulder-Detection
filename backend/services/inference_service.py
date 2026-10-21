"""
Model Loading and Inference Service
===================================
Coordinates loading of YOLO11n/s detection and U-Net segmentation models,
providing a unified interface for inference across routes and CLI scripts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.models.detection.inference import (
    load_model as load_yolo_model,
    model_status as yolo_status,
    run_detection,
)
from backend.models.segmentation.inference import (
    load_model as load_unet_model,
    model_status as unet_status,
    run_segmentation,
)
from backend.validation.gate import LunarValidationGate
from backend.validation.classifier import load_classifier

log = logging.getLogger(__name__)


class InferenceService:
    """Unified service for loading models and dispatching inference requests."""

    @staticmethod
    def initialize_models() -> dict[str, Any]:
        """Attempt to load detection, segmentation, and classifier models at startup."""
        yolo_ok, yolo_msg = load_yolo_model()
        if yolo_ok:
            log.info("YOLO detection: %s", yolo_msg)
        else:
            log.warning("YOLO detection unavailable: %s", yolo_msg)

        unet_ok, unet_msg = load_unet_model()
        if unet_ok:
            log.info("U-Net segmentation: %s", unet_msg)
        else:
            log.warning("U-Net segmentation unavailable: %s", unet_msg)

        clf_ok, clf_msg = load_classifier()
        if clf_ok:
            log.info("Lunar classifier: %s", clf_msg)
        else:
            log.warning("Lunar classifier unavailable: %s", clf_msg)

        return {
            "detection": {"loaded": yolo_ok, "message": yolo_msg},
            "segmentation": {"loaded": unet_ok, "message": unet_msg},
            "classifier": {"loaded": clf_ok, "message": clf_msg},
        }

    @staticmethod
    def get_health() -> dict[str, Any]:
        """Return operational and readiness status of models and validation gate."""
        return {
            "status": "ok",
            "detection": yolo_status(),
            "segmentation": unet_status(),
            "validation_gate": LunarValidationGate.get_status(),
        }

    @staticmethod
    def validate_image_gate(
        image_bytes: bytes,
        filename: str = "",
        acceptance_threshold: float = 0.70,
        ambiguity_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """Run lunar image validation gate check without invoking primary models."""
        return LunarValidationGate.validate_image(
            image_bytes=image_bytes,
            filename=filename,
            acceptance_threshold=acceptance_threshold,
            ambiguity_threshold=ambiguity_threshold,
        )

    @staticmethod
    def detect_boulders(image_bytes: bytes, conf: float = 0.25, iou: float = 0.45) -> dict[str, Any]:
        """Dispatch detection inference for boulders, craters, and rilles."""
        return run_detection(image_bytes, conf=conf, iou=iou)

    @staticmethod
    def segment_landslides(image_bytes: bytes, threshold: float = 0.5) -> dict[str, Any]:
        """Dispatch U-Net segmentation inference for landslide coverage."""
        return run_segmentation(image_bytes, threshold=threshold)
