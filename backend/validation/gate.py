"""
Lunar Image Validation Gatekeeper
=================================
Orchestrates the entire validation pipeline:
1. Basic file integrity, MIME, decode check, and dimension bounds.
2. Lunar domain classifier evaluation.
3. Configurable conservative decision gate & scoring policy.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

from backend.validation.file_validator import validate_file_integrity
from backend.validation.classifier import classify_image_domain, get_classifier_status

log = logging.getLogger(__name__)

# Configurable decision thresholds
# Default conservative policy:
# - Lunar probability >= ACCEPTANCE_THRESHOLD (0.70) -> ALLOWED (green)
# - Lunar probability < 0.35 -> REJECTED (non-lunar, red)
# - 0.35 <= Lunar probability < 0.70 -> AMBIGUOUS (uncertain, amber)
DEFAULT_ACCEPTANCE_THRESHOLD = 0.70
DEFAULT_AMBIGUITY_THRESHOLD = 0.35


class LunarValidationGate:
    """Decision gate enforcing upload validation before primary deep learning models."""

    @staticmethod
    def get_status() -> dict[str, Any]:
        """Return status of validation gate including classifier availability."""
        clf_status = get_classifier_status()
        return {
            "gate_enabled": True,
            "classifier": clf_status,
            "thresholds": {
                "acceptance": DEFAULT_ACCEPTANCE_THRESHOLD,
                "ambiguity": DEFAULT_AMBIGUITY_THRESHOLD,
            },
        }

    @staticmethod
    def validate_image(
        image_bytes: bytes,
        filename: str = "",
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Validate an image through file validation and lunar domain screening.
        
        Returns:
            dict containing:
                - is_permitted (bool): whether primary models are permitted to execute
                - status (str: "accepted" | "rejected" | "ambiguous" | "invalid_file" | "unavailable")
                - category (str: "lunar" | "non_lunar" | "uncertain" | "invalid")
                - confidence (float)
                - explanation (str)
                - file_metadata (dict)
                - decoded_image (np.ndarray or None)
        """
        # Step 1: File integrity validation
        file_res = validate_file_integrity(image_bytes, filename=filename)
        if not file_res["is_valid"]:
            return {
                "is_permitted": False,
                "status": "invalid_file",
                "category": "invalid",
                "confidence": 0.0,
                "explanation": f"File validation failed: {file_res['error']}",
                "file_metadata": {
                    "format": file_res.get("format"),
                    "size_bytes": len(image_bytes),
                    "dimensions": None,
                },
                "decoded_image": None,
            }

        img_bgr = file_res["decoded_image"]
        w = file_res["width"]
        h = file_res["height"]
        file_meta = {
            "format": file_res["format"],
            "size_bytes": len(image_bytes),
            "dimensions": f"{w}x{h}",
            "width": w,
            "height": h,
            "channels": file_res["channels"],
        }

        # Step 2: Domain classification
        clf_res = classify_image_domain(img_bgr)
        if not clf_res.get("available"):
            # Classifier checkpoint is missing or failed to initialize
            return {
                "is_permitted": False,
                "status": "unavailable",
                "category": "uncertain",
                "confidence": 0.0,
                "explanation": (
                    "Lunar domain classifier checkpoint is unavailable. "
                    "Cannot verify lunar origin without trained classifier weights."
                ),
                "file_metadata": file_meta,
                "decoded_image": img_bgr,
            }

        if clf_res.get("error"):
            return {
                "is_permitted": False,
                "status": "rejected",
                "category": "uncertain",
                "confidence": 0.0,
                "explanation": f"Classifier evaluation error: {clf_res['error']}",
                "file_metadata": file_meta,
                "decoded_image": img_bgr,
            }

        lunar_prob = clf_res["probabilities"]["lunar"]
        non_lunar_prob = clf_res["probabilities"]["non_lunar"]

        # Step 3: Conservative decision gate
        if lunar_prob >= acceptance_threshold:
            return {
                "is_permitted": True,
                "status": "accepted",
                "category": "lunar",
                "confidence": round(lunar_prob, 4),
                "explanation": (
                    f"Image exhibits high visual similarity to supported lunar surface imagery "
                    f"(confidence: {lunar_prob:.1%}). Accepted for detection and segmentation analysis."
                ),
                "file_metadata": file_meta,
                "decoded_image": img_bgr,
            }

        if lunar_prob < ambiguity_threshold:
            return {
                "is_permitted": False,
                "status": "rejected",
                "category": "non_lunar",
                "confidence": round(non_lunar_prob, 4),
                "explanation": (
                    f"Input identified as non-lunar imagery (non-lunar confidence: {non_lunar_prob:.1%}). "
                    f"Analysis blocked to preserve deep learning pipeline integrity."
                ),
                "file_metadata": file_meta,
                "decoded_image": img_bgr,
            }

        # Ambiguous / uncertain range
        return {
            "is_permitted": False,
            "status": "ambiguous",
            "category": "uncertain",
            "confidence": round(lunar_prob, 4),
            "explanation": (
                f"Classification confidence ({lunar_prob:.1%}) is insufficient to confirm lunar origin "
                f"(requires >= {acceptance_threshold:.0%}). Upload a clearer lunar surface image or review manually."
            ),
            "file_metadata": file_meta,
            "decoded_image": img_bgr,
        }
