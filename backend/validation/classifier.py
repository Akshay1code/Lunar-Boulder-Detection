"""
Lunar vs Non-Lunar Classifier Architecture & Inference
======================================================
Dedicated CNN binary image classifier for Lunar domain gatekeeping.

Classes:
  0: non_lunar
  1: lunar

Model Architecture:
  Lightweight convolutional backbone (MobileNetV3-Small / Custom LunarClassifierNet)
  Input size: 224x224 RGB
  Output: 2 logits -> Softmax -> confidence score in [0.0, 1.0]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_WEIGHTS = BACKEND_DIR / "weights" / "lunar_classifier_best.pth"
CLASSIFIER_INPUT_SIZE = 224


class LunarClassifierNet(nn.Module):
    """
    Compact, robust CNN classifier designed specifically for lunar vs non-lunar
    feature extraction (albedo distribution, craggy topography, lack of vegetation/urban color spectra).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


# Singleton classifier state
_classifier_model: nn.Module | None = None
_classifier_weights_path: Path | None = None
_classifier_load_error: str | None = None


def load_classifier(weights_path: Path | None = None) -> tuple[bool, str]:
    """Attempt to load trained Lunar Classifier checkpoint."""
    global _classifier_model, _classifier_weights_path, _classifier_load_error

    target_path = weights_path or DEFAULT_CLASSIFIER_WEIGHTS
    if not target_path.is_file():
        msg = (
            f"Lunar classifier checkpoint not found at {target_path}. "
            "Please train or download weights using train_lunar_classifier.py."
        )
        _classifier_load_error = msg
        _classifier_model = None
        return False, msg

    try:
        model = LunarClassifierNet(num_classes=2)
        state = torch.load(str(target_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            raise ValueError("Unsupported checkpoint structure.")

        model.eval()
        _classifier_model = model
        _classifier_weights_path = target_path
        _classifier_load_error = None
        log.info("Lunar classifier successfully loaded from %s", target_path)
        return True, f"Loaded {target_path.name}"
    except Exception as exc:
        msg = f"Failed to load lunar classifier checkpoint: {type(exc).__name__}: {exc}"
        _classifier_load_error = msg
        _classifier_model = None
        return False, msg


def get_classifier_status() -> dict[str, Any]:
    """Return operational state of lunar classifier."""
    global _classifier_model, _classifier_weights_path, _classifier_load_error
    if _classifier_model is None and DEFAULT_CLASSIFIER_WEIGHTS.is_file():
        load_classifier(DEFAULT_CLASSIFIER_WEIGHTS)

    return {
        "available": _classifier_model is not None,
        "checkpoint": _classifier_weights_path.name if _classifier_weights_path else None,
        "checkpoint_path": str(_classifier_weights_path) if _classifier_weights_path else None,
        "classes": {0: "non_lunar", 1: "lunar"},
        "input_size": CLASSIFIER_INPUT_SIZE,
        "error": _classifier_load_error,
    }


def classify_image_domain(image_bgr: np.ndarray) -> dict[str, Any]:
    """
    Run domain classification on a decoded BGR image matrix.
    
    Returns:
        dict with:
            - is_lunar (bool)
            - label (str: "lunar" | "non_lunar" | "unavailable")
            - confidence (float: 0.0 to 1.0)
            - probabilities (dict: {"lunar": float, "non_lunar": float})
            - error (str or None)
    """
    global _classifier_model
    if _classifier_model is None:
        if DEFAULT_CLASSIFIER_WEIGHTS.is_file():
            ok, _ = load_classifier(DEFAULT_CLASSIFIER_WEIGHTS)
        if _classifier_model is None:
            return {
                "available": False,
                "label": "unknown",
                "is_lunar": False,
                "confidence": 0.0,
                "probabilities": {"lunar": 0.0, "non_lunar": 0.0},
                "error": _classifier_load_error or "Lunar classifier model weights not loaded.",
            }

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(norm.transpose(2, 0, 1))).unsqueeze(0)

        with torch.no_grad():
            logits = _classifier_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_non_lunar = float(probs[0])
        prob_lunar = float(probs[1])

        label = "lunar" if prob_lunar >= 0.5 else "non_lunar"
        confidence = prob_lunar if label == "lunar" else prob_non_lunar

        return {
            "available": True,
            "label": label,
            "is_lunar": label == "lunar",
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "lunar": round(prob_lunar, 4),
                "non_lunar": round(prob_non_lunar, 4),
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "available": True,
            "label": "unknown",
            "is_lunar": False,
            "confidence": 0.0,
            "probabilities": {"lunar": 0.0, "non_lunar": 0.0},
            "error": f"Classifier inference failed: {type(exc).__name__}: {exc}",
        }
