"""
File Validation Module
======================
Validates raw file bytes before any deep learning models receive them:
- Max file size limits (default 16 MB)
- Magic byte detection (JPEG, PNG, WEBP, BMP, TIFF)
- Complete image decoding verification via OpenCV / Pillow
- Dimension bounds check (min 64x64, max 8192x8192)
- Rejection of empty, truncated, or corrupt payloads
"""
from __future__ import annotations

import io
from typing import Any
import cv2
import numpy as np
from PIL import Image

MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB
MIN_DIMENSION = 64
MAX_DIMENSION = 8192

# Magic byte signatures
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WebP begins RIFF....WEBP
    b"BM": "image/bmp",
    b"II*\x00": "image/tiff",  # Little-endian TIFF
    b"MM\x00*": "image/tiff",  # Big-endian TIFF
}


def detect_mime_from_bytes(data: bytes) -> str | None:
    """Check magic bytes directly to prevent spoofed file extensions."""
    if len(data) < 12:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "image/tiff"
    return None


def validate_file_integrity(
    image_bytes: bytes,
    filename: str = "",
    max_bytes: int = MAX_UPLOAD_BYTES,
    min_dim: int = MIN_DIMENSION,
    max_dim: int = MAX_DIMENSION,
) -> dict[str, Any]:
    """
    Perform deep file and image sanity validation.
    
    Returns:
        dict with:
            - is_valid (bool)
            - error (str or None)
            - format (str or None)
            - width (int or None)
            - height (int or None)
            - channels (int or None)
            - decoded_image (np.ndarray or None)
    """
    if not image_bytes:
        return {
            "is_valid": False,
            "error": "Uploaded file is empty (0 bytes).",
            "format": None,
            "width": None,
            "height": None,
            "channels": None,
            "decoded_image": None,
        }

    if len(image_bytes) > max_bytes:
        size_mb = len(image_bytes) / (1024 * 1024)
        max_mb = max_bytes // (1024 * 1024)
        return {
            "is_valid": False,
            "error": f"File size exceeds limit ({size_mb:.1f} MB > {max_mb} MB).",
            "format": None,
            "width": None,
            "height": None,
            "channels": None,
            "decoded_image": None,
        }

    detected_mime = detect_mime_from_bytes(image_bytes)
    if not detected_mime:
        return {
            "is_valid": False,
            "error": "Unsupported or unrecognized file signature. Must be a valid JPEG, PNG, WEBP, BMP, or TIFF image.",
            "format": None,
            "width": None,
            "height": None,
            "channels": None,
            "decoded_image": None,
        }

    # Verify Pillow can open and verify structural integrity without truncation
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img.verify()
    except Exception as exc:
        return {
            "is_valid": False,
            "error": f"Image file is corrupt or truncated: {exc}",
            "format": detected_mime,
            "width": None,
            "height": None,
            "channels": None,
            "decoded_image": None,
        }

    # Verify OpenCV decoding to ndarray
    np_buf = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_valid": False,
            "error": "Image decoding failed. Data cannot be decoded into pixel matrix.",
            "format": detected_mime,
            "width": None,
            "height": None,
            "channels": None,
            "decoded_image": None,
        }

    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) > 2 else 1

    if h < min_dim or w < min_dim:
        return {
            "is_valid": False,
            "error": f"Image resolution ({w}x{h}) is below minimum requirement ({min_dim}x{min_dim}).",
            "format": detected_mime,
            "width": w,
            "height": h,
            "channels": channels,
            "decoded_image": None,
        }

    if h > max_dim or w > max_dim:
        return {
            "is_valid": False,
            "error": f"Image resolution ({w}x{h}) exceeds maximum permitted dimension ({max_dim}x{max_dim}).",
            "format": detected_mime,
            "width": w,
            "height": h,
            "channels": channels,
            "decoded_image": None,
        }

    return {
        "is_valid": True,
        "error": None,
        "format": detected_mime,
        "width": w,
        "height": h,
        "channels": channels,
        "decoded_image": img_bgr,
    }
