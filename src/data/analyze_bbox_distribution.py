"""Compatibility wrapper for src.preprocessing.bbox_statistics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.bbox_statistics import compute_bbox_statistics, main

if __name__ == "__main__":
    main()
