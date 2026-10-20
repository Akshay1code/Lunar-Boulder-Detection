"""Compatibility wrapper for src.preprocessing.generate_visualizations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.generate_visualizations import generate_visualizations, main

if __name__ == "__main__":
    main()
