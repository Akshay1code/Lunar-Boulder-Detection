"""Compatibility wrapper for src.preprocessing.dataset_quality_audit."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_quality_audit import run_quality_audit, main

if __name__ == "__main__":
    main()
