"""Compatibility wrapper for src.preprocessing.duplicate_audit."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.duplicate_audit import run_duplicate_audit, main

if __name__ == "__main__":
    main()
