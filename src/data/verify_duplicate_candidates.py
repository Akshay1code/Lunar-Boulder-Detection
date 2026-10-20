"""Compatibility wrapper: forwards to scripts/review/verify_duplicate_candidates.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parents[2] / "scripts" / "review" / "verify_duplicate_candidates.py"
runpy.run_path(str(target), run_name="__main__")
