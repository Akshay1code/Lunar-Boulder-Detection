"""Compatibility wrapper: forwards to scripts/review/review_cross_split_pair.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parents[2] / "scripts" / "review" / "review_cross_split_pair.py"
runpy.run_path(str(target), run_name="__main__")
