"""Compatibility wrapper: forwards to scripts/review/prepare_top5_review.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parents[2] / "scripts" / "review" / "prepare_top5_review.py"
runpy.run_path(str(target), run_name="__main__")
