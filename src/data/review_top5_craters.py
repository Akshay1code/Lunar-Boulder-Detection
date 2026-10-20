"""Compatibility wrapper: forwards to scripts/review/review_top5_craters.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parents[2] / "scripts" / "review" / "review_top5_craters.py"
runpy.run_path(str(target), run_name="__main__")
