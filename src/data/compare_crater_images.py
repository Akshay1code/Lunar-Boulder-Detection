"""Compatibility wrapper: forwards to scripts/review/compare_crater_images.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parents[2] / "scripts" / "review" / "compare_crater_images.py"
runpy.run_path(str(target), run_name="__main__")
