"""Lunar CV Preprocessing and Dataset Quality Audit Package."""

from pathlib import Path

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]

__all__ = ["get_project_root"]
