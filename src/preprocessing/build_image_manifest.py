"""Stage 1: Build image-level manifest and inventory across all splits.

Scans train, valid, and test splits to catalog all images, label counterparts,
dimensions, readability, and YOLO annotation counts by class.
"""

from pathlib import Path
import argparse
import csv
import re
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_source_prefix(image_path: Path) -> str:
    """Remove the Roboflow .rf.<hash> suffix to obtain the source prefix."""
    stem = image_path.stem
    return re.sub(r"\.rf\.[^.]+$", "", stem)


def inspect_label(label_path: Path):
    """Inspect YOLO label file for valid class IDs and coordinate bounds."""
    counts = {0: 0, 1: 0}
    annotation_count = 0
    invalid_rows = 0

    if not label_path.exists():
        return counts, 0, 0, False, True

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    empty_label = len(lines) == 0

    for line in lines:
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("Expected 5 YOLO values")

            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]

            if class_id not in counts:
                raise ValueError(f"Unexpected class ID {class_id}")

            if not all(0 <= value <= 1 for value in coords):
                raise ValueError("Coordinates outside [0, 1]")

            counts[class_id] += 1
            annotation_count += 1

        except (ValueError, OverflowError):
            invalid_rows += 1

    return counts, annotation_count, invalid_rows, empty_label, False


def build_manifest(root_dir: Path, output_file: Path):
    """Build dataset manifest and write to CSV."""
    rows = []

    for split in SPLITS:
        image_dir = root_dir / split / "images"
        label_dir = root_dir / split / "labels"

        if not image_dir.exists():
            print(f"WARNING: Missing image directory: {image_dir}")
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_readable = True
            except Exception:
                width, height = None, None
                image_readable = False

            (
                class_counts,
                annotation_count,
                invalid_rows,
                empty_label,
                missing_label,
            ) = inspect_label(label_path)

            rows.append({
                "split": split,
                "image_path": str(image_path.relative_to(root_dir)).replace("\\", "/"),
                "label_path": str(label_path.relative_to(root_dir)).replace("\\", "/"),
                "source_prefix": get_source_prefix(image_path),
                "width": width,
                "height": height,
                "image_readable": image_readable,
                "label_exists": not missing_label,
                "annotation_count": annotation_count,
                "crater_count": class_counts.get(0, 0),
                "rille_count": class_counts.get(1, 0),
                "invalid_label_rows": invalid_rows,
                "empty_label": empty_label,
            })

    output_file.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "split", "image_path", "label_path", "source_prefix",
        "width", "height", "image_readable", "label_exists",
        "annotation_count", "crater_count", "rille_count",
        "invalid_label_rows", "empty_label",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved to:\n{output_file}")
    print(f"Total images listed: {len(rows)}")

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        annotations = sum(row["annotation_count"] for row in split_rows)
        print(f"  {split:5}: {len(split_rows):4} images, {annotations:4} valid annotations")

    print(f"Images unreadable:  {sum(not row['image_readable'] for row in rows)}")
    print(f"Missing labels:     {sum(not row['label_exists'] for row in rows)}")
    print(f"Invalid label rows: {sum(row['invalid_label_rows'] for row in rows)}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build image manifest for Lunar CV dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root folder containing train/valid/test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "isro" / "automated_audit" / "batch_review" / "image_level_manifest.csv",
        help="Output path for the manifest CSV",
    )
    args = parser.parse_args()

    build_manifest(args.root, args.output)


if __name__ == "__main__":
    main()
