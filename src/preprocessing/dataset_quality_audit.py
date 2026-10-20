"""Stage 2: Dataset Quality Audit.

Validates image/label counterparts, YAML class mappings, coordinate bounds,
box geometries, and produces split and class distribution summaries.
"""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LARGE_BOX_REVIEW_THRESHOLD = 0.25
HIGH_PRIORITY_BOX_THRESHOLD = 0.50


def write_csv(path: Path, fieldnames: list, rows: list):
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_class_names(dataset_root: Path) -> list:
    """Read class names and count from data.yaml."""
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml at {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]

    if not isinstance(names, list) or not names:
        raise ValueError(f"Could not parse valid class names in {yaml_path}")

    expected_count = config.get("nc", len(names))
    if expected_count != len(names):
        raise ValueError(
            f"data.yaml declares nc={expected_count}, but contains {len(names)} class names."
        )

    print("Classes found in data.yaml:")
    for class_id, name in enumerate(names):
        print(f"  {class_id}: {name}")

    return names


def run_quality_audit(dataset_root: Path, output_dir: Path):
    """Run full structural quality audit on dataset."""
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    class_names = load_class_names(dataset_root)

    issues = []
    split_rows = []
    bbox_records = []
    counts_by_split = {split: Counter() for split in SPLITS}
    image_dimensions = Counter()
    image_extensions = Counter()
    valid_boxes_by_class = defaultdict(list)

    def add_issue(split, filename, issue_type, details):
        issues.append({
            "split": split,
            "file": str(filename),
            "issue": issue_type,
            "details": details,
        })

    for split in SPLITS:
        split_dir = dataset_root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.is_dir():
            add_issue(split, images_dir, "MISSING_IMAGES_DIRECTORY", "Expected an images directory.")
        if not labels_dir.is_dir():
            add_issue(split, labels_dir, "MISSING_LABELS_DIRECTORY", "Expected a labels directory.")

        images = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ) if images_dir.is_dir() else []

        labels = sorted(
            labels_dir.rglob("*.txt")
        ) if labels_dir.is_dir() else []

        image_by_stem = {p.stem: p for p in images}
        label_by_stem = {p.stem: p for p in labels}
        image_sizes = {}
        unreadable_images = 0
        empty_label_files = 0
        annotations = 0

        for image_path in images:
            image_extensions[image_path.suffix.lower()] += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                image_sizes[image_path.stem] = (width, height)
                image_dimensions[f"{width}x{height}"] += 1
            except Exception as error:
                unreadable_images += 1
                add_issue(split, image_path, "UNREADABLE_IMAGE", str(error))

            if image_path.stem not in label_by_stem:
                add_issue(split, image_path, "MISSING_LABEL_FILE", "No matching label file found.")

        for label_path in labels:
            if label_path.stem not in image_by_stem:
                add_issue(split, label_path, "LABEL_WITHOUT_IMAGE", "No matching image found.")

        for label_path in labels:
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                add_issue(split, label_path, "UNREADABLE_LABEL", str(error))
                continue

            object_lines = [
                (line_number, line.strip())
                for line_number, line in enumerate(lines, start=1)
                if line.strip()
            ]

            if not object_lines:
                empty_label_files += 1
                add_issue(split, label_path, "EMPTY_LABEL_FILE", "The label file contains no annotations.")
                continue

            dimensions = image_sizes.get(label_path.stem)
            image_width, image_height = dimensions or (None, None)

            for line_number, line in object_lines:
                parts = line.split()
                if len(parts) != 5:
                    add_issue(split, label_path, "MALFORMED_ANNOTATION", f"Line {line_number}: expected 5 values, found {len(parts)}.")
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, box_w, box_h = map(float, parts[1:])
                except ValueError:
                    add_issue(split, label_path, "NON_NUMERIC_ANNOTATION", f"Line {line_number}: invalid numeric values.")
                    continue

                class_is_valid = 0 <= class_id < len(class_names)
                if not class_is_valid:
                    add_issue(split, label_path, "INVALID_CLASS_ID", f"Line {line_number}: class ID {class_id} not defined in data.yaml.")
                else:
                    counts_by_split[split][class_id] += 1
                    annotations += 1

                coords = [cx, cy, box_w, box_h]
                finite = all(math.isfinite(value) for value in coords)
                box_is_valid = finite and class_is_valid

                if not finite:
                    add_issue(split, label_path, "NON_FINITE_COORDINATE", f"Line {line_number}: NaN or inf coordinate.")
                    box_is_valid = False
                    continue

                if any(value < 0 or value > 1 for value in coords):
                    add_issue(split, label_path, "COORDINATE_OUT_OF_RANGE", f"Line {line_number}: coordinates outside [0, 1].")
                    box_is_valid = False

                if box_w <= 0 or box_h <= 0:
                    add_issue(split, label_path, "NON_POSITIVE_BOX_SIZE", f"Line {line_number}: width and height must be positive.")
                    box_is_valid = False

                if (cx - box_w / 2 < 0 or cy - box_h / 2 < 0 or cx + box_w / 2 > 1 or cy + box_h / 2 > 1):
                    add_issue(split, label_path, "BOX_OUTSIDE_IMAGE", f"Line {line_number}: box extends beyond image bounds.")
                    box_is_valid = False

                if not class_is_valid or not box_is_valid:
                    continue

                width_px = box_w * image_width if image_width else None
                height_px = box_h * image_height if image_height else None
                area = box_w * box_h

                box = {
                    "split": split,
                    "image": label_path.stem,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "center_x_normalized": cx,
                    "center_y_normalized": cy,
                    "width_normalized": box_w,
                    "height_normalized": box_h,
                    "relative_area": area,
                    "width_pixels": width_px,
                    "height_pixels": height_px,
                }
                bbox_records.append(box)
                valid_boxes_by_class[class_id].append(box)

        split_rows.append({
            "split": split,
            "images": len(images),
            "label_files": len(labels),
            "annotations_with_valid_class_id": annotations,
            "missing_label_files": sum(1 for p in images if p.stem not in label_by_stem),
            "empty_label_files": empty_label_files,
            "labels_without_images": sum(1 for p in labels if p.stem not in image_by_stem),
            "unreadable_images": unreadable_images,
        })

    class_rows = []
    for split in SPLITS:
        total = sum(counts_by_split[split].values())
        for class_id, class_name in enumerate(class_names):
            count = counts_by_split[split][class_id]
            class_rows.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "annotation_count": count,
                "percentage": round(100 * count / total, 2) if total else 0,
            })

    bbox_stats = []
    for class_id, class_name in enumerate(class_names):
        boxes = valid_boxes_by_class[class_id]

        def avg(key):
            values = [box[key] for box in boxes if box[key] is not None]
            return round(statistics.mean(values), 6) if values else ""

        bbox_stats.append({
            "class_id": class_id,
            "class_name": class_name,
            "valid_box_count": len(boxes),
            "mean_width_normalized": avg("width_normalized"),
            "mean_height_normalized": avg("height_normalized"),
            "mean_relative_area": avg("relative_area"),
            "mean_width_pixels": avg("width_pixels"),
            "mean_height_pixels": avg("height_pixels"),
        })

    large_box_rows = []
    for box in bbox_records:
        area = box["relative_area"]
        if area >= LARGE_BOX_REVIEW_THRESHOLD:
            large_box_rows.append({
                "split": box["split"],
                "image": box["image"],
                "class_id": box["class_id"],
                "class_name": box["class_name"],
                "center_x_normalized": box["center_x_normalized"],
                "center_y_normalized": box["center_y_normalized"],
                "width_normalized": box["width_normalized"],
                "height_normalized": box["height_normalized"],
                "relative_area": round(area, 6),
                "relative_area_percent": round(area * 100, 2),
                "review_priority": "HIGH" if area >= HIGH_PRIORITY_BOX_THRESHOLD else "REVIEW",
                "review_reason": "Covers at least 50% of image" if area >= HIGH_PRIORITY_BOX_THRESHOLD else "Covers 25% to less than 50% of image",
                "width_pixels": box["width_pixels"],
                "height_pixels": box["height_pixels"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "split_summary.csv", list(split_rows[0].keys()), split_rows)
    write_csv(output_dir / "class_distribution.csv", ["split", "class_id", "class_name", "annotation_count", "percentage"], class_rows)
    write_csv(output_dir / "bbox_statistics.csv", [
        "class_id", "class_name", "valid_box_count",
        "mean_width_normalized", "mean_height_normalized",
        "mean_relative_area", "mean_width_pixels", "mean_height_pixels"
    ], bbox_stats)
    write_csv(output_dir / "large_bbox_review.csv", [
        "split", "image", "class_id", "class_name",
        "center_x_normalized", "center_y_normalized",
        "width_normalized", "height_normalized",
        "relative_area", "relative_area_percent",
        "review_priority", "review_reason",
        "width_pixels", "height_pixels",
    ], large_box_rows)
    write_csv(output_dir / "issues.csv", ["split", "file", "issue", "details"], issues)

    summary = {
        "dataset_root": str(dataset_root),
        "classes": {str(i): name for i, name in enumerate(class_names)},
        "splits": split_rows,
        "image_dimensions": dict(image_dimensions),
        "image_extensions": dict(image_extensions),
        "class_distribution": class_rows,
        "issue_counts": dict(Counter(issue["issue"] for issue in issues)),
        "total_issue_records": len(issues),
        "valid_bbox_statistics": bbox_stats,
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== DATASET QUALITY AUDIT SUMMARY ===")
    for row in split_rows:
        print(f"  {row['split']:5} | Images: {row['images']:4} | Labels: {row['label_files']:4} | Annotations: {row['annotations_with_valid_class_id']:4}")

    print("\nClass counts:")
    for row in class_rows:
        print(f"  {row['split']:5} | {row['class_name']:10} | {row['annotation_count']:5} ({row['percentage']}%)")

    print(f"\nIssue records: {len(issues)}")
    print(f"Large bounding boxes flagged for review: {len(large_box_rows)}")
    print(f"Reports saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Comprehensive YOLO dataset quality audit."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root folder containing data.yaml and splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "isro",
        help="Output directory for audit reports",
    )
    args = parser.parse_args()

    run_quality_audit(args.root, args.output)


if __name__ == "__main__":
    main()
