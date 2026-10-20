"""Stage 3: Annotation and Bounding-Box Distribution Statistics.

Computes comprehensive bounding-box metrics (percentiles, aspect ratios,
large-box thresholds) per class and per split.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import math
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
CLASS_NAMES = {0: "crater", 1: "rille"}


def percentile(values: list, p: float) -> float:
    """Nearest-rank percentile computation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def summarize_boxes(group: list) -> dict:
    """Compute summary statistics for a collection of bounding boxes."""
    areas = [r["area"] for r in group]
    widths = [r["width"] for r in group]
    heights = [r["height"] for r in group]
    ratios = [r["aspect_ratio"] for r in group]

    return {
        "count": len(group),
        "mean_area": round(statistics.mean(areas), 6) if areas else 0,
        "median_area": round(statistics.median(areas), 6) if areas else 0,
        "p90_area": round(percentile(areas, 0.90), 6),
        "p95_area": round(percentile(areas, 0.95), 6),
        "max_area": round(max(areas, default=0), 6),
        "median_width": round(statistics.median(widths), 6) if widths else 0,
        "median_height": round(statistics.median(heights), 6) if heights else 0,
        "median_aspect_ratio": round(statistics.median(ratios), 6) if ratios else 0,
        "boxes_area_ge_25pct": sum(a >= 0.25 for a in areas),
        "boxes_area_ge_50pct": sum(a >= 0.50 for a in areas),
    }


def compute_bbox_statistics(root_dir: Path, output_file: Path):
    """Scan labels across splits and compute bbox distribution statistics."""
    records = []
    invalid_rows = 0
    empty_label_files = 0

    for split in SPLITS:
        labels_dir = root_dir / split / "labels"
        if not labels_dir.exists():
            print(f"WARNING: Missing labels directory: {labels_dir}")
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_label_files += 1
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("Expected 5 YOLO values")

                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])

                    if class_id not in CLASS_NAMES:
                        raise ValueError(f"Unknown class ID: {class_id}")

                    if not all(math.isfinite(v) for v in (x, y, width, height)):
                        raise ValueError("Non-finite coordinate")

                    if width <= 0 or height <= 0:
                        raise ValueError("Non-positive box dimension")

                    area = width * height
                    aspect_ratio = width / height

                    records.append({
                        "split": split,
                        "image": label_path.stem,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "width": width,
                        "height": height,
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                    })

                except (ValueError, OverflowError):
                    invalid_rows += 1

    groups = defaultdict(list)
    for record in records:
        groups[("ALL", record["class_name"])].append(record)
        groups[(record["split"], record["class_name"])].append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split", "class_name", "count", "mean_area", "median_area",
        "p90_area", "p95_area", "max_area", "median_width",
        "median_height", "median_aspect_ratio",
        "boxes_area_ge_25pct", "boxes_area_ge_50pct"
    ]

    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for (split, class_name), group in sorted(groups.items()):
            writer.writerow({
                "split": split,
                "class_name": class_name,
                **summarize_boxes(group),
            })

    print("\n=== BOUNDING-BOX DISTRIBUTION STATISTICS ===")
    print(f"Valid boxes analyzed: {len(records)}")
    print(f"Invalid rows skipped: {invalid_rows}")
    print(f"Empty label files:    {empty_label_files}")

    for class_name in CLASS_NAMES.values():
        group = groups[("ALL", class_name)]
        stats = summarize_boxes(group)
        print(f"\n{class_name.upper()} ({stats['count']} boxes):")
        print(f"  Median area:         {stats['median_area']:.4f}")
        print(f"  90th percentile:     {stats['p90_area']:.4f}")
        print(f"  95th percentile:     {stats['p95_area']:.4f}")
        print(f"  Maximum area:        {stats['max_area']:.4f}")
        print(f"  Median aspect ratio: {stats['median_aspect_ratio']:.3f}")
        print(f"  Boxes >= 25% area:   {stats['boxes_area_ge_25pct']}")
        print(f"  Boxes >= 50% area:   {stats['boxes_area_ge_50pct']}")

    print(f"\nStatistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Compute bounding box statistics and distribution."
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
        default=PROJECT_ROOT / "reports" / "isro" / "bbox_distribution_statistics.csv",
        help="Output CSV path for bbox distribution statistics",
    )
    args = parser.parse_args()

    compute_bbox_statistics(args.root, args.output)


if __name__ == "__main__":
    main()
