"""Stage 4: Exact and Perceptual Duplicate Image Screening.

Performs byte-level SHA-256 and 64-bit perceptual difference hashing (dHash)
across all dataset images. Identifies exact duplicates, near-duplicate
candidates, and cross-split filename-prefix groups.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import csv
import hashlib
import json
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 digest of file contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def difference_hash(path: Path) -> int:
    """Calculate 64-bit perceptual difference hash using Pillow."""
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        if hasattr(image, "get_flattened_data"):
            pixels = list(image.get_flattened_data())
        else:
            pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return value


def write_csv(path: Path, columns: list, rows: list):
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_duplicate_audit(root_dir: Path, output_dir: Path, threshold: int = 4):
    """Run duplicate and near-duplicate audit."""
    root = Path(root_dir).resolve()
    output = Path(output_dir).resolve()

    records = []
    unreadable = []

    print(f"Scanning images in {root}...")
    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            print(f"Warning: missing folder {images_dir}")
            continue

        images = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"  {split}: {len(images)} images found")

        for path in images:
            try:
                rel_path = str(path.relative_to(root)).replace("\\", "/")
                file_hash = sha256_file(path)
                dhash_val = difference_hash(path)

                records.append({
                    "path": path,
                    "rel_path": rel_path,
                    "name": path.name,
                    "split": split,
                    "sha256": file_hash,
                    "dhash": dhash_val,
                })
            except Exception as error:
                unreadable.append({
                    "split": split,
                    "image": str(path.relative_to(root)).replace("\\", "/"),
                    "error": str(error),
                })

    print(f"Total images scanned successfully: {len(records)}")

    # 1. Exact byte duplicates (SHA-256)
    exact_groups = defaultdict(list)
    for record in records:
        exact_groups[record["sha256"]].append(record)

    exact_rows = []
    for digest, group in exact_groups.items():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                exact_rows.append({
                    "image_a": a["rel_path"],
                    "split_a": a["split"],
                    "image_b": b["rel_path"],
                    "split_b": b["split"],
                    "sha256": digest,
                    "cross_split": a["split"] != b["split"],
                })

    # 2. Perceptual near-duplicate screening (dHash Hamming distance <= threshold)
    near_rows = []
    for i in range(len(records)):
        a = records[i]
        for j in range(i + 1, len(records)):
            b = records[j]

            if a["sha256"] == b["sha256"]:
                continue

            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if distance <= threshold:
                same_prefix = (
                    a["name"].split(".rf.")[0] == b["name"].split(".rf.")[0]
                )
                near_rows.append({
                    "image_a": a["rel_path"],
                    "split_a": a["split"],
                    "image_b": b["rel_path"],
                    "split_b": b["split"],
                    "hamming_distance": distance,
                    "cross_split": a["split"] != b["split"],
                    "same_filename_prefix": same_prefix,
                })

    # 3. Source-like filename prefixes across splits
    prefix_groups = defaultdict(list)
    for record in records:
        prefix = record["name"].split(".rf.")[0]
        prefix_groups[prefix].append(record)

    prefix_rows = []
    for prefix, group in sorted(prefix_groups.items()):
        splits = sorted({item["split"] for item in group})
        if len(splits) > 1:
            prefix_rows.append({
                "source_like_prefix": prefix,
                "splits": ", ".join(splits),
                "image_count": len(group),
                "filenames": " | ".join(
                    f"{item['split']}/{item['name']}" for item in group
                ),
            })

    # Write primary automated_audit outputs
    output.mkdir(parents=True, exist_ok=True)

    write_csv(
        output / "exact_duplicate_pairs.csv",
        ["image_a", "split_a", "image_b", "split_b", "sha256", "cross_split"],
        exact_rows,
    )
    write_csv(
        output / "near_duplicate_candidates.csv",
        ["image_a", "split_a", "image_b", "split_b", "hamming_distance", "cross_split", "same_filename_prefix"],
        near_rows,
    )
    write_csv(
        output / "cross_split_prefix_groups.csv",
        ["source_like_prefix", "splits", "image_count", "filenames"],
        prefix_rows,
    )
    write_csv(
        output / "unreadable_images.csv",
        ["split", "image", "error"],
        unreadable,
    )

    summary = {
        "images_discovered": len(records) + len(unreadable),
        "images_hashed": len(records),
        "unreadable_images": len(unreadable),
        "exact_duplicate_pairs": len(exact_rows),
        "exact_cross_split_pairs": sum(r["cross_split"] for r in exact_rows),
        "near_duplicate_candidates": len(near_rows),
        "near_cross_split_candidates": sum(r["cross_split"] for r in near_rows),
        "cross_split_prefix_groups": len(prefix_rows),
        "perceptual_hash": "64-bit difference hash (dHash)",
        "candidate_threshold_hamming_distance": threshold,
        "warning": "Similarity candidates require verification. No dataset files were modified.",
    }

    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirror compatibility outputs to reports/isro/duplicates/ if directory exists or as needed
    duplicates_dir = root / "reports" / "isro" / "duplicates"
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        duplicates_dir / "exact_duplicates.csv",
        ["image_a", "split_a", "image_b", "split_b", "cross_split", "sha256"],
        exact_rows,
    )
    write_csv(
        duplicates_dir / "near_duplicate_candidates.csv",
        ["image_a", "split_a", "image_b", "split_b", "cross_split", "hamming_distance"],
        [{k: r[k] for k in ["image_a", "split_a", "image_b", "split_b", "cross_split", "hamming_distance"]} for r in near_rows],
    )
    dup_summary = {
        "dataset_root": str(root),
        "images_scanned_successfully": len(records),
        "unreadable_images": len(unreadable),
        "exact_duplicate_pairs": len(exact_rows),
        "near_duplicate_candidate_pairs": len(near_rows),
        "near_duplicate_threshold": threshold,
        "exact_cross_split_pairs": sum(r["cross_split"] for r in exact_rows),
        "near_cross_split_pairs": sum(r["cross_split"] for r in near_rows),
        "note": "Near-duplicate pairs are candidates only. Inspect them before making any dataset changes.",
    }
    with (duplicates_dir / "duplicate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dup_summary, f, indent=2)

    print("\n=== DUPLICATE SCREENING SUMMARY ===")
    print(f"Images hashed:               {len(records)}")
    print(f"Unreadable images:           {len(unreadable)}")
    print(f"Exact duplicate pairs:       {len(exact_rows)}")
    print(f"Near-duplicate candidates:   {len(near_rows)}")
    print(f"Cross-split near candidates: {summary['near_cross_split_candidates']}")
    print(f"Cross-split prefix groups:   {len(prefix_rows)}")
    print(f"Outputs written to:          {output}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Exact and perceptual near-duplicate audit."
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
        default=PROJECT_ROOT / "reports" / "isro" / "automated_audit",
        help="Output folder for automated audit outputs",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Maximum dHash Hamming distance threshold (default: 4)",
    )
    args = parser.parse_args()

    run_duplicate_audit(args.root, args.output, args.threshold)


if __name__ == "__main__":
    main()
