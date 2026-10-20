"""Stage 5: Cross-Split Source-Overlap Screening.

Uses filename-prefix grouping, SIFT feature matching, and RANSAC homography
to screen for potential shared-source imagery across train, valid, and test splits.
Produces pairwise match metrics, group summaries, and visual contact sheets.
"""

from pathlib import Path
import argparse
import csv
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THUMB_W = 400
THUMB_H = 400
LABEL_H = 65


def resolve_image_path(root_dir: Path, item: str) -> Path:
    """Resolve image path from CSV entry (handling split/stem or split/images/stem)."""
    rel = Path(item.strip())
    if "images" in rel.parts:
        return root_dir / rel
    return root_dir / rel.parent / "images" / rel.name


def load_sift_features(path: Path, sift):
    """Load grayscale image and compute SIFT keypoints and descriptors."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def compare_images_sift(test_path: Path, train_path: Path, sift, matcher):
    """Compare two images with SIFT and RANSAC homography estimation."""
    test_kp, test_desc = load_sift_features(test_path, sift)
    train_kp, train_desc = load_sift_features(train_path, sift)

    result = {
        "test_keypoints": len(test_kp) if test_kp is not None else 0,
        "train_keypoints": len(train_kp) if train_kp is not None else 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "status": "insufficient_features",
    }

    if test_desc is None or train_desc is None:
        return result

    if len(test_desc) < 2 or len(train_desc) < 2:
        return result

    pairs = matcher.knnMatch(test_desc, train_desc, k=2)
    good = [
        m for pair in pairs if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.70 * n.distance
    ]

    result["good_matches"] = len(good)
    result["status"] = "too_few_matches"

    if len(good) < 4:
        return result

    src = np.float32([test_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([train_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    except cv2.error:
        return result

    if mask is None:
        result["status"] = "homography_failed"
        return result

    inliers = int(mask.sum())
    ratio = inliers / len(good) if good else 0.0

    result["inliers"] = inliers
    result["inlier_ratio"] = round(ratio, 4)

    if inliers >= 15 and ratio >= 0.50:
        result["status"] = "review_candidate"
    else:
        result["status"] = "weak_or_inconclusive"

    return result


def create_contact_sheet(image_entries: list, output_path: Path):
    """Create a labeled contact sheet without altering original images."""
    if not image_entries:
        return False

    columns = 2
    rows = math.ceil(len(image_entries) / columns)
    cell_h = THUMB_H + LABEL_H

    sheet = Image.new("RGB", (columns * THUMB_W, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (split, path) in enumerate(image_entries):
        if not path.exists():
            continue

        try:
            with Image.open(path) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (THUMB_W, THUMB_H))
        except Exception as exc:
            print(f"Could not open {path}: {exc}")
            continue

        x = (i % columns) * THUMB_W
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))

        label = f"{split}: {path.name}"
        draw.text((x + 5, y + THUMB_H + 4), label[:58], fill="black")
        draw.text((x + 5, y + THUMB_H + 23), label[58:116], fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def run_source_overlap_audit(root_dir: Path, output_dir: Path):
    """Execute cross-split source-overlap audit."""
    root = Path(root_dir).resolve()
    out_dir = Path(output_dir).resolve()
    sheets_dir = out_dir / "contact_sheets"

    out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    groups_csv = root / "reports" / "isro" / "automated_audit" / "cross_split_prefix_groups.csv"
    if not groups_csv.exists():
        raise FileNotFoundError(
            f"Missing required cross_split_prefix_groups.csv at {groups_csv}. "
            f"Please run duplicate_audit.py first."
        )

    with groups_csv.open(newline="", encoding="utf-8-sig") as f:
        groups = list(csv.DictReader(f))

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    train_test_pair_results = []
    train_test_group_results = []

    print("\n--- Screening Train-Test Cross-Split Groups ---")
    for group in groups:
        prefix = group.get("source_like_prefix", "")
        splits = {s.strip().lower() for s in group.get("splits", "").split(",")}

        if not {"train", "test"}.issubset(splits):
            continue

        entries = []
        for item in group.get("filenames", "").split("|"):
            item = item.strip()
            if not item:
                continue
            rel = Path(item)
            split = rel.parts[0] if rel.parts else "unknown"
            path = resolve_image_path(root, item)
            entries.append((split, path))

        train_entries = [(s, p) for s, p in entries if s == "train"]
        test_entries = [(s, p) for s, p in entries if s == "test"]

        safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)
        sheet_path = sheets_dir / f"{safe_prefix}.jpg"

        missing = [str(p) for _, p in entries if not p.exists()]
        sheet_ok = create_contact_sheet(entries, sheet_path)

        group_pairs = 0
        review_candidates = 0

        for _, test_path in test_entries:
            for _, train_path in train_entries:
                res = compare_images_sift(test_path, train_path, sift, matcher)
                group_pairs += 1
                if res["status"] == "review_candidate":
                    review_candidates += 1

                train_test_pair_results.append({
                    "source_like_prefix": prefix,
                    "test_image": str(test_path.relative_to(root)).replace("\\", "/"),
                    "train_image": str(train_path.relative_to(root)).replace("\\", "/"),
                    **res,
                })

        train_test_group_results.append({
            "source_like_prefix": prefix,
            "train_images": len(train_entries),
            "test_images": len(test_entries),
            "pairs_compared": group_pairs,
            "review_candidates": review_candidates,
            "missing_images": len(missing),
            "contact_sheet": str(sheet_path.relative_to(root)).replace("\\", "/") if sheet_ok else "",
            "review_status": "check_missing_paths" if missing else "manual_review_required",
            "missing_paths": " | ".join(missing),
        })

        print(f"  {prefix}: {group_pairs} pairs, {review_candidates} review candidate(s)")

    # Screen Train-Validation groups (specifically high-interest groups)
    train_valid_targets = {
        "Screenshot-2022-02-25-16-19-45_png_jpg",
        "Screenshot-2022-02-25-16-20-17_png_jpg",
        "Screenshot-2022-02-25-16-20-20_png_jpg",
        "Screenshot-2022-02-25-16-20-58_png_jpg",
        "images-_10__jpg",
    }
    train_valid_results = []

    print("\n--- Screening Train-Validation Priority Groups ---")
    manifest_csv = out_dir / "image_level_manifest.csv"
    if manifest_csv.exists():
        with manifest_csv.open(encoding="utf-8", newline="") as f:
            manifest_rows = list(csv.DictReader(f))

        selected = [r for r in manifest_rows if r["source_prefix"] in train_valid_targets]
        for prefix in sorted(train_valid_targets):
            train_imgs = [r for r in selected if r["source_prefix"] == prefix and r["split"] == "train"]
            valid_imgs = [r for r in selected if r["source_prefix"] == prefix and r["split"] == "valid"]

            for valid in valid_imgs:
                for train in train_imgs:
                    v_path = root / valid["image_path"]
                    t_path = root / train["image_path"]
                    res = compare_images_sift(v_path, t_path, sift, matcher)

                    train_valid_results.append({
                        "source_prefix": prefix,
                        "train_image": train["image_path"],
                        "valid_image": valid["image_path"],
                        "good_matches": res["good_matches"],
                        "inliers": res["inliers"],
                        "inlier_ratio": res["inlier_ratio"],
                    })

    # Save outputs
    pair_csv = out_dir / "train_test_sift_pair_results.csv"
    group_csv = out_dir / "train_test_group_summary.csv"
    valid_csv = out_dir / "train_valid_sift_review.csv"

    pair_fields = [
        "source_like_prefix", "test_image", "train_image",
        "test_keypoints", "train_keypoints", "good_matches",
        "inliers", "inlier_ratio", "status",
    ]
    group_fields = [
        "source_like_prefix", "train_images", "test_images",
        "pairs_compared", "review_candidates", "missing_images",
        "contact_sheet", "review_status", "missing_paths",
    ]
    valid_fields = [
        "source_prefix", "train_image", "valid_image",
        "good_matches", "inliers", "inlier_ratio",
    ]

    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(train_test_pair_results)

    with group_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(train_test_group_results)

    if train_valid_results:
        with valid_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=valid_fields)
            writer.writeheader()
            writer.writerows(train_valid_results)

    print("\n=== SOURCE OVERLAP AUDIT COMPLETE ===")
    print(f"Train-test groups checked: {len(train_test_group_results)}")
    print(f"Train-test pairs evaluated: {len(train_test_pair_results)}")
    print(f"Train-valid pairs evaluated: {len(train_valid_results)}")
    print(f"Results saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Cross-split SIFT and homography screening."
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
        default=PROJECT_ROOT / "reports" / "isro" / "automated_audit" / "batch_review",
        help="Output directory for batch review CSVs and contact sheets",
    )
    args = parser.parse_args()

    run_source_overlap_audit(args.root, args.output)


if __name__ == "__main__":
    main()
