import cv2
import csv
import numpy as np
from pathlib import Path
from itertools import combinations

ROOT = Path(".")
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OUT = ROOT / "reports/isro/automated_audit/feature_match_candidates.csv"

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)

images = []
for split in SPLITS:
    folder = ROOT / split / "images"
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            images.append((split, path))

print("Images found:", len(images))
print("Extracting image features...")

records = []
for i, (split, path) in enumerate(images, 1):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Unreadable:", path)
        continue

    kp, desc = detector.detectAndCompute(img, None)
    records.append({
        "split": split,
        "path": path,
        "image": img,
        "keypoints": kp,
        "descriptors": desc,
    })

    if i % 200 == 0:
        print(f"Processed {i}/{len(images)}")

print("Readable images:", len(records))
print("Comparing image features. This may take a while...")

OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "split_a", "image_a", "split_b", "image_b",
    "good_matches", "inliers", "inlier_ratio",
    "overlap_pct", "cross_split"
]

candidate_count = 0
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for i, j in combinations(range(len(records)), 2):
        a, b = records[i], records[j]

        # Prioritize likely matches; skip images without descriptors.
        if a["descriptors"] is None or b["descriptors"] is None:
            continue

        pairs = matcher.knnMatch(a["descriptors"], b["descriptors"], k=2)
        good = [
            m for pair in pairs if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.7 * n.distance
        ]

        if len(good) < 30:
            continue

        src = np.float32(
            [a["keypoints"][m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [b["keypoints"][m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inliers = int(mask.sum())
        ratio = inliers / len(good)

        # Keep candidates for review, not automatic duplicate declarations.
        if inliers < 25 or ratio < 0.50:
            continue

        h, w = b["image"].shape
        warped_mask = cv2.warpPerspective(
            np.ones(a["image"].shape, dtype=np.uint8), H, (w, h),
            flags=cv2.INTER_NEAREST
        )
        overlap = 100 * np.count_nonzero(warped_mask) / (w * h)
        cross_split = a["split"] != b["split"]

        writer.writerow({
            "split_a": a["split"],
            "image_a": a["path"].name,
            "split_b": b["split"],
            "image_b": b["path"].name,
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": round(ratio, 4),
            "overlap_pct": round(overlap, 2),
            "cross_split": cross_split,
        })
        candidate_count += 1

        if candidate_count % 100 == 0:
            print("Candidate pairs saved:", candidate_count)

print("Audit complete.")
print("Candidate pairs:", candidate_count)
print("Report:", OUT)
