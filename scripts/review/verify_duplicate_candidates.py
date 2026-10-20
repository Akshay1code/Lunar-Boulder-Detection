"""Verify near-duplicate candidates with SIFT and homography."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
candidate_file = ROOT / "reports/isro/automated_audit/near_duplicate_candidates.csv"
output_file = ROOT / "reports/isro/automated_audit/verified_near_duplicate_candidates.csv"

if not candidate_file.exists():
    raise FileNotFoundError(f"Missing input candidate file: {candidate_file}")

df = pd.read_csv(candidate_file)
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)
cache = {}


def load_features(relative_path):
    if relative_path not in cache:
        path = ROOT / Path(relative_path)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            cache[relative_path] = None
        else:
            kp, desc = detector.detectAndCompute(img, None)
            cache[relative_path] = (img, kp, desc)
    return cache[relative_path]


results = []
total = len(df)

for i, row in df.iterrows():
    a_path, b_path = row["image_a"], row["image_b"]
    a = load_features(a_path)
    b = load_features(b_path)

    result = {
        "image_a": a_path,
        "split_a": row["split_a"],
        "image_b": b_path,
        "split_b": row["split_b"],
        "hamming_distance": row["hamming_distance"],
        "cross_split": row["cross_split"],
        "same_filename_prefix": row.get("same_filename_prefix", False),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "overlap_pct": 0.0,
        "mean_abs_diff": "",
        "status": "insufficient_features",
    }

    if a and b and a[2] is not None and b[2] is not None:
        img_a, kp_a, des_a = a
        img_b, kp_b, des_b = b
        pairs = matcher.knnMatch(des_a, des_b, k=2)
        good = [
            m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < 0.7 * n.distance
        ]
        result["good_matches"] = len(good)

        if len(good) >= 4:
            src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            if H is not None and mask is not None:
                inliers = int(mask.sum())
                result["inliers"] = inliers
                result["inlier_ratio"] = round(inliers / len(good), 4)

                h, w = img_b.shape
                warped = cv2.warpPerspective(img_a, H, (w, h))
                valid = cv2.warpPerspective(
                    np.ones(img_a.shape, dtype=np.uint8), H, (w, h),
                    flags=cv2.INTER_NEAREST
                ) > 0
                overlap_pct = 100 * valid.sum() / (w * h)
                result["overlap_pct"] = round(float(overlap_pct), 2)

                if valid.any():
                    result["mean_abs_diff"] = round(
                        float(cv2.absdiff(warped, img_b)[valid].mean()), 2
                    )

                if inliers >= 25 and result["inlier_ratio"] >= 0.5:
                    result["status"] = "strong_geometric_match"
                else:
                    result["status"] = "weak_geometric_match"

    results.append(result)

out = pd.DataFrame(results)
out = out.sort_values(
    ["cross_split", "inliers", "inlier_ratio"],
    ascending=[False, False, False]
)
output_file.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_file, index=False)

print("\nAudit complete.")
print("Candidate pairs:", len(out))
print("Strong geometric matches:", (out["status"] == "strong_geometric_match").sum())
print("Saved:", output_file)
