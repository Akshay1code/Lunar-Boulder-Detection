"""Inspect detailed SIFT inlier correspondences between train and valid images.

Draws geometric inlier match lines between:
- train: Screenshot-2022-02-25-16-19-45_png_jpg.rf.e816dd9935a61b07a447dbb4b021a053.jpg
- valid: Screenshot-2022-02-25-16-19-45_png_jpg.rf.bf5eb30447c28884dd4b8a63e6045ef4.jpg
"""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

train_path = ROOT / (
    "train/images/"
    "Screenshot-2022-02-25-16-19-45_png_jpg."
    "rf.e816dd9935a61b07a447dbb4b021a053.jpg"
)

valid_path = ROOT / (
    "valid/images/"
    "Screenshot-2022-02-25-16-19-45_png_jpg."
    "rf.bf5eb30447c28884dd4b8a63e6045ef4.jpg"
)

output_path = ROOT / (
    "reports/isro/automated_audit/batch_review/"
    "contact_sheets/sift_inlier_matches.jpg"
)

img1 = cv2.imread(str(train_path), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise SystemExit("Could not read one of the images. Check the paths.")

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    raise SystemExit("Not enough visual features to compare.")

matcher = cv2.BFMatcher(cv2.NORM_L2)
pairs = matcher.knnMatch(des1, des2, k=2)

good = [
    pair[0] for pair in pairs
    if len(pair) == 2
    and pair[0].distance < 0.75 * pair[1].distance
]

if len(good) < 4:
    raise SystemExit(f"Only {len(good)} good matches; cannot estimate geometry.")

src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

if H is None or mask is None:
    raise SystemExit("Could not estimate a reliable geometric transform.")

inlier_matches = [
    match for match, keep in zip(good, mask.ravel()) if keep
]

visual = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), visual)

print("Good matches:", len(good))
print("Geometric inliers:", len(inlier_matches))
print("Inlier ratio:", round(len(inlier_matches) / len(good), 4))
print("Saved visualization:", output_path)
