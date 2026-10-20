"""Visualizes SIFT matches between the unresolved validation-test candidate pair:

valid/images/4165_jpg.rf.cae5c14fc0e6abbc22135c127a531299.jpg vs
test/images/4191_jpg.rf.266ac35f07499b841f2da6d2056b512d.jpg
"""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

a_path = ROOT / "valid/images/4165_jpg.rf.cae5c14fc0e6abbc22135c127a531299.jpg"
b_path = ROOT / "test/images/4191_jpg.rf.266ac35f07499b841f2da6d2056b512d.jpg"

a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)

if a is None or b is None:
    raise SystemExit("Could not load one or both images.")

sift = cv2.SIFT_create()
ka, da = sift.detectAndCompute(a, None)
kb, db = sift.detectAndCompute(b, None)

if da is None or db is None:
    raise SystemExit("Not enough features to compare.")

matcher = cv2.BFMatcher(cv2.NORM_L2)
pairs = matcher.knnMatch(da, db, k=2)

good = [
    m for pair in pairs if len(pair) == 2
    for m, n in [pair]
    if m.distance < 0.85 * n.distance
]

print("Matches at ratio 0.85:", len(good))

if len(good) >= 4:
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        raise SystemExit("Could not estimate a reliable homography.")

    inliers = int(mask.sum())
    print("Geometric inliers:", inliers)

    inlier_matches = [m for m, keep in zip(good, mask.ravel()) if keep]
    print("Matches displayed:", len(inlier_matches))

    drawn = cv2.drawMatches(
        a, ka, b, kb,
        inlier_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    out = ROOT / "reports/isro/visualizations/cross_split_inliers.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), drawn)
    print("Saved visualization to:", out)
else:
    print("Too few matches for a meaningful visualization.")
