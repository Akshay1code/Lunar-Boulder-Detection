"""Detect and describe local SIFT features for Screenshot-2022-02-25-15-03-38 pair."""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
images_dir = ROOT / "train" / "images"

a_path = next(images_dir.glob("Screenshot-2022-02-25-15-03-38_png_jpg.rf.feaa5a009b016bc875752daff1d47d8c.*"))
b_path = next(images_dir.glob("Screenshot-2022-02-25-15-03-38_png_jpg.rf.e695043607ae553a9a8a82e1b7ab80fa.*"))

a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)

if a is None or b is None:
    raise SystemExit("Could not load one or both images.")

detector = cv2.SIFT_create()
kp_a, des_a = detector.detectAndCompute(a, None)
kp_b, des_b = detector.detectAndCompute(b, None)

print("Keypoints A:", len(kp_a))
print("Keypoints B:", len(kp_b))

if des_a is None or des_b is None:
    raise SystemExit("Not enough detectable features to compare.")

matcher = cv2.BFMatcher(cv2.NORM_L2)
pairs = matcher.knnMatch(des_a, des_b, k=2)

good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]
print("Good feature matches:", len(good))

if len(good) >= 4:
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    print("Geometric inliers:", inliers)
    print("Homography found:", H is not None)

    if H is not None:
        overlay = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        out = ROOT / "reports/isro/visualizations/crater_feature_overlay.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), overlay)
        print("Saved overlay:", out)
else:
    print("Too few matches for reliable geometric verification.")
