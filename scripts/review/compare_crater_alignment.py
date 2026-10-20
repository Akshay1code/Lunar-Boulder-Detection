"""Homography alignment and difference image generation for 15-03-38 candidate pair."""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
images_dir = ROOT / "train" / "images"
stem_a = "Screenshot-2022-02-25-15-03-38_png_jpg.rf.feaa5a009b016bc875752daff1d47d8c"
stem_b = "Screenshot-2022-02-25-15-03-38_png_jpg.rf.e695043607ae553a9a8a82e1b7ab80fa"


def load_image(stem):
    paths = []
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        paths.extend(images_dir.glob(stem + ext))
    if not paths:
        raise FileNotFoundError(stem)
    image = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {paths[0]}")
    return image


a = load_image(stem_a)
b = load_image(stem_b)

detector = cv2.SIFT_create()
kp_a, des_a = detector.detectAndCompute(a, None)
kp_b, des_b = detector.detectAndCompute(b, None)

matcher = cv2.BFMatcher(cv2.NORM_L2)
pairs = matcher.knnMatch(des_a, des_b, k=2)
good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]

if len(good) < 4:
    raise SystemExit("Too few matches for alignment.")

src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
if H is None or mask is None:
    raise SystemExit("Homography estimation failed.")

inliers = int(mask.sum())
height, width = b.shape

aligned_a = cv2.warpPerspective(a, H, (width, height))
valid_a = cv2.warpPerspective(np.ones(a.shape, dtype=np.uint8), H, (width, height))
valid_b = np.ones(b.shape, dtype=np.uint8)
overlap = (valid_a > 0) & (valid_b > 0)
overlap_pixels = int(overlap.sum())
overlap_pct = 100 * overlap_pixels / (width * height)

if overlap_pixels:
    diff = cv2.absdiff(aligned_a, b)
    mae = float(diff[overlap].mean())
else:
    mae = float("nan")

out = ROOT / "reports/isro/visualizations"
out.mkdir(parents=True, exist_ok=True)

overlay = cv2.addWeighted(aligned_a, 0.5, b, 0.5, 0)
overlay[~overlap] = 0
cv2.imwrite(str(out / "crater_aligned_overlay.jpg"), overlay)

diff_view = cv2.absdiff(aligned_a, b)
diff_view[~overlap] = 0
cv2.imwrite(str(out / "crater_aligned_difference.jpg"), diff_view)

print("Good matches:", len(good))
print("Geometric inliers:", inliers)
print(f"Inlier ratio: {inliers / len(good):.2%}")
print(f"Overlap of image B: {overlap_pct:.2f}%")
print(f"Mean absolute difference within overlap: {mae:.2f}/255")
print("Saved:", out / "crater_aligned_overlay.jpg")
print("Saved:", out / "crater_aligned_difference.jpg")
