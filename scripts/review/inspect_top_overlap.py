"""Generate contact sheet for top train-valid overlap pairs.

Visualizes unique training images and the shared validation image for:
Screenshot-2022-02-25-16-19-45_png_jpg
"""

from pathlib import Path
from PIL import Image, ImageDraw
import csv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reports/isro/automated_audit/batch_review/train_valid_sift_review.csv"
OUT_PATH = ROOT / "reports/isro/automated_audit/batch_review/contact_sheets/top_train_valid_overlap.jpg"

PREFIX = "Screenshot-2022-02-25-16-19-45_png_jpg"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")

with CSV_PATH.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

rows = [row for row in rows if row["source_prefix"] == PREFIX]
rows.sort(key=lambda row: int(row["inliers"]), reverse=True)

if not rows:
    raise SystemExit("No matching pairs found in the CSV.")

paths = []
for row in rows[:3]:
    paths.append(("TRAIN", ROOT / row["train_image"]))

paths.append(("VALID", ROOT / rows[0]["valid_image"]))

thumb_w, thumb_h = 320, 360
sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 2), "white")
draw = ImageDraw.Draw(sheet)

for i, (split, path) in enumerate(paths):
    if not path.exists():
        print("Missing:", path)
        continue

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((thumb_w - 20, thumb_h - 55))

    x = (i % 2) * thumb_w
    y = (i // 2) * thumb_h

    sheet.paste(im, (x + 10, y + 35))
    draw.text((x + 10, y + 8), f"{split}: {path.name[:32]}", fill="black")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT_PATH, quality=95)
print("Contact sheet saved:", OUT_PATH)
