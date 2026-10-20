"""Create contact sheet for all high-priority large crater bounding boxes."""

from pathlib import Path
import csv
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT / "reports/isro/large_bbox_review.csv"
output = ROOT / "reports/isro/visualizations/high_priority_craters.jpg"

if not csv_path.exists():
    raise FileNotFoundError(f"Missing CSV at {csv_path}")

with csv_path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r["class_name"] == "crater" and r["review_priority"] == "HIGH"]

tiles = []
font = ImageFont.load_default()

for row in rows:
    split, stem = row["split"], row["image"]
    matches = list((ROOT / split / "images").glob(stem + ".*"))
    label_path = ROOT / split / "labels" / (stem + ".txt")

    if not matches or not label_path.is_file():
        print("Missing image/label:", split, stem)
        continue

    with Image.open(matches[0]) as im:
        image = im.convert("RGB")

    draw = ImageDraw.Draw(image)
    w, h = image.size

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        if cls not in (0, 1) or not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            continue

        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = (255, 50, 50) if cls == 0 else (30, 220, 80)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    image.thumbnail((300, 250))
    tile = Image.new("RGB", (320, 300), "white")
    tile.paste(image, ((320 - image.width) // 2, 35))
    d = ImageDraw.Draw(tile)
    d.text((6, 5), f'{split} | area={float(row["relative_area"]):.1%}', fill="black", font=font)
    d.text((6, 17), stem[:45], fill="black", font=font)
    tiles.append(tile)

if not tiles:
    raise SystemExit("No matching images found.")

columns = 3
sheet = Image.new("RGB", (columns * 320, math.ceil(len(tiles) / columns) * 300), "white")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % columns) * 320, (i // columns) * 300))

output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output)
print(f"Created contact sheet with {len(tiles)} images: {output}")
