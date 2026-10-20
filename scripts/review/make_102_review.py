"""Create visual review sheet for the 102_png_jpg prefix group."""

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv

ROOT = Path(__file__).resolve().parents[2]
report = ROOT / "reports/isro/automated_audit/cross_split_prefix_groups.csv"
out = ROOT / "reports/isro/automated_audit/102_png_jpg_review.jpg"

if not report.exists():
    raise FileNotFoundError(f"Missing report at {report}")

with report.open(newline="", encoding="utf-8-sig") as f:
    rows = csv.DictReader(f)
    row = next(r for r in rows if r["source_like_prefix"] == "102_png_jpg")

paths = []
for item in row["filenames"].split("|"):
    rel = Path(item.strip())
    paths.append(ROOT / rel.parent / "images" / rel.name)

thumb_w, thumb_h, label_h = 480, 480, 65
sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 2), "white")
draw = ImageDraw.Draw(sheet)

for i, img_path in enumerate(paths):
    if not img_path.exists():
        print(f"Missing: {img_path}")
        continue
    with Image.open(img_path) as im:
        thumb = ImageOps.contain(im.convert("RGB"), (thumb_w, thumb_h))

    x = (i % 2) * thumb_w
    y = (i // 2) * (thumb_h + label_h)
    sheet.paste(thumb, (x, y))
    draw.text((x + 5, y + thumb_h + 5), str(img_path.relative_to(ROOT)), fill="black")

out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=92)
print(f"Saved review sheet: {out}")
