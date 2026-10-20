"""Create visual review sheet for Screenshot-2022-02-25-14-12-12_png_jpg."""

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv

ROOT = Path(__file__).resolve().parents[2]
report = ROOT / "reports/isro/automated_audit/cross_split_prefix_groups.csv"
prefix = "Screenshot-2022-02-25-14-12-12_png_jpg"
out = ROOT / "reports/isro/automated_audit/screenshot_141212_review.jpg"

if not report.exists():
    raise FileNotFoundError(f"Missing report at {report}")

with report.open(newline="", encoding="utf-8-sig") as f:
    row = next(r for r in csv.DictReader(f) if r["source_like_prefix"] == prefix)

paths = []
for item in row["filenames"].split("|"):
    rel = Path(item.strip())
    paths.append(ROOT / rel.parent / "images" / rel.name)

w, h, label_h = 480, 480, 65
sheet = Image.new("RGB", (w * 2, (h + label_h) * 2), "white")
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(paths):
    if not path.exists():
        print(f"Missing: {path}")
        continue
    with Image.open(path) as im:
        thumb = ImageOps.contain(im.convert("RGB"), (w, h))
    x, y = (i % 2) * w, (i // 2) * (h + label_h)
    sheet.paste(thumb, (x, y))
    draw.text((x + 5, y + h + 5), str(path.relative_to(ROOT)), fill="black")

out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=92)
print(f"Saved review sheet: {out}")
