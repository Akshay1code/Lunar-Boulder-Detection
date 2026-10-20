"""Create contact sheets for remaining candidate train-validation prefix groups."""

from pathlib import Path
import csv
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports/isro/automated_audit/batch_review/image_level_manifest.csv"
OUT_DIR = ROOT / "reports/isro/automated_audit/batch_review/contact_sheets"

TARGETS = [
    "Screenshot-2022-02-25-16-20-17_png_jpg",
    "Screenshot-2022-02-25-16-20-20_png_jpg",
    "Screenshot-2022-02-25-16-20-58_png_jpg",
    "images-_10__jpg",
]

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest at {MANIFEST}")

with MANIFEST.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

OUT_DIR.mkdir(parents=True, exist_ok=True)

for prefix in TARGETS:
    group = [r for r in rows if r["source_prefix"] == prefix]
    group.sort(key=lambda r: (r["split"], r["image_path"]))

    if not group:
        print("No images found for prefix:", prefix)
        continue

    cell_w, cell_h = 320, 350
    cols = 3
    sheet = Image.new(
        "RGB",
        (cols * cell_w, ((len(group) + cols - 1) // cols) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(group):
        path = ROOT / row["image_path"]
        if not path.exists():
            print("Missing image:", path)
            continue

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_w - 20, cell_h - 55))

        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x + 10, y + 35))
        draw.text(
            (x + 10, y + 8),
            f'{row["split"]}: {path.name[:34]}',
            fill="black",
        )

    safe_name = prefix.replace("/", "_")
    out_path = OUT_DIR / f"review_{safe_name}.jpg"
    sheet.save(out_path, quality=92)
    print("Saved contact sheet:", out_path)
