"""Review visual correspondence for the top two train-test overlap groups.

Generates a 2x2 side-by-side contact sheet of the train and test images for:
- Screenshot-2022-02-25-15-00-39_png_jpg
- Screenshot-2022-02-25-15-01-32_png_jpg
"""

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/isro/automated_audit/batch_review/contact_sheets/final_train_test_review.jpg"

pairs = [
    (
        "15-00-39 TRAIN",
        "train/images/Screenshot-2022-02-25-15-00-39_png_jpg.rf.0314ffcc4d08ad8ddb0d01c7f8f3bab1.jpg",
    ),
    (
        "15-00-39 TEST",
        "test/images/Screenshot-2022-02-25-15-00-39_png_jpg.rf.1249ddb0d7752077a9c388476cba3a2b.jpg",
    ),
    (
        "15-01-32 TRAIN",
        "train/images/Screenshot-2022-02-25-15-01-32_png_jpg.rf.d3b41b4206d0f8376f16e023919ed937.jpg",
    ),
    (
        "15-01-32 TEST",
        "test/images/Screenshot-2022-02-25-15-01-32_png_jpg.rf.98c03b32a48bab1c9b918a8ce28e8f9d.jpg",
    ),
]

cell_w, cell_h = 420, 380
sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), "white")
draw = ImageDraw.Draw(sheet)

for i, (label, rel_path) in enumerate(pairs):
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((cell_w - 20, cell_h - 50))

    x = (i % 2) * cell_w
    y = (i // 2) * cell_h
    sheet.paste(im, (x + 10, y + 35))
    draw.text((x + 10, y + 8), label, fill="black")

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT, quality=95)
print("Saved final train-test review sheet:", OUT)
