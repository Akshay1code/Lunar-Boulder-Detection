"""Compare pixel difference and label rows between two candidate crater images."""

from pathlib import Path
from PIL import Image, ImageChops, ImageStat
import hashlib

ROOT = Path(__file__).resolve().parents[2]
stem_a = "Screenshot-2022-02-25-15-03-38_png_jpg.rf.feaa5a009b016bc875752daff1d47d8c"
stem_b = "Screenshot-2022-02-25-15-03-38_png_jpg.rf.e695043607ae553a9a8a82e1b7ab80fa"


def locate_image(stem):
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = ROOT / "train" / "images" / (stem + ext)
        if p.exists():
            return p
    return None


a, b = locate_image(stem_a), locate_image(stem_b)
if not a or not b:
    raise SystemExit("Could not locate both images.")

print("Image A:", a.name)
print("Image B:", b.name)
print("Same file bytes:", hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest())

with Image.open(a) as im:
    ia = im.convert("RGB")
with Image.open(b) as im:
    ib = im.convert("RGB")

print("Dimensions:", ia.size, ib.size)
if ia.size != ib.size:
    ib = ib.resize(ia.size)

diff = ImageChops.difference(ia, ib)
mae = sum(ImageStat.Stat(diff).mean) / 3
print(f"Mean absolute pixel difference (0-255): {mae:.2f}")

la = ROOT / "train" / "labels" / (stem_a + ".txt")
lb = ROOT / "train" / "labels" / (stem_b + ".txt")

if la.exists() and lb.exists():
    lines_a = sorted(x.strip() for x in la.read_text().splitlines() if x.strip())
    lines_b = sorted(x.strip() for x in lb.read_text().splitlines() if x.strip())
    print("Label rows A:", len(lines_a))
    print("Label rows B:", len(lines_b))
    print("Annotations text identical:", lines_a == lines_b)
else:
    print("Could not find both label files.")
