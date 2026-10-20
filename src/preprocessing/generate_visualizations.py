"""Stage 6: Visual Review and Contact-Sheet Generation.

Renders YOLO bounding boxes on sample images with class-colored overlays and
generates summary contact sheets for qualitative review.
"""

from pathlib import Path
import argparse
import math
import random
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

COLORS = [
    (230, 60, 60),   # Class 0: crater (red)
    (30, 150, 80),   # Class 1: rille (green)
    (50, 110, 220),
    (200, 130, 20),
]


def load_class_names(dataset_root: Path) -> list:
    """Read class names from data.yaml."""
    yaml_path = dataset_root / "data.yaml"
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    return names


def draw_annotations(image_path: Path, label_path: Path, class_names: list, output_path: Path):
    """Draw bounding boxes and class tags on an image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    font = ImageFont.load_default()
    boxes_drawn = 0

    if label_path.exists():
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if not 0 <= class_id < len(class_names):
                continue
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                continue
            if bw <= 0 or bh <= 0:
                continue

            x1 = max(0, (cx - bw / 2) * image_width)
            y1 = max(0, (cy - bh / 2) * image_height)
            x2 = min(image_width, (cx + bw / 2) * image_width)
            y2 = min(image_height, (cy + bh / 2) * image_height)

            color = COLORS[class_id % len(COLORS)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, min(image_width, image_height) // 320))

            label = f"{class_names[class_id]} (ID {class_id})"
            text_box = draw.textbbox((0, 0), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]

            tx = max(0, int(x1))
            ty = max(0, int(y1) - text_h - 6)
            draw.rectangle((tx, ty, tx + text_w + 6, ty + text_h + 4), fill=color)
            draw.text((tx + 3, ty + 2), label, fill="white", font=font)
            boxes_drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path, boxes_drawn


def create_contact_sheet(entries: list, output_path: Path, columns: int = 3):
    """Combine thumbnail entries into a multi-column contact sheet."""
    if not entries:
        return

    tile_w, tile_h = 440, 350
    rows = math.ceil(len(entries) / columns)

    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (split, stem, img_path) in enumerate(entries):
        col = idx % columns
        row = idx // columns
        x = col * tile_w
        y = row * tile_h

        draw.rectangle((x + 2, y + 2, x + tile_w - 3, y + tile_h - 3), outline=(190, 190, 190))
        title = f"{split}: {stem[:42]}"
        draw.text((x + 10, y + 10), title, fill="black", font=font)

        with Image.open(img_path) as source:
            thumb = source.convert("RGB")
        thumb.thumbnail((tile_w - 20, tile_h - 45), Image.Resampling.LANCZOS)

        img_x = x + (tile_w - thumb.width) // 2
        img_y = y + 35
        sheet.paste(thumb, (img_x, img_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def generate_visualizations(root_dir: Path, output_dir: Path, samples_per_split: int = 6):
    """Generate annotation visualizations and split contact sheets."""
    root = Path(root_dir).resolve()
    out = Path(output_dir).resolve()

    class_names = load_class_names(root)
    rng = random.Random(42)
    all_samples = []

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"Skipping {split}: missing images or labels folder.")
            continue

        images = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels_by_stem = {p.stem: p for p in labels_dir.rglob("*.txt") if p.is_file()}
        candidates = [p for p in images if p.stem in labels_by_stem]

        sample_count = min(samples_per_split, len(candidates))
        chosen = rng.sample(candidates, sample_count)

        split_samples = []
        sample_dir = out / "samples" / split

        for image_path in chosen:
            label_path = labels_by_stem[image_path.stem]
            out_sample_path = sample_dir / f"{image_path.stem}.jpg"
            saved_path, box_count = draw_annotations(image_path, label_path, class_names, out_sample_path)

            entry = (split, image_path.stem, saved_path)
            split_samples.append(entry)
            all_samples.append(entry)
            print(f"  {split}: {image_path.name} -> {box_count} boxes rendered")

        create_contact_sheet(split_samples, out / f"{split}_contact_sheet.jpg")

    create_contact_sheet(all_samples, out / "all_splits_contact_sheet.jpg")

    print("\n=== VISUALIZATION GENERATION COMPLETE ===")
    print(f"Total samples rendered: {len(all_samples)}")
    print(f"Contact sheets written to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Render annotation visualizations and contact sheets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root folder containing train/valid/test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "isro" / "visualizations",
        help="Output folder for visualizations and contact sheets",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=6,
        help="Number of random images to visualize per split",
    )
    args = parser.parse_args()

    generate_visualizations(args.root, args.output, args.samples_per_split)


if __name__ == "__main__":
    main()
