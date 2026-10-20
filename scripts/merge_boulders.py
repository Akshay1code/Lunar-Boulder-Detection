"""
Merge the Roboflow boulder dataset into the Lunar-CV crater/rille dataset.

What it does
  1. Pools every image in the boulder export (train/valid/test), converts polygon
     labels to boxes, remaps class 0 ("Boulder") -> class 2.
  2. Re-splits the boulder images into train / valid / test (seeded) because the
     export came with everything in train.
  3. Writes them to  <root>/boulders/{train,valid,test}/{images,labels}
     (train images are repeated --repeat times to offset the small count).
  4. Writes YAML files next to data.yaml:
       data_merged.yaml          crater+rille+boulder, train/val/test (use for training)
       data_eval_craters_rilles.yaml  original test set only (compare against v2)
       data_eval_boulders.yaml        boulder test images only
       data_boulder_only.yaml         1-class boulder model (separate-model option)
Your original train/valid/test folders are NOT modified.

Usage (from project root):
    python scripts/merge_boulders.py
    python scripts/merge_boulders.py --src boulder_raw --repeat 3 --seed 0
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAMES = ["crater", "rille", "boulder"]


def parse_label(path: Path, new_id: int):
    """Return list of 'cls cx cy w h' strings; polygons are converted to boxes."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) == 5:
            cx, cy, w, h = map(float, t[1:])
        elif len(t) >= 7 and (len(t) - 1) % 2 == 0:  # polygon: cls x1 y1 x2 y2 ...
            v = list(map(float, t[1:]))
            xs, ys = v[0::2], v[1::2]
            x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
            y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        else:
            continue
        if w <= 0 or h <= 0:
            continue
        out.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def write_yaml(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"  wrote {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "boulder_raw"), help="unzipped Roboflow boulder dataset")
    ap.add_argument("--out", default=str(ROOT / "boulders"))
    ap.add_argument("--root", default=str(ROOT), help="project root holding train/valid/test + data.yaml")
    ap.add_argument("--boulder-id", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--repeat", type=int, default=3, help="copies of each boulder TRAIN image (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out, root = Path(a.src), Path(a.out), Path(a.root)
    if not src.exists():
        raise SystemExit(f"Boulder dataset not found: {src}")

    # 1. collect
    items = []
    for split in ("train", "valid", "val", "test"):
        d = src / split / "images"
        if d.is_dir():
            for im in sorted(d.iterdir()):
                if im.suffix.lower() in IMG_EXTS:
                    lab = src / split / "labels" / (im.stem + ".txt")
                    items.append((im, parse_label(lab, a.boulder_id)))
    if not items:
        raise SystemExit("No images found.")
    with_boxes = [x for x in items if x[1]]
    empty = [x for x in items if not x[1]]
    print(f"Found {len(items)} images: {len(with_boxes)} with boulders, {len(empty)} with no labels")
    print(f"Total boulder boxes: {sum(len(l) for _, l in items)}")

    # 2. split (only images that contain boulders go to val/test)
    rng = random.Random(a.seed)
    rng.shuffle(with_boxes)
    n_val = max(1, round(len(with_boxes) * a.val_frac))
    n_test = max(1, round(len(with_boxes) * a.test_frac))
    split_items = {
        "valid": with_boxes[:n_val],
        "test": with_boxes[n_val:n_val + n_test],
        "train": with_boxes[n_val + n_test:] + empty,
    }

    # 3. write
    if out.exists():
        shutil.rmtree(out)
    for split, lst in split_items.items():
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        copies = a.repeat if split == "train" else 1
        for im, lines in lst:
            for k in range(copies):
                stem = im.stem if k == 0 else f"{im.stem}_rep{k}"
                shutil.copy2(im, out / split / "images" / (stem + im.suffix))
                (out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        nb = sum(len(l) for _, l in lst)
        print(f"  {split:<6} {len(lst):>3} images ({len(lst) * copies} files), {nb} boxes")

    # 4. yamls
    o = out.relative_to(root).as_posix() if out.is_relative_to(root) else out.as_posix()
    names = "names: " + str(NAMES) + "\nnc: 3"
    target_dir = root / "configs" if (root / "configs").is_dir() else root
    write_yaml(target_dir / "data_merged.yaml", f"""
path: .
train: [train/images, {o}/train/images]
val: [valid/images, {o}/valid/images]
test: [test/images, {o}/test/images]
{names}
""")
    write_yaml(target_dir / "data_eval_craters_rilles.yaml", f"""
path: .
train: train/images
val: test/images
test: test/images
{names}
""")
    write_yaml(target_dir / "data_eval_boulders.yaml", f"""
path: .
train: {o}/train/images
val: {o}/test/images
test: {o}/test/images
{names}
""")
    write_yaml(target_dir / "data_boulder_only.yaml", f"""
path: .
train: {o}/train/images
val: {o}/valid/images
test: {o}/test/images
nc: 1
names: ['boulder']
""")
    print("\nDone. Train with:  python scripts/train.py --data configs/data_merged.yaml --name v3")


if __name__ == "__main__":
    main()