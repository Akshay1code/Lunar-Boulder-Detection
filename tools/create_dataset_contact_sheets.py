"""Preview existing dataset references; never train, annotate, or change assets.

Default: all splits, six image/mask/overlay rows per PNG, unique output run.
--check-only verifies inputs without creating any output files or directories.
Requires only the already installed Pillow package (no PyTorch or GPU).
"""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import textwrap
import uuid

sys.dont_write_bytecode = True
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
TILE = 256
WIDTH = 808
ROW_HEIGHT = 365
HEADER = 85


def dataset_root():
    """Read the existing loader's literal path without importing/executing it."""
    tree = ast.parse((ROOT / "dataset_loader.py").read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DATASET_ROOT" for t in node.targets
        ):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "Path" and len(value.args) == 1):
                path = ast.literal_eval(value.args[0])
                if isinstance(path, str):
                    return Path(path).resolve()
    raise ValueError("Cannot read DATASET_ROOT safely; supply --dataset-root explicitly.")


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def inventory(images_dir, masks_dir):
    issues = []
    if not images_dir.is_dir():
        issues.append(f"Image directory missing: {images_dir}")
    if not masks_dir.is_dir():
        issues.append(f"Mask directory missing: {masks_dir}")
    actual = {p.name for p in images_dir.iterdir()
              if p.is_file() and p.suffix.lower() in EXTENSIONS} if images_dir.is_dir() else set()
    coco_path = images_dir / "_annotations.coco.json"
    metadata = {}
    if coco_path.is_file():
        data = json.loads(coco_path.read_text(encoding="utf-8-sig"))
        for row in data["images"]:
            name = row["file_name"]
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe COCO filename in {coco_path}: {name!r}")
            if name in metadata:
                raise ValueError(f"Duplicate COCO filename: {name}")
            metadata[name] = row
        for name in sorted(actual - metadata.keys()):
            issues.append(f"Image not listed in COCO: {images_dir / name}")
    else:
        issues.append(f"COCO metadata missing; using image filenames: {coco_path}")
    names = sorted(actual | metadata.keys())
    stems = {}
    for name in names:
        stems.setdefault(Path(name).stem, []).append(name)
    collisions = {s for s, matches in stems.items() if len(matches) > 1}
    expected_masks = {Path(name).stem + "_mask.png" for name in names}
    if masks_dir.is_dir():
        for p in sorted(masks_dir.glob("*_mask.png")):
            if p.name not in expected_masks:
                issues.append(f"Mask has no matching image/COCO record: {p}")
    if not names:
        issues.append(f"No image records found: {images_dir}")
    return names, metadata, collisions, issues


def load_pair(image_path, mask_path, metadata, collision=False):
    errors = []
    image = mask = None
    try:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        errors.append(f"Image unavailable: {exc}")
    try:
        if collision:
            raise ValueError("Ambiguous mask association: multiple images share this stem")
        with Image.open(mask_path) as src:
            if src.mode != "L":
                raise ValueError(f"Expected 8-bit grayscale mask, found {src.mode}")
            mask = src.copy()
        histogram = mask.histogram()
        if sum(histogram[1:255]):
            raise ValueError("Mask contains values other than 0 and 255")
        if image is not None and mask.size != image.size:
            raise ValueError(f"Image/mask dimensions differ: {image.size} vs {mask.size}")
        if metadata and mask.size != (metadata["width"], metadata["height"]):
            raise ValueError("Mask dimensions disagree with COCO")
    except (OSError, ValueError, KeyError, Image.DecompressionBombError) as exc:
        mask = None
        errors.append(f"Mask unavailable/unusable: {exc}")
    if image is not None and metadata and image.size != (metadata["width"], metadata["height"]):
        errors.append("Image dimensions disagree with COCO; overlay withheld")
        mask = None
    coverage = None if mask is None else mask.histogram()[255] / (mask.width * mask.height)
    return image, mask, coverage, errors


def paste_thumbnail(canvas, source, x, y, nearest=False):
    if source is None:
        ImageDraw.Draw(canvas).text((x + 12, y + 105), "UNAVAILABLE / NOT DISPLAYED",
                                   font=font(13), fill="#a00000")
        return
    thumbnail = source.copy()
    thumbnail.thumbnail((TILE, TILE), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    canvas.paste(thumbnail, (x + (TILE-thumbnail.width)//2, y + (TILE-thumbnail.height)//2))


def draw_row(canvas, row_number, split, name, image, mask, coverage, errors):
    y = HEADER + row_number * ROW_HEIGHT
    draw = ImageDraw.Draw(canvas)
    draw.text((12, y), f"{split} | {name}", font=font(12), fill="black")
    status = ("MASK UNAVAILABLE: no empty-mask substitution" if mask is None else
              "EMPTY REFERENCE: missing labels do NOT confirm absence" if coverage == 0 else
              f"UNVERIFIED rectangular reference | stored foreground: {coverage*100:.3f}%")
    draw.text((12, y + 20), status, font=font(14), fill="#8b3400")
    overlay = None
    if image is not None and mask is not None:
        tinted = Image.blend(image, Image.new("RGB", image.size, (0, 255, 255)), .35)
        overlay = Image.composite(tinted, image, mask)
    for x, title, pixels, nearest in [(12,"Original image",image,False),
            (276,"Existing mask - UNVERIFIED",mask,True), (540,"Cyan reference overlay (35%)",overlay,False)]:
        draw.text((x,y+43),title,font=font(13),fill="black")
        paste_thumbnail(canvas,pixels,x,y+65,nearest)
    message = " | ".join(errors)
    if message:
        lines = textwrap.wrap(message, width=112)
        draw.text((12,y+325),"\n".join(lines[:2]),font=font(11),fill="#a00000")
    draw.line((12,y+ROW_HEIGHT-3,WIDTH-12,y+ROW_HEIGHT-3),fill="#cccccc")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Defaults to DATASET_ROOT in dataset_loader.py")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--rows-per-sheet", type=int, default=6)
    parser.add_argument("--check-only", action="store_true", help="Read-only verification; no previews or reports written")
    args = parser.parse_args(argv)
    if not 1 <= args.rows_per_sheet <= 12:
        parser.error("--rows-per-sheet must be between 1 and 12")
    if len(args.splits) != len(set(args.splits)):
        parser.error("Each split may be requested only once")
    run = None
    try:
        images_root = args.dataset_root.resolve() if args.dataset_root else dataset_root()
        masks_root = ROOT / "dataset_masks"
        print(f"Images: {images_root}\nMasks: {masks_root}")
        report = {"description":"Preview of unverified references; no geological validation",
                  "dataset_root":str(images_root), "mask_root":str(masks_root),
                  "splits":{}, "issues":[]}
        if not args.check_only:
            output = ROOT / "contact_sheets"
            if output.resolve() != output:
                raise ValueError("Redirected output directory refused")
            output.mkdir(exist_ok=True)
            run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex)
            run.mkdir(exist_ok=False)
        for split in args.splits:
            image_dir, mask_dir = images_root / split, masks_root / split
            names, metadata, collisions, issues = inventory(image_dir, mask_dir)
            report["issues"].extend(issues)
            for issue in issues:
                print(f"WARNING: {issue}", file=sys.stderr)
            summary = {"records":len(names),"usable_pairs":0,"empty_reference_masks":0,"sheets":[],"errors":[]}
            report["splits"][split] = summary
            if run:
                (run / split).mkdir(exist_ok=False)
            for start in range(0,len(names),args.rows_per_sheet):
                batch = names[start:start+args.rows_per_sheet]
                canvas = None
                if run:
                    canvas = Image.new("RGB",(WIDTH,HEADER+ROW_HEIGHT*len(batch)),"white")
                    draw = ImageDraw.Draw(canvas)
                    draw.text((12,10),f"{split.upper()} | Reference review | Images {start+1}-{start+len(batch)} / {len(names)}",font=font(20),fill="black")
                    draw.text((12,40),"Unverified masks. Empty labels are not confirmed negatives.",font=font(16),fill="#8b3400")
                    draw.text((12,62),"No predictions or geological classifications. No source assets are altered.",font=font(13),fill="black")
                for j,name in enumerate(batch):
                    image_path = image_dir/name
                    mask_path = mask_dir/(Path(name).stem+"_mask.png")
                    image,mask,coverage,errors = load_pair(image_path,mask_path,metadata.get(name),Path(name).stem in collisions)
                    if image is not None and mask is not None and not errors:
                        summary["usable_pairs"] += 1
                    if coverage == 0:
                        summary["empty_reference_masks"] += 1
                    if errors:
                        summary["errors"].append({"image_path":str(image_path),"mask_path":str(mask_path),"errors":errors})
                        print(f"WARNING: {split}/{name}: {' | '.join(errors)}",file=sys.stderr)
                    if canvas is not None:
                        draw_row(canvas,j,split,name,image,mask,coverage,errors)
                if canvas is not None:
                    relative = f"{split}/{split}_{start//args.rows_per_sheet+1:03d}.png"
                    with (run/relative).open("xb") as stream:
                        canvas.save(stream,format="PNG")
                    summary["sheets"].append(relative)
                    canvas.close()
            print(f"{split}: {summary['usable_pairs']}/{len(names)} usable pairs; {summary['empty_reference_masks']} empty references; {len(summary['sheets'])} sheets")
        problems = bool(report["issues"] or any(x["errors"] for x in report["splits"].values()))
        report["status"] = "completed_with_input_issues" if problems else "completed"
        if run:
            with (run/"report.json").open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
            print(f"Previews and input-check report: {run}")
        else:
            print("Read-only check complete. No files written.")
        return 1 if problems else 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"STOP: {exc}",file=sys.stderr)
        if run:
            print(f"Incomplete previews remain at {run}; existing files were not overwritten.",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
