"""Read-only source audit; writes only CSV/TXT in a new dataset_audits run.

No training, resizing, mask generation or geological classification.
--check-only prints the summary without creating reports or directories.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
FIELDS = ["split", "record_type", "filename", "image_path", "mask_path",
          "coco_image_ids", "image_width", "image_height", "mask_width", "mask_height",
          "dimensions_match", "mask_mode", "mask_dtype", "unique_values",
          "unexpected_values", "mask_state", "foreground_percent", "issues"]


def configured_paths():
    # Parse literals only: importing the loader would unnecessarily load PyTorch.
    tree = ast.parse((ROOT / "dataset_loader.py").read_text(encoding="utf-8-sig"))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"DATASET_ROOT", "MASK_ROOT"}:
                value = node.value
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == "Path" and len(value.args) == 1):
                    path = Path(ast.literal_eval(value.args[0]))
                    result[target.id] = (path if path.is_absolute() else ROOT / path).resolve()
    if set(result) != {"DATASET_ROOT", "MASK_ROOT"}:
        raise ValueError("Cannot safely read literal dataset paths from dataset_loader.py")
    return result["DATASET_ROOT"], result["MASK_ROOT"]


def decode(path):
    with Image.open(path) as src:
        src.load()
        return src.copy()


def inspect_pair(split, name, image_path, mask_path, coco_rows, issues, record_type="image"):
    row = dict.fromkeys(FIELDS, "")
    row.update(split=split, record_type=record_type, filename=name,
               image_path=str(image_path) if image_path else "", mask_path=str(mask_path),
               coco_image_ids=json.dumps([x["id"] for x in coco_rows]), mask_state="unavailable")
    problems = list(issues)
    image_size = None
    if image_path:
        try:
            image = decode(image_path)
            image_size = image.size
            row.update(image_width=image.width, image_height=image.height)
            image.close()
            if any(image_size != (x["width"], x["height"]) for x in coco_rows):
                problems.append("image_dimensions_disagree_with_COCO")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            problems.append(f"missing_or_unreadable_image: {exc}")
    try:
        mask = decode(mask_path)
        row.update(mask_width=mask.width, mask_height=mask.height, mask_mode=mask.mode)
        arr = np.asarray(mask)
        values = np.unique(arr).tolist()
        row.update(mask_dtype=str(arr.dtype), unique_values=json.dumps(values))
        unexpected = [v for v in values if v not in (0, 255)]
        row["unexpected_values"] = json.dumps(unexpected)
        if unexpected:
            problems.append("unexpected_mask_values_expected_0_255")
        if image_size is not None:
            row["dimensions_match"] = image_size == mask.size
            if image_size != mask.size:
                problems.append("image_mask_dimension_mismatch")
        if any(mask.size != (x["width"], x["height"]) for x in coco_rows):
            problems.append("mask_dimensions_disagree_with_COCO")
        if mask.mode != "L" or arr.ndim != 2 or arr.dtype != np.uint8:
            problems.append("unexpected_mask_format_expected_8bit_grayscale")
            row["mask_state"] = "uninterpreted_format"
        else:
            coverage = float(np.count_nonzero(arr > 0) / arr.size * 100)
            row["foreground_percent"] = coverage
            row["mask_state"] = "empty_unverified" if coverage == 0 else "nonempty_unverified"
        mask.close()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        problems.append(f"missing_or_unreadable_mask: {exc}")
    row["issues"] = " | ".join(problems)
    return row


def scan(images_root, masks_root):
    rows, notes = [], []
    for split in SPLITS:
        image_dir, mask_dir = images_root / split, masks_root / split
        images = {p.name: p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSIONS} if image_dir.is_dir() else {}
        masks = {p.name: p for p in mask_dir.iterdir()
                 if p.is_file() and p.suffix.lower() == ".png"} if mask_dir.is_dir() else {}
        if not image_dir.is_dir():
            notes.append(f"{split}: image directory missing: {image_dir}")
        if not mask_dir.is_dir():
            notes.append(f"{split}: mask directory missing: {mask_dir}")
        metadata = defaultdict(list)
        try:
            data = json.loads((image_dir / "_annotations.coco.json").read_text(encoding="utf-8-sig"))
            for item in data["images"]:
                name = item["file_name"]
                if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError(f"Unsafe COCO filename: {name!r}")
                metadata[name].append(item)
        except (OSError, KeyError, ValueError) as exc:
            notes.append(f"{split}: COCO metadata issue: {exc}; inspect CSV inventory carefully")
        names = sorted(images.keys() | metadata.keys())
        stems, filenames = Counter(Path(n).stem.casefold() for n in names), Counter(n.casefold() for n in names)
        used_masks = set()
        for name in names:
            mask_name = Path(name).stem + "_mask.png"
            used_masks.add(mask_name)
            issues = []
            if len(metadata[name]) > 1:
                issues.append("duplicate_COCO_filename")
            if filenames[name.casefold()] > 1:
                issues.append("case_insensitive_filename_collision")
            if stems[Path(name).stem.casefold()] > 1:
                issues.append("ambiguous_mask_stem_collision")
            if name not in metadata or not metadata[name]:
                issues.append("image_not_listed_in_COCO")
            rows.append(inspect_pair(split,name,image_dir/name,mask_dir/mask_name,metadata[name],issues))
        for name in sorted(masks.keys() - used_masks):
            rows.append(inspect_pair(split,name,None,masks[name],[],["unmatched_mask"],"unmatched_mask"))
        if not names:
            notes.append(f"{split}: no image records found")
    occurrences = defaultdict(list)
    for row in rows:
        if row["record_type"] == "image":
            occurrences[row["filename"].casefold()].append(row)
    for matches in occurrences.values():
        if len({r["split"] for r in matches}) > 1:
            for row in matches:
                row["issues"] = " | ".join(filter(None,[row["issues"],"filename_repeated_across_splits_not_proof_of_duplicate_content"]))
    return rows, notes


def summary_text(rows, notes, images_root, masks_root):
    lines = ["SEGMENTATION DATASET AUDIT", f"Images: {images_root}", f"Masks: {masks_root}",
             "Source data are read-only. No resizing, training, relabeling or repairs performed.",
             "Masks are unverified rectangular references, not scientifically verified ground truth.",
             "Empty masks are neither accepted nor rejected: missing labels do not establish absence.",
             "Foreground percentage = pixels > 0 / original mask pixel count * 100.",
             "Unexpected-format masks have no foreground percentage; missing masks are not empty masks.",
             "Filename repetition is not a content-duplicate test; image hashes are not compared.", ""]
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split and r["record_type"] == "image"]
        states = Counter(r["mask_state"] for r in subset)
        lines += [split.upper(), f"Image records: {len(subset)}",
                  f"Empty reference masks: {states['empty_unverified']}; nonempty: {states['nonempty_unverified']}",
                  f"Unavailable masks: {states['unavailable']}; uninterpreted formats: {states['uninterpreted_format']}",
                  f"Image/mask dimension mismatches: {sum(r['dimensions_match'] is False for r in subset)}",
                  f"Records with issues: {sum(bool(r['issues']) for r in subset)}",
                  f"Unmatched mask files: {sum(r['split']==split and r['record_type']=='unmatched_mask' for r in rows)}"]
        observed = sorted({v for r in subset if r["unique_values"] for v in json.loads(r["unique_values"])})
        lines.append(f"Observed mask values (raw): {observed}")
        # Exclude anomalous/missing/ambiguous pairs from coverage statistics.
        usable = [r for r in subset if not r["issues"] and r["foreground_percent"] != ""]
        for label, group in [("all usable pairs",usable), ("usable nonempty masks",[r for r in usable if r['foreground_percent'] > 0])]:
            values = [r["foreground_percent"] for r in group]
            if values:
                q = np.percentile(values,[0,25,50,75,100])
                lines.append(f"Coverage % ({label}, n={len(values)}): mean={np.mean(values):.6f}; min/Q1/median/Q3/max="+" / ".join(f"{v:.6f}" for v in q))
            else:
                lines.append(f"Coverage ({label}): unavailable")
        lines.append("Coverage summaries exclude records with issues; raw per-mask measurements remain in CSV.")
        lines.append("")
    lines.append("INVENTORY NOTES")
    lines.extend(notes or ["None"])
    lines.append("FILE ISSUES")
    lines.extend([f"{r['split']}/{r['filename']}: {r['issues']}" for r in rows if r['issues']] or ["None"])
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true",help="Print summary only; do not create reports")
    args = parser.parse_args(argv)
    try:
        images_root, masks_root = configured_paths()
        rows, notes = scan(images_root,masks_root)
        summary = summary_text(rows,notes,images_root,masks_root)
        print(summary)
        if not args.check_only:
            output = ROOT / "dataset_audits"
            if output.resolve() != output:
                raise ValueError("Redirected audit output directory refused")
            output.mkdir(exist_ok=True)
            run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex)
            run.mkdir(exist_ok=False)
            with (run/"image_mask_audit.csv").open("x",encoding="utf-8-sig",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with (run/"summary.txt").open("x",encoding="utf-8") as stream:
                stream.write(summary)
            print(f"New audit reports: {run}")
        else:
            print("Check only: no files written.")
        return 1 if notes or any(r["issues"] for r in rows) else 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"Audit stopped: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
