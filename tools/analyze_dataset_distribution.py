"""Original-resolution train/valid descriptive analysis; source files read-only.

Writes CSV/TXT only in a unique report directory. --check-only writes nothing.
No model loading, training, test access, resizing, or annotation changes.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_segmentation_dataset import ROOT, configured_paths, EXTENSIONS
import numpy as np
from PIL import Image

SPLITS = ('train', 'valid')
BINS = tuple(f'intensity_{lo:03d}_{lo+15:03d}_percent' for lo in range(0, 256, 16))
STATS = ('brightness_mean', 'contrast_std', 'contrast_p95_minus_p05',
         'intensity_min', 'intensity_p05', 'intensity_p25', 'intensity_median',
         'intensity_p75', 'intensity_p95', 'intensity_max', 'foreground_percent')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(before):
    changed = [str(p) for p, expected in before.items() if not p.is_file() or digest(p) != expected]
    if changed:
        raise RuntimeError(f'Source hash discrepancy; no repair attempted: {changed}')


def scan(images_root, masks_root, before):
    rows, problems = [], []
    for split in SPLITS:
        image_dir, mask_dir = images_root/split, masks_root/split
        if not image_dir.is_dir() or not mask_dir.is_dir():
            problems.append(f'{split}: missing image or mask directory: {image_dir}; {mask_dir}')
            continue
        coco = image_dir/'_annotations.coco.json'
        before[coco] = digest(coco)
        items = json.loads(coco.read_text(encoding='utf-8-sig'))['images']
        metadata = {}
        for item in items:
            name = item['file_name']
            if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError(f'Unsafe COCO filename: {name!r}')
            if name in metadata:
                raise ValueError(f'Duplicate COCO filename in {split}: {name}')
            metadata[name] = item
        actual = {p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
        names = sorted(actual | metadata.keys())
        if not names:
            problems.append(f'{split}: no image records')
        stems = Counter(Path(n).stem.casefold() for n in names)
        expected_masks = {Path(n).stem+'_mask.png' for n in names}
        for path in mask_dir.iterdir():
            if path.is_file() and path.suffix.lower() in EXTENSIONS and path.name not in expected_masks:
                problems.append(f'Unmatched mask: {path}')
        for name in names:
            image_path, mask_path = image_dir/name, mask_dir/(Path(name).stem+'_mask.png')
            row = dict(split=split, filename=name, image_path=str(image_path), mask_path=str(mask_path),
                       status='invalid', issues='')
            try:
                if stems[Path(name).stem.casefold()] != 1:
                    raise ValueError('Ambiguous image stem/mask association')
                if name not in metadata:
                    raise ValueError('Image not listed in COCO')
                for path in (image_path, mask_path):
                    before[path] = digest(path)
                with Image.open(image_path) as source:
                    source.load()
                    if source.mode not in ('L', 'RGB'):
                        raise ValueError(f'Unsupported original image mode: {source.mode}')
                    row.update(image_mode=source.mode, width=source.width, height=source.height)
                    # Pillow RGB->L: rounded 8-bit 0.299R + 0.587G + 0.114B.
                    gray = np.asarray(source.convert('L'))
                with Image.open(mask_path) as mask:
                    mask.load()
                    array = np.asarray(mask)
                    if mask.mode != 'L' or array.dtype != np.uint8 or not set(np.unique(array)) <= {0, 255}:
                        raise ValueError('Mask must be 8-bit grayscale with values 0/255')
                    if array.shape != gray.shape:
                        raise ValueError('Original image/mask dimensions differ')
                    foreground = float(np.count_nonzero(array)/array.size*100)
                item = metadata[name]
                if (row['width'], row['height']) != (item['width'], item['height']):
                    raise ValueError('Original dimensions disagree with COCO')
                q = np.percentile(gray, [0, 5, 25, 50, 75, 95, 100])
                histogram = np.histogram(gray, bins=np.arange(0, 257, 16))[0]/gray.size*100
                row.update(coco_image_id=item['id'], status='valid',
                           reference_group='nonempty_unverified' if foreground else 'empty_unverified',
                           foreground_percent=foreground, brightness_mean=float(gray.mean()),
                           contrast_std=float(gray.std()), contrast_p95_minus_p05=float(q[5]-q[1]),
                           image_sha256=before[image_path], mask_sha256=before[mask_path])
                row.update(zip(('intensity_min', 'intensity_p05', 'intensity_p25', 'intensity_median',
                                'intensity_p75', 'intensity_p95', 'intensity_max'), map(float, q)))
                row.update(zip(BINS, map(float, histogram)))
            except (OSError, ValueError, KeyError) as exc:
                row['issues'] = str(exc)
                problems.append(f'{split}/{name}: {exc}')
            rows.append(row)
        print(f'{split}: inspected {len(names)} image records', flush=True)
    return rows, problems


def summary_text(rows, problems, images_root, masks_root):
    lines = ['TRAIN/VALID DESCRIPTIVE DATASET DISTRIBUTION', f'Images: {images_root}', f'Masks: {masks_root}',
             'Original exported resolution; no resizing or normalization. Train and valid only.',
             'Grayscale: Pillow L conversion, rounded 8-bit 0.299R+0.587G+0.114B for RGB.',
             'Brightness = mean intensity (0-255); contrast = population intensity standard deviation.',
             'Robust contrast = 95th minus 5th intensity percentile. Percentiles use NumPy linear interpolation.',
             'Histograms: 16 bins [0..15], [16..31], ..., [240..255], percent of each image.',
             'Foreground = nonzero reference pixels / original mask pixels * 100.',
             'All group summaries weight images equally; histogram summaries average per-image percentages.',
             'References are scientifically unverified rectangular annotations. Empty masks are not confirmed negatives.',
             'These describe the available exported images, not necessarily original sensor data.',
             'Differences may reflect illumination, terrain, cropping, or export processing. Descriptive statistics',
             'alone do not prove dataset shift, explain model performance, or validate annotation correctness.',
             'Source-scene independence and representative sampling remain unverified; no significance test is made.',
             'No model/checkpoint is loaded and no test split is accessed.', '']
    if problems:
        lines += ['FAILED: missing/invalid inventory or pairs. Distribution comparisons withheld to avoid silent exclusions.', *problems]
        return '\n'.join(lines)+'\n'
    for split in SPLITS:
        subset = [r for r in rows if r['split'] == split]
        counts = Counter(r['reference_group'] for r in subset)
        lines += [f'{split}: {len(subset)} pairs; empty={counts["empty_unverified"]}; nonempty={counts["nonempty_unverified"]}']
        for group in ('all', 'nonempty_unverified', 'empty_unverified'):
            selected = [r for r in subset if group == 'all' or r['reference_group'] == group]
            lines.append(f'{split}/{group}: n={len(selected)}')
            if not selected:
                lines.append('Statistics unavailable (empty group).')
                continue
            dims = Counter((r['width'], r['height']) for r in selected)
            lines.append(f'Original width x height counts: {dict(dims)}')
            for field in STATS:
                values = [r[field] for r in selected]
                q = np.percentile(values, [0, 25, 50, 75, 100])
                lines.append(f'{field}: mean={np.mean(values):.6f}; min/Q1/median/Q3/max='+' / '.join(f'{v:.6f}' for v in q))
            lines.append('Mean normalized intensity histogram % (bins in ascending order): '+
                         ', '.join(f'{np.mean([r[b] for r in selected]):.4f}' for b in BINS))
            lines.append('')
    lines.append('VALID MINUS TRAIN: differences in equal-image-weight group means (descriptive only)')
    for group in ('all', 'nonempty_unverified', 'empty_unverified'):
        subsets = [[r for r in rows if r['split'] == split and (group == 'all' or r['reference_group'] == group)] for split in SPLITS]
        if all(subsets):
            for field in ('brightness_mean', 'contrast_std', 'contrast_p95_minus_p05', 'foreground_percent'):
                delta = np.mean([r[field] for r in subsets[1]])-np.mean([r[field] for r in subsets[0]])
                lines.append(f'{group}/{field}: {delta:+.6f}')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        for path in (ROOT/'dataset_loader.py', ROOT/'tools/audit_segmentation_dataset.py'):
            before[path] = digest(path)
        images_root, masks_root = configured_paths()  # Parse loader literals; no PyTorch import.
        rows, problems = scan(images_root, masks_root, before)
        summary = summary_text(rows, problems, images_root, masks_root)
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ['split', 'filename', 'issues']
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        if len(list(csv.DictReader(io.StringIO(buffer.getvalue())))) != len(rows):
            raise ValueError('CSV serialization failed')
        verify(before)
        if not args.check_only:
            parent = ROOT/'dataset_distribution_reports'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output/'image_statistics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                stream.write(buffer.getvalue())
            with (output/'summary.txt').open('x', encoding='utf-8') as stream:
                stream.write(summary)
        print(summary)
        print(f'Reports saved: {output}' if output else 'Check only: no reports written.')
        return 1 if problems else 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Partial reports may remain in {output}; no repair attempted.', file=sys.stderr)
        return 1
    finally:
        verify(before)


if __name__ == '__main__':
    raise SystemExit(main())
