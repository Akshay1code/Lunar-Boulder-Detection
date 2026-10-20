"""Review empty UNVERIFIED validation references using saved shape-analysis ranks.

Three highest, three median-nearest, and three lowest coverage examples;
filename breaks ties, no duplicates. Low group includes zero predictions.
--check-only verifies inference/rendering without writing outputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_experiment_e_validation import (
    ROOT, CHECKPOINT, validation_dataset, overlap, verify_hashes, digest, load_model,
)
from preview_validation_errors import render
import cv2
import numpy as np
import torch
from PIL import Image

SOURCE = ROOT/'prediction_shape_analysis/20261015T040854Z_9fa9bb7e11d24abe9d2ec51f12089f65/per_image_shapes.csv'
NOTICE = 'Empty UNVERIFIED references are not confirmed negatives; predictions are not automatically false positives.'


def select(records):
    ordered = sorted(records, key=lambda r: (float(r['prediction_foreground_percent']), r['filename']))
    if not ordered:
        return []
    median = float(np.median([float(r['prediction_foreground_percent']) for r in ordered]))
    candidates = [
        ('high', sorted(ordered, key=lambda r: (-float(r['prediction_foreground_percent']), r['filename']))),
        ('moderate', sorted(ordered, key=lambda r: (abs(float(r['prediction_foreground_percent'])-median), r['filename']))),
        ('low', ordered),
    ]
    used, result = set(), []
    for group, entries in candidates:
        chosen = [r for r in entries if r['filename'] not in used][:3]
        if len(chosen) < 3:
            print(f'{group}: only {len(chosen)} distinct examples available')
        for row in chosen:
            used.add(row['filename'])
            result.append((group, row))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        for path in [SOURCE, CHECKPOINT]+[ROOT/name for name in (
                'unet_model.py', 'dataset_loader.py', 'tools/preview_unet_predictions.py',
                'tools/preview_validation_errors.py', 'tools/evaluate_experiment_e_validation.py')]:
            before[path] = digest(path)
        with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        dataset = validation_dataset()
        lookup = {a.name: (i, a, b) for i, (a, b) in enumerate(dataset.pairs)}
        if len(records) != 80 or len({r['filename'] for r in records}) != 80 or {r['filename'] for r in records} != set(lookup):
            raise ValueError('CSV does not match the 80 unique validation records')
        for row in records:
            _, image_path, mask_path = lookup[row['filename']]
            if row['split'] != 'valid' or row['measurement_resolution'] != '256x256' or float(row['threshold']) != .5:
                raise ValueError('CSV split/resolution/threshold mismatch')
            if Path(row['image_path']) != image_path or Path(row['mask_path']) != mask_path:
                raise ValueError(f"CSV pair association mismatch: {row['filename']}")
            if Path(row['checkpoint_path']) != CHECKPOINT or row['checkpoint_sha256'] != before[CHECKPOINT]:
                raise ValueError('CSV checkpoint path/hash mismatch')
            coverage = float(row['prediction_foreground_percent'])
            if not np.isfinite(coverage) or not 0 <= coverage <= 100:
                raise ValueError('Invalid saved prediction coverage')
        empty = [r for r in records if int(r['reference_foreground_pixels']) == 0]
        if len(empty) != 61 or any(r['reference_status'] != 'empty_UNVERIFIED' for r in empty):
            raise ValueError('Expected 61 empty UNVERIFIED reference records')
        # Verify all ranking candidates still match their saved image/mask hashes.
        for row in empty:
            _, image_path, mask_path = lookup[row['filename']]
            for path, key in ((image_path, 'image_sha256'), (mask_path, 'mask_sha256')):
                before[path] = digest(path)
                if before[path] != row[key]:
                    raise ValueError(f'Saved asset hash mismatch: {path}')
        chosen = select(empty)
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        manifest, pngs = [], []
        for number, (group, saved) in enumerate(chosen, 1):
            index, image_path, mask_path = lookup[saved['filename']]
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if original is None or raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or raw.any():
                raise ValueError(f'Expected readable image and empty grayscale reference: {image_path}')
            tensor, target = dataset[index]
            if bool(target.any()):
                raise ValueError('Preprocessed reference is not empty')
            with torch.no_grad():
                logits = model(tensor.unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Invalid model output')
                prediction = logits.sigmoid()[0, 0].numpy() >= .5
            metrics = overlap(prediction, target[0].numpy() > 0)
            saved_dice = None if saved['dice_unverified'] == 'N/A' else float(saved['dice_unverified'])
            if (metrics['predicted_pixels'] != int(saved['prediction_foreground_pixels'])
                    or abs(metrics['predicted_foreground_percent']-float(saved['prediction_foreground_percent'])) > 1e-9
                    or metrics['dice_unverified'] != saved_dice or saved['checkpoint_position'] != tag):
                raise ValueError(f"Fresh inference disagrees with saved metrics: {saved['filename']}")
            record = {'filename': image_path.name, **metrics}
            figure = render(original, raw, prediction, record, f'{group} coverage / empty UNVERIFIED', tag)
            buffer = io.BytesIO()
            figure.save(buffer, format='PNG')
            figure.close()
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.load()
                if decoded.size != (1440, 530):
                    raise ValueError('Invalid preview dimensions')
            name = f'{number:02d}_{group}_{image_path.stem}.png'
            pngs.append((name, buffer.getvalue()))
            manifest.append(dict(group=group, filename=image_path.name, split='valid', image_path=str(image_path),
                mask_path=str(mask_path), image_sha256=before[image_path], mask_sha256=before[mask_path],
                source_csv=str(SOURCE), source_csv_sha256=before[SOURCE], checkpoint=str(CHECKPOINT),
                checkpoint_sha256_before=before[CHECKPOINT], checkpoint_position=tag, threshold=.5,
                prediction_coverage_percent=metrics['predicted_foreground_percent'],
                dice_unverified='N/A' if saved_dice is None else saved_dice, dice_status=metrics['metric_status'],
                reference_status='empty_UNVERIFIED', preview=name, review_status='unreviewed', notes=NOTICE))
            print(f"{group}: {image_path.name} | coverage={metrics['predicted_foreground_percent']:.3f}% | Dice={saved_dice if saved_dice is not None else 'N/A (both empty)'}")
        verify_hashes(before)
        after = digest(CHECKPOINT)
        for row in manifest:
            row['checkpoint_sha256_after'] = after
        if not args.check_only:
            parent = ROOT/'empty_reference_reviews'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            for name, data in pngs:
                with (output/name).open('xb') as stream:
                    stream.write(data)
            with (output/'review_manifest.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
        print(NOTICE)
        print(f'Checkpoint SHA-256 before: {before[CHECKPOINT]}\nCheckpoint SHA-256 after: {after}')
        print(f'Saved {len(manifest)} reviews: {output}' if output else f'Checks passed: {len(manifest)} reviews rendered in memory; no files written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Partial outputs may remain in {output}; no overwrite or repair attempted.', file=sys.stderr)
        return 1
    finally:
        verify_hashes(before)


if __name__ == '__main__':
    raise SystemExit(main())
