"""Validation-only descriptive shape analysis against UNVERIFIED references.

CPU inference only. --check-only performs analysis without writing reports.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_experiment_e_validation import (
    ROOT, RUN, CHECKPOINT, THRESHOLD, validation_dataset, overlap,
    verify_hashes, digest, load_model,
)
import cv2
import numpy as np
import torch


def shape_metrics(mask):
    """One axis-aligned bounding box around ALL foreground pixels; inclusive edges."""
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Expected a 2D boolean mask')
    y, x = np.nonzero(mask)
    count = len(x)
    result = dict(foreground_percent=100*count/mask.size, foreground_pixels=count,
                  bbox_pixels=None, bbox_area_fraction=None, bbox_fill_ratio=None,
                  bbox_xmin=None, bbox_ymin=None, bbox_xmax=None, bbox_ymax=None)
    if count:
        xmin, xmax, ymin, ymax = int(x.min()), int(x.max()), int(y.min()), int(y.max())
        area = (xmax-xmin+1)*(ymax-ymin+1)
        result.update(bbox_pixels=area, bbox_area_fraction=area/mask.size,
                      bbox_fill_ratio=count/area, bbox_xmin=xmin, bbox_xmax=xmax,
                      bbox_ymin=ymin, bbox_ymax=ymax)
    return result


def summary_group(name, rows):
    lines = [f'{name}: {len(rows)} images']
    fields = [f'{prefix}_{metric}' for prefix in ('prediction', 'reference')
              for metric in ('foreground_percent', 'bbox_area_fraction', 'bbox_fill_ratio')]+['dice_unverified']
    for field in fields:
        values = [r[field] for r in rows if r[field] is not None]
        if values:
            quantiles = np.percentile(values, [0, 25, 50, 75, 100])
            lines.append(f'{field}: defined={len(values)}, N/A={len(rows)-len(values)}, mean={np.mean(values):.6f}; '
                         'min/Q1/median/Q3/max='+' / '.join(f'{v:.6f}' for v in quantiles))
        else:
            lines.append(f'{field}: N/A (0 defined)')
    paired = [r for r in rows if r['prediction_bbox_fill_ratio'] is not None and r['reference_bbox_fill_ratio'] is not None]
    if paired:
        delta = [r['prediction_bbox_fill_ratio']-r['reference_bbox_fill_ratio'] for r in paired]
        lines.append(f'Paired prediction-minus-reference fill ratio: n={len(delta)}, mean={np.mean(delta):.6f}, median={np.median(delta):.6f}')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        for path in [CHECKPOINT, RUN/'configuration.json']+[ROOT/name for name in (
                'unet_model.py', 'dataset_loader.py', 'tools/preview_unet_predictions.py',
                'tools/evaluate_experiment_e_validation.py')]:
            before[path] = digest(path)
        config = json.loads((RUN/'configuration.json').read_text(encoding='utf-8'))
        if THRESHOLD != .5 or config['threshold'] != .5 or config['image_size'] != 256 or config['model'] != 'UNet':
            raise ValueError('Unexpected baseline configuration')
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        dataset = validation_dataset()
        for pair in dataset.pairs:
            for path in pair:
                before[path] = digest(path)
        rows = []
        for index, (image_path, mask_path) in enumerate(dataset.pairs):
            raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or not set(np.unique(raw)) <= {0, 255}:
                raise ValueError(f'Invalid 0/255 grayscale reference: {mask_path}')
            tensor, target = dataset[index]
            reference = target[0].numpy() > 0
            if bool(raw.any()) != bool(reference.any()):
                raise ValueError(f'Reference emptiness changed during preprocessing: {mask_path}')
            with torch.no_grad():
                logits = model(tensor.unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f'Invalid output: {image_path}')
                prediction = logits.sigmoid()[0, 0].numpy() >= THRESHOLD
            row = dict(filename=image_path.name, split='valid', image_path=str(image_path), mask_path=str(mask_path),
                       image_sha256=before[image_path], mask_sha256=before[mask_path], checkpoint_path=str(CHECKPOINT),
                       checkpoint_sha256=before[CHECKPOINT], checkpoint_position=tag, threshold=THRESHOLD,
                       measurement_resolution='256x256', reference_status='nonempty_UNVERIFIED' if reference.any() else 'empty_UNVERIFIED',
                       dice_unverified=overlap(prediction, reference)['dice_unverified'])
            for prefix, mask in (('prediction', prediction), ('reference', reference)):
                row.update({f'{prefix}_{key}': value for key, value in shape_metrics(mask).items()})
            rows.append(row)
            if (index+1) % 20 == 0:
                print(f'Evaluated {index+1}/{len(dataset)} validation images', flush=True)
        nonempty = [r for r in rows if r['reference_foreground_pixels']]
        empty = [r for r in rows if not r['reference_foreground_pixels']]
        if (len(rows), len(nonempty), len(empty)) != (80, 19, 61):
            raise ValueError('Validation counts differ from expected 80 total/19 nonempty/61 empty')
        verify_hashes(before)
        after = digest(CHECKPOINT)
        summary = '\n\n'.join([
            'DESCRIPTIVE PREDICTION SHAPE ANALYSIS — references UNVERIFIED',
            f'Checkpoint: {CHECKPOINT}\nPosition: {tag}\nSHA-256 before: {before[CHECKPOINT]}\nSHA-256 after: {after}\nHash match: {before[CHECKPOINT] == after}',
            'Validation only; CPU eval/no_grad; threshold >= 0.5. Existing UNet and loader preprocessing unchanged.\n'
            'All measurements use the 256x256 model grid; masks use existing nearest-neighbor resizing.\n'
            'Coverage is a percentage. Bounding-box area/image area and fill ratio are fractions in [0,1].\n'
            'A single axis-aligned box encloses ALL foreground components; area=(xmax-xmin+1)*(ymax-ymin+1).\n'
            'Fill ratio=foreground pixels/box pixels. Empty masks: coverage=0, all box metrics=N/A.\n'
            'Dice=2*intersection/(prediction+reference counts); both-empty=N/A, one-empty=0, no smoothing.\n'
            'Aggregates weight images equally and exclude N/A, reporting the defined count.',
            summary_group('All validation', rows), summary_group('Nonempty UNVERIFIED references', nonempty),
            summary_group('Empty UNVERIFIED references', empty),
            'LIMITATIONS: Rectangularity alone does not prove annotation leakage or establish geological accuracy.\n'
            'High fill ratios describe occupancy of an axis-aligned bounding box, not a geological classification.\n'
            'Isolated pixels, multiple disjoint rectangles, rotation, threshold choice, and resizing affect these values.\n'
            'A low union-box fill ratio does not rule out individually rectangular components.\n'
            'Empty reference masks are not confirmed negatives. No categorical shape-bias conclusion is automated.',
        ])+'\n'
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: 'N/A' if v is None else v for k, v in row.items()} for row in rows)
        if len(list(csv.DictReader(io.StringIO(buffer.getvalue())))) != 80:
            raise ValueError('CSV serialization failed')
        if not args.check_only:
            parent = ROOT/'prediction_shape_analysis'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output/'per_image_shapes.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                stream.write(buffer.getvalue())
            with (output/'summary.txt').open('x', encoding='utf-8') as stream:
                stream.write(summary)
        print(summary)
        print(f'Reports saved: {output}' if output else 'Checks passed; no reports written.')
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
