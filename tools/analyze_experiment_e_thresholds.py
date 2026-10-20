"""Validation-only threshold analysis; CPU inference, never training.

Reuses each image's probabilities across six thresholds in memory. Existing
thresholds/configuration are never changed. --check-only writes no reports.
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
    ROOT, RUN, CHECKPOINT, DISCLAIMER, validation_dataset,
    overlap, verify_hashes, digest, load_model,
)
import cv2
import numpy as np
import torch

THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def mean_defined(records, key):
    values = [r[key] for r in records if r[key] is not None]
    return float(np.mean(values)) if values else None


def summarize(threshold, records):
    nonempty = [r for r in records if r['reference_pixels'] > 0]
    empty = [r for r in records if r['reference_pixels'] == 0]
    defined = sum(r['dice_unverified'] is not None for r in records)
    return {
        'threshold': threshold, 'validation_images': len(records),
        'nonempty_references': len(nonempty), 'empty_unverified_references': len(empty),
        'defined_all_scores': defined, 'undefined_both_empty': len(records)-defined,
        'defined_nonempty_scores': sum(r['dice_unverified'] is not None for r in nonempty),
        'mean_dice_all_unverified': mean_defined(records, 'dice_unverified'),
        'mean_iou_all_unverified': mean_defined(records, 'iou_unverified'),
        'mean_dice_nonempty_unverified': mean_defined(nonempty, 'dice_unverified'),
        'mean_iou_nonempty_unverified': mean_defined(nonempty, 'iou_unverified'),
        'mean_predicted_foreground_percent_all': mean_defined(records, 'predicted_foreground_percent'),
        'empty_references_with_nonempty_predictions': sum(r['predicted_pixels'] > 0 for r in empty),
        'interpretation': DISCLAIMER,
    }


def display(value):
    return 'N/A' if value is None else f'{value:.6f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Run inference and report checks without writing files')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        protected = [CHECKPOINT, RUN/'configuration.json'] + [ROOT/name for name in (
            'unet_model.py', 'dataset_loader.py', 'tools/preview_unet_predictions.py',
            'tools/evaluate_experiment_e_validation.py')]
        for path in protected:
            before[path] = digest(path)
        config = json.loads((RUN/'configuration.json').read_text(encoding='utf-8'))
        if config['image_size'] != 256 or config['model'] != 'UNet' or config['threshold'] != 0.5:
            raise ValueError('Unexpected baseline model/preprocessing/threshold configuration')
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        dataset = validation_dataset()
        for pair in dataset.pairs:
            for path in pair:
                before[path] = digest(path)
        metrics = {threshold: [] for threshold in THRESHOLDS}
        inference_calls = 0
        for index, (image_path, mask_path) in enumerate(dataset.pairs):
            raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or not set(np.unique(raw)) <= {0, 255}:
                raise ValueError(f'Invalid grayscale 0/255 reference: {mask_path}')
            image, target = dataset[index]  # Existing preprocessing and dimension checks.
            reference = target[0].numpy() > 0
            if bool(raw.any()) != bool(reference.any()):
                raise ValueError(f'Reference emptiness changed during preprocessing: {mask_path}')
            with torch.no_grad():
                logits = model(image.unsqueeze(0))
                inference_calls += 1
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f'Invalid model output: {image_path}')
                probabilities = logits.sigmoid()[0, 0].numpy()
            # Cache only one 256x256 float32 probability map (~256 KiB).
            previous_pixels = reference.size + 1
            for threshold in THRESHOLDS:
                record = overlap(probabilities >= threshold, reference)
                if record['predicted_pixels'] > previous_pixels:
                    raise ValueError('Foreground increased as threshold increased')
                previous_pixels = record['predicted_pixels']
                metrics[threshold].append(record)
            del probabilities
            if (index+1) % 20 == 0:
                print(f'Evaluated {index+1}/{len(dataset)} validation images at all six thresholds', flush=True)
        if inference_calls != len(dataset):
            raise ValueError('Expected exactly one inference per validation image')
        rows = [summarize(t, metrics[t]) for t in THRESHOLDS]
        for row in rows:
            if (row['validation_images'], row['nonempty_references'], row['empty_unverified_references']) != (80, 19, 61):
                raise ValueError(f'Unexpected validation counts: {row}')
        lines = [DISCLAIMER, f'Checkpoint: {CHECKPOINT}', f'SHA-256: {before[CHECKPOINT]}',
                 f'Checkpoint position: {tag}', f'CPU inference calls: {inference_calls}; validation images only.',
                 'Existing preprocessing: RGB, INTER_AREA 256x256, float32 /255; masks nearest-neighbor then >0.',
                 'Predicted foreground means sigmoid probability >= threshold.',
                 'Dice = 2*intersection/(predicted+reference pixels); IoU = intersection/union. No smoothing.',
                 'Both-empty scores: N/A, excluded from macro averages; never assigned perfect agreement.',
                 'Empty reference and nonempty prediction: Dice/IoU=0, describing reference disagreement only.',
                 'Empty references are NOT confirmed negatives. Missing or invalid masks cause failure.',
                 'All means are equal-weight per image; overlap means include only defined scores.',
                 'Defined-image counts can change across thresholds: all-image means are not a fixed-denominator comparison.',
                 'Nonempty-reference means use the same 19 images at every threshold.',
                 'Foreground coverage averages use all 80 images, including empty references.',
                 'This is exploratory validation analysis; no threshold is selected or applied to existing scripts.',
                 '', 'Threshold | Defined/all | Dice all | IoU all | Dice nonempty | IoU nonempty | Pred FG % all | Empty ref with prediction']
        for row in rows:
            lines.append(f"{row['threshold']:.2f} | {row['defined_all_scores']}/80 | "
                         + ' | '.join(display(row[key]) for key in (
                             'mean_dice_all_unverified', 'mean_iou_all_unverified',
                             'mean_dice_nonempty_unverified', 'mean_iou_nonempty_unverified',
                             'mean_predicted_foreground_percent_all'))
                         + f" | {row['empty_references_with_nonempty_predictions']}/61")
        summary = '\n'.join(lines)+'\n'
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: 'N/A' if v is None else v for k, v in row.items()} for row in rows)
        csv_text = buffer.getvalue()
        if len(list(csv.DictReader(io.StringIO(csv_text)))) != 6:
            raise ValueError('CSV serialization failed')
        verify_hashes(before)
        if not args.check_only:
            parent = ROOT/'threshold_analysis'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output/'threshold_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                stream.write(csv_text)
            with (output/'summary.txt').open('x', encoding='utf-8') as stream:
                stream.write(summary)
        print(summary)
        print(f'Reports saved: {output}' if output else 'Checks passed; no reports written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Partial reports may remain in {output}; no overwrite or repair attempted.', file=sys.stderr)
        return 1
    finally:
        verify_hashes(before)


if __name__ == '__main__':
    raise SystemExit(main())
