"""Validation-only Experiment E previews of agreement with unverified references.

Ranks fresh inference at threshold 0.5. No training or test access.
--check-only evaluates and renders in memory without writing files.
"""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_experiment_e_validation import (
    ROOT, RUN, CHECKPOINT, THRESHOLD, DISCLAIMER, validation_dataset,
    overlap, verify_hashes, digest, load_model,
)
from preview_unet_predictions import font
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def select_groups(records):
    nonempty = [r for r in records if r['reference_pixels'] > 0]
    candidates = [
        ('lowest_dice', sorted(nonempty, key=lambda r: (r['dice_unverified'], r['filename']))),
        ('empty_reference_prediction', sorted(
            [r for r in records if not r['reference_pixels'] and r['predicted_pixels']],
            key=lambda r: (-r['predicted_pixels'], r['filename']))),
        ('highest_dice', sorted(nonempty, key=lambda r: (-r['dice_unverified'], r['filename']))),
    ]
    used, groups = set(), []
    for name, candidates_for_group in candidates:
        eligible = [r for r in candidates_for_group if r['filename'] not in used]
        selected = eligible[:3]
        used.update(r['filename'] for r in selected)
        groups.append((name, selected, len(eligible)))
    return groups


def render(original, reference, prediction, record, group, tag):
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enlarged = cv2.resize(prediction.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    overlay[enlarged] = np.rint(.65 * rgb[enlarged] + .35 * np.array([255, 0, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (1440, 530), 'white')
    draw = ImageDraw.Draw(canvas)
    dice = 'N/A' if record['dice_unverified'] is None else f"{record['dice_unverified']:.6f}"
    draw.text((12, 8), f"VALID | {record['filename']}", font=font(16), fill='black')
    draw.text((12, 34), f"Experiment E best.pth | {tag} | {group} | threshold >= 0.5", font=font(16), fill='black')
    draw.text((12, 60), f"At 256x256: Dice (unverified) {dice} | reference foreground {record['reference_foreground_percent']:.3f}%"
              f" | predicted foreground {record['predicted_foreground_percent']:.3f}%", font=font(16), fill='black')
    panels = [(rgb, 'Original lunar image', False), (reference, 'Reference mask - UNVERIFIED', True),
              (prediction.astype(np.uint8)*255, 'Predicted binary mask', True),
              (overlay, 'Prediction overlay (red 35%)', False)]
    for i, (array, title, nearest) in enumerate(panels):
        x = 12 + i*357
        draw.text((x, 98), title, font=font(16), fill='black')
        panel = Image.fromarray(array)
        panel.thumbnail((340, 340), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas.paste(panel, (x + (340-panel.width)//2, 125 + (340-panel.height)//2))
        panel.close()
    draw.text((12, 478), 'Unverified rectangular references; visual agreement does not establish geological accuracy.', font=font(16), fill='#8b3400')
    draw.text((12, 502), 'Empty reference masks are not confirmed negatives. These groups describe reference agreement only.', font=font(16), fill='#8b3400')
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        protected = [CHECKPOINT, RUN/'configuration.json'] + [ROOT/name for name in (
            'unet_model.py', 'dataset_loader.py', 'tools/preview_unet_predictions.py',
            'tools/preview_experiment_e.py', 'tools/evaluate_experiment_e_validation.py')]
        for path in protected:
            before[path] = digest(path)
        config = json.loads((RUN/'configuration.json').read_text(encoding='utf-8'))
        if THRESHOLD != 0.5 or config['threshold'] != 0.5 or config['image_size'] != 256 or config['model'] != 'UNet':
            raise ValueError('Unexpected Experiment E preprocessing/model/threshold configuration')
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        dataset = validation_dataset()
        for pair in dataset.pairs:
            for path in pair:
                before[path] = digest(path)
        records, predictions = [], []
        for index, (image_path, mask_path) in enumerate(dataset.pairs):
            raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or not set(np.unique(raw)) <= {0, 255}:
                raise ValueError(f'Invalid reference mask: {mask_path}')
            image, target = dataset[index]
            reference = target[0].numpy() > 0
            if bool(raw.any()) != bool(reference.any()):
                raise ValueError(f'Reference emptiness changed during preprocessing: {mask_path}')
            with torch.no_grad():
                logits = model(image.unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f'Invalid model output: {image_path}')
                prediction = logits.sigmoid()[0, 0].numpy() >= THRESHOLD
            predictions.append(prediction)
            records.append({'index': index, 'filename': image_path.name, **overlap(prediction, reference)})
            if (index+1) % 20 == 0:
                print(f'Evaluated {index+1}/{len(dataset)} validation images', flush=True)
        groups = select_groups(records)
        verify_hashes(before)
        if not args.check_only:
            parent = ROOT/'validation_previews'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
        total = 0
        for group, selected, available in groups:
            print(f'{group}: selected {len(selected)}; {available} eligible after duplicate exclusion')
            if len(selected) < 3:
                print(f'NOTE: {group} has fewer than 3 available images.')
            for rank, record in enumerate(selected, 1):
                index = record['index']
                image_path, mask_path = dataset.pairs[index]
                original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                reference = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if original is None or reference is None or original.shape[:2] != reference.shape:
                    raise ValueError(f'Unreadable or dimension-mismatched pair: {image_path}')
                figure = render(original, reference, predictions[index], record, group, tag)
                buffer = io.BytesIO()
                figure.save(buffer, format='PNG')
                figure.close()
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    decoded.load()
                    if decoded.size != (1440, 530):
                        raise ValueError('Invalid rendered PNG dimensions')
                if output:
                    with (output/f'{group}_{rank:02d}_{image_path.stem}.png').open('xb') as stream:
                        stream.write(buffer.getvalue())
                print(f"  {record['filename']} | Dice={record['dice_unverified']:.6f}"
                      f" | reference={record['reference_foreground_percent']:.3f}%"
                      f" | predicted={record['predicted_foreground_percent']:.3f}%")
                total += 1
        print(DISCLAIMER)
        print(f'Saved {total} PNGs: {output}' if output else f'Checks passed: {total} PNGs rendered in memory; no outputs written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Partial previews may remain in {output}; no overwrite or repair attempted.', file=sys.stderr)
        return 1
    finally:
        verify_hashes(before)


if __name__ == '__main__':
    raise SystemExit(main())
