"""Inspect Experiment E sigmoid probabilities on one fixed training image.

CPU inference only. --check-only renders in memory without saving outputs.
Reference masks and model probabilities are not geological confirmation.
"""
import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_unet_predictions import ROOT, training_dataset, load_model, digest, font
from preview_experiment_e import CHECKPOINT
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

FILENAME = 'ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg'


def colors(probabilities):
    # Fixed 0..1 scale, never normalized to this image's observed min/max.
    levels = np.rint(np.clip(probabilities, 0, 1)*255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(levels, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def render(original, reference, probabilities, tag):
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize((probabilities >= .5).astype(np.uint8),
                            (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    overlay[foreground] = np.rint(.65*rgb[foreground]+.35*np.array([255, 0, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (1720, 610), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'TRAIN | {FILENAME}', font=font(18), fill='black')
    draw.text((12, 36), f'Experiment E best.pth | {tag} | overlay threshold >= 0.5 | foreground: {np.mean(probabilities >= .5)*100:.3f}%', font=font(18), fill='black')
    panels = [(rgb, 'Original lunar image', False), (reference, 'Reference mask - UNVERIFIED', True),
              (colors(probabilities), 'Sigmoid probability (fixed 0-1 scale)', True),
              (overlay, 'Prediction overlay (red 35%)', False)]
    for i, (array, title, nearest) in enumerate(panels):
        x = 12+i*425
        draw.text((x, 80), title, font=font(18), fill='black')
        panel = Image.fromarray(array)
        panel.thumbnail((400, 400), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas.paste(panel, (x+(400-panel.width)//2, 112+(400-panel.height)//2))
        panel.close()
    # Color bar uses exactly the same mapping as the probability panel.
    gradient = np.tile(np.linspace(0, 1, 400, dtype=np.float32), (18, 1))
    bar = Image.fromarray(colors(gradient))
    canvas.paste(bar, (862, 524))
    bar.close()
    for value in (0, .25, .5, .75, 1):
        x = 862+int(value*399)
        draw.line((x, 542, x, 547), fill='black')
        draw.text((x-6, 550), f'{value:g}', font=font(14), fill='black')
    draw.text((12, 582), 'Unverified rectangular references; sigmoid model scores are not calibrated geological confidence or confirmed landslide detections.', font=font(17), fill='#8b3400')
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        dataset = training_dataset()
        matches = [i for i, (image, _) in enumerate(dataset.pairs) if image.name == FILENAME]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one training pair for {FILENAME}; found {len(matches)}')
        index = matches[0]
        image_path, mask_path = dataset.pairs[index]
        for path in [image_path, mask_path, CHECKPOINT] + [ROOT/name for name in (
                'unet_model.py', 'dataset_loader.py', 'tools/preview_unet_predictions.py', 'tools/preview_experiment_e.py')]:
            before[path] = digest(path)
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        tensor, _ = dataset[index]  # Exact existing preprocessing and dimension checks.
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if original is None or reference is None or reference.ndim != 2 or reference.dtype != np.uint8:
            raise ValueError('Unreadable image or invalid grayscale reference mask')
        if not set(np.unique(reference)) <= {0, 255} or original.shape[:2] != reference.shape:
            raise ValueError('Unexpected reference values or mismatched original dimensions')
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
            if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Invalid model output')
            probabilities = logits.sigmoid()[0, 0].numpy()
        lines = [f'Image: {image_path}', f'Reference (UNVERIFIED): {mask_path}',
                 f'Checkpoint: {CHECKPOINT} ({tag})', f'Checkpoint SHA-256: {before[CHECKPOINT]}',
                 'Statistics at model resolution 256x256; population standard deviation.',
                 f'Minimum: {float(probabilities.min()):.10g}', f'Maximum: {float(probabilities.max()):.10g}',
                 f'Mean: {float(probabilities.mean()):.10g}', f'Standard deviation: {float(probabilities.std()):.10g}']
        for threshold in (.2, .5, .8):
            lines.append(f'Pixels at or above {threshold:.1f}: {np.mean(probabilities >= threshold)*100:.6f}%')
        lines += ['Threshold comparison uses >=, matching existing prediction scripts.',
                  'Heatmap uses fixed Viridis 0-1 scale (8-bit color display); no per-image min/max normalization.',
                  'probabilities.npy preserves the continuous float32 sigmoid values before thresholding.',
                  'Reference annotations are UNVERIFIED. No geological accuracy or calibrated confidence is established.']
        summary = '\n'.join(lines)+'\n'
        figure = render(original, reference, probabilities, tag)
        buffer = io.BytesIO()
        figure.save(buffer, format='PNG')
        figure.close()
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            decoded.load()
            if decoded.size != (1720, 610):
                raise ValueError('Invalid PNG dimensions')
        if not args.check_only:
            parent = ROOT/'probability_inspections'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output/'probability_comparison.png').open('xb') as stream:
                stream.write(buffer.getvalue())
            with (output/'probabilities.npy').open('xb') as stream:
                np.save(stream, probabilities, allow_pickle=False)
            with (output/'statistics.txt').open('x', encoding='utf-8') as stream:
                stream.write(summary)
        print(summary)
        print(f'Outputs saved: {output}' if output else 'Checks passed; no outputs written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Partial outputs may remain in {output}; no overwrite or repair attempted.', file=sys.stderr)
        return 1
    finally:
        changed = [str(path) for path, expected in before.items() if not path.is_file() or digest(path) != expected]
        if changed:
            raise RuntimeError(f'Protected file hash discrepancy; no repairs attempted: {changed}')


if __name__ == '__main__':
    raise SystemExit(main())
