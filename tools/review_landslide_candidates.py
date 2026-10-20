"""CPU inference for review only; no geological classification or training.

Folder input is nonrecursive. Do not supply copied/renamed test images: path
guards cannot establish the provenance of arbitrary external copies.
--check-only renders in memory without writing outputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import textwrap
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_unet_predictions import ROOT, digest, load_model, font
from dataset_loader import DATASET_ROOT
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

CHECKPOINT = ROOT/'experiments/experiment_b_exposure/runs/20261013T133640Z_402e62786bab407598214d53f521fae9/best.pth'
LABEL = 'Unverified lunar landslide candidate — geological confirmation required.'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def guarded(path):
    # Check lexical and resolved paths before reading contents, including links.
    for candidate in (path.absolute(), path.resolve()):
        if any(part.casefold() == 'test' for part in candidate.parts):
            raise ValueError(f'Test path refused: {path}')
        if candidate.is_relative_to((DATASET_ROOT/'test').absolute()):
            raise ValueError(f'Test split refused: {path}')
    return path.resolve()


def inputs(path):
    path = guarded(path)
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'Unsupported image extension: {path}')
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Input not found: {path}')
    result = []
    for entry in sorted(path.iterdir()):
        if entry.suffix.lower() in EXTENSIONS:
            safe = guarded(entry)
            if not safe.is_file():
                raise ValueError(f'Image entry is not a regular file: {entry}')
            result.append(safe)
        else:
            print(f'Not selected (nonrecursive image-only input): {entry.name}')
    if not result:
        raise ValueError(f'No supported images directly inside: {path}')
    if len(result) != len(set(result)):
        raise ValueError('Multiple entries resolve to the same image')
    return result


def preprocess(original):
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32)/255.0
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def render(original, prediction, filename, threshold, tag):
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    overlay[foreground] = np.rint(.65*rgb[foreground]+.35*np.array([255, 0, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (1500, 710), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), LABEL, font=font(20), fill='#8b3400')
    draw.text((12, 40), filename, font=font(16), fill='black')
    checkpoint_text = 'Checkpoint: '+str(CHECKPOINT.relative_to(ROOT))
    for n, line in enumerate(textwrap.wrap(checkpoint_text, 145)):
        draw.text((12, 65+n*22), line, font=font(15), fill='black')
    draw.text((12, 114), f'{tag} | threshold >= {threshold:g} | foreground at 256x256: {np.mean(prediction > 0)*100:.3f}%', font=font(18), fill='black')
    for n, (array, title, nearest) in enumerate(((rgb, 'Original image', False),
            (prediction, 'Predicted binary mask (256x256)', True), (overlay, 'Prediction overlay (red 35%)', False))):
        x = 12+n*496
        draw.text((x, 150), title, font=font(18), fill='black')
        panel = Image.fromarray(array)
        panel.thumbnail((475, 475), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas.paste(panel, (x+(475-panel.width)//2, 182+(475-panel.height)//2))
        panel.close()
    draw.text((12, 677), 'Model trained against unverified rectangular references. No geological accuracy or confirmed detection is established.', font=font(16), fill='#8b3400')
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One image or a nonrecursive image directory')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error('--threshold must be finite and between 0 and 1')
    before, output = {}, None
    try:
        paths = inputs(args.input)
        for path in [CHECKPOINT, ROOT/'unet_model.py', ROOT/'dataset_loader.py', ROOT/'tools/preview_unet_predictions.py']+paths:
            before[path] = digest(path)
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        records = []
        for number, path in enumerate(paths, 1):
            original = cv2.imread(str(guarded(path)), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError(f'Unreadable image (not skipped): {path}')
            with torch.no_grad():
                logits = model(preprocess(original).unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f'Invalid model output: {path}')
                prediction = (logits.sigmoid()[0, 0].numpy() >= args.threshold).astype(np.uint8)*255
            figure = render(original, prediction, path.name, args.threshold, tag)
            buffer = io.BytesIO()
            figure.save(buffer, format='PNG')
            figure.close()
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.load()
                if decoded.size != (1500, 710):
                    raise ValueError('Invalid preview dimensions')
            name = f'{number:04d}_{path.stem}_candidate_review.png'
            if not args.check_only:
                if output is None:
                    parent = ROOT/'candidate_reviews'
                    if parent.resolve() != parent:
                        raise ValueError('Redirected output directory refused')
                    parent.mkdir(exist_ok=True)
                    output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
                    output.mkdir(exist_ok=False)
                with (output/name).open('xb') as stream:
                    stream.write(buffer.getvalue())
            coverage = float(np.mean(prediction > 0)*100)
            records.append(dict(input_path=str(path), image_sha256=before[path], checkpoint=str(CHECKPOINT),
                                checkpoint_sha256=before[CHECKPOINT], checkpoint_position=tag,
                                threshold=args.threshold, foreground_percent_256x256=coverage,
                                preview=name, interpretation=LABEL))
            print(f'{number}/{len(paths)} {path.name}: foreground={coverage:.3f}% | {LABEL}', flush=True)
        if output:
            with (output/'review_records.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        print(f'Reviews saved: {output}' if output else 'Checks passed; no files written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        if output:
            print(f'Incomplete run; partial previews remain in {output}. No overwrite or repair attempted.', file=sys.stderr)
        return 1
    finally:
        changed = [str(path) for path, expected in before.items() if not path.is_file() or digest(path) != expected]
        if changed:
            raise RuntimeError(f'Protected file hash discrepancy; no repair attempted: {changed}')


if __name__ == '__main__':
    raise SystemExit(main())
