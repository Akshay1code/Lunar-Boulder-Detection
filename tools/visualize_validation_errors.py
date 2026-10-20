"""Validation-only F error review. No training; all references are unverified.

Run with python -B tools/visualize_validation_errors.py. Each invocation writes
only to a new unique directory. CPU inference uses the original loader/model.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import csv
import json
from datetime import datetime, timezone
import uuid
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from evaluate_experiment_e_validation import validation_dataset, overlap, aggregate, verify_hashes
from preview_unet_predictions import ROOT, digest, load_model, font

RUN = ROOT / 'experiments/experiment_f_dice_loss/runs/20261014T102910Z_262e760875cd4383a83f9d45a975d403'
CHECKPOINT = RUN / 'best.pth'
BASELINE = ROOT / 'validation_evaluations/20261017T155435Z_F_vs_H_e4c59285706a444493b8a99b4c3d34c7/per_image_metrics.csv'
THRESHOLD = 0.5


def fmt(value):
    return 'N/A' if value is None else f'{value:.6f}'


def render_row(image_path, mask_path, prediction, row):
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if original is None or reference is None or original.shape[:2] != reference.shape:
        raise ValueError(f'Invalid original pair: {image_path}')
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    expanded = cv2.resize(prediction.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    overlay[expanded] = np.rint(.65 * rgb[expanded] + .35 * np.array([255, 0, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (1080, 345), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((8, 4), f"ID {row['coco_image_id']} | valid | {image_path.name}", font=font(13), fill='black')
    d.text((8, 23), f"Dice {fmt(row['dice_unverified'])} | IoU {fmt(row['iou_unverified'])} | ref {row['reference_foreground_percent']:.2f}% | pred {row['predicted_foreground_percent']:.2f}% | threshold >= 0.5", font=font(14), fill='black')
    panels = [(rgb, 'Original image (thumbnail)', False), (reference, 'Reference - UNVERIFIED', True),
              (prediction.astype(np.uint8)*255, 'Prediction - experimental', True), (overlay, 'Prediction overlay: red 35%', False)]
    for j, (a, title, nearest) in enumerate(panels):
        x = 8 + 268*j
        d.text((x, 48), title, font=font(14), fill='black')
        panel = Image.fromarray(a)
        panel.thumbnail((256, 256), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas.paste(panel, (x+(256-panel.width)//2, 72+(256-panel.height)//2))
    return canvas


def main():
    before = {}
    output = None
    try:
        protected = [CHECKPOINT, RUN/'last.pth', RUN/'configuration.json', RUN/'summary.json', BASELINE,
                     ROOT/'unet_model.py', ROOT/'dataset_loader.py', Path(__file__).resolve(),
                     ROOT/'tools/evaluate_experiment_e_validation.py', ROOT/'tools/preview_unet_predictions.py']
        before.update({p: digest(p) for p in protected})
        config = json.loads((RUN/'configuration.json').read_text())
        summary = json.loads((RUN/'summary.json').read_text())
        if (config['model'], config['image_size'], config['threshold']) != ('UNet', 256, THRESHOLD):
            raise ValueError('Configuration differs from existing evaluation methodology')
        torch.set_num_threads(2)
        model, tag = load_model(CHECKPOINT)
        if tag != f"update {summary['best_update']}":
            raise ValueError('Best checkpoint update disagrees with run summary')
        dataset = validation_dataset()
        coco_path = dataset.image_dir/'_annotations.coco.json'
        before[coco_path] = digest(coco_path)
        coco = json.loads(coco_path.read_text())
        ids = {x['file_name']: x['id'] for x in coco['images']}
        if len(ids) != len(coco['images']):
            raise ValueError('Duplicate COCO filenames')
        for pair in dataset.pairs:
            before.update({p: digest(p) for p in pair})
        with BASELINE.open(encoding='utf-8-sig', newline='') as f:
            old = {r['filename']: r for r in csv.DictReader(f) if r['model'] == 'F'}
        if set(old) != {p.name for p, _ in dataset.pairs}:
            raise ValueError('Saved comparison uses a different validation selection')
        rows, predictions = [], []
        for i, (ip, mp) in enumerate(dataset.pairs):
            raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8 or not set(np.unique(raw)) <= {0, 255}:
                raise ValueError(f'Invalid reference: {mp}')
            image, target = dataset[i]
            ref = target[0].numpy() > 0
            if bool(raw.any()) != bool(ref.any()):
                raise ValueError('Reference emptiness changed during resizing')
            with torch.no_grad():
                logits = model(image.unsqueeze(0))
                if logits.shape != (1, 1, 256, 256) or not torch.isfinite(logits).all():
                    raise ValueError(f'Invalid inference: {ip}')
                pred = logits.sigmoid()[0, 0].numpy() >= THRESHOLD
            metrics = overlap(pred, ref)
            previous = old[ip.name]
            if previous['checkpoint_sha256'] != before[CHECKPOINT] or previous['image_sha256'] != before[ip] or previous['mask_sha256'] != before[mp]:
                raise ValueError('Saved comparison provenance mismatch')
            for field in ['dice_unverified', 'iou_unverified', 'predicted_foreground_percent', 'reference_foreground_percent']:
                expected = None if previous[field] == 'N/A' else float(previous[field])
                actual = metrics[field]
                if (expected is None) != (actual is None) or (actual is not None and abs(actual-expected) > 1e-10):
                    raise ValueError(f'Saved metric mismatch: {ip.name}, {field}')
            p, r, tp = metrics['predicted_pixels'], metrics['reference_pixels'], metrics['intersection_pixels']
            rows.append(dict(filename=ip.name, coco_image_id=ids[ip.name], split='valid',
                             image_path=str(ip), mask_path=str(mp), **metrics,
                             extra_pixels_unverified=p-tp, missed_pixels_unverified=r-tp,
                             extra_image_percent=100*(p-tp)/ref.size, missed_image_percent=100*(r-tp)/ref.size,
                             missed_reference_percent=100*(r-tp)/r if r else None,
                             precision_unverified=tp/p if p else None, recall_unverified=tp/r if r else None))
            predictions.append(pred)
            if (i+1) % 20 == 0:
                print(f'Evaluated {i+1}/80; saved F metrics reproduced', flush=True)
        nonempty = [r for r in rows if r['reference_pixels']]
        empty = [r for r in rows if not r['reference_pixels']]
        if (len(nonempty), len(empty)) != (19, 61):
            raise ValueError('Reference-group counts changed')
        parent = ROOT/'experiments/validation_error_analysis'
        if parent.resolve() != parent:
            raise ValueError('Redirected output directory refused')
        parent.mkdir(exist_ok=True)
        output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex)
        output.mkdir(exist_ok=False)
        ordered = sorted(range(80), key=lambda i: (not bool(rows[i]['reference_pixels']), rows[i]['dice_unverified'] if rows[i]['reference_pixels'] else -rows[i]['extra_image_percent'], rows[i]['filename']))
        sheets = []
        for start in range(0, 80, 4):
            sheet = Image.new('RGB', (1080, 1430), 'white')
            ImageDraw.Draw(sheet).text((8, 5), 'F validation | UNVERIFIED references | Empty references are NOT confirmed negatives', font=font(16), fill='black')
            name = f'contact_{start//4+1:02d}.png'
            for j, idx in enumerate(ordered[start:start+4]):
                ip, mp = dataset.pairs[idx]
                panel = render_row(ip, mp, predictions[idx], rows[idx])
                sheet.paste(panel, (0, 42+345*j))
                panel.close()
                rows[idx]['contact_sheet'] = name
                rows[idx]['sheet_row'] = j+1
            with (output/name).open('xb') as f:
                sheet.save(f, format='PNG')
            sheet.close()
            with Image.open(output/name) as im:
                im.load()
                assert im.size == (1080, 1430)
            sheets.append(name)
        with (output/'per_image_metrics.csv').open('x', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: 'N/A' if v is None else v for k, v in r.items()} for r in rows)
        with (output/'per_image_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            assert len(list(csv.DictReader(f))) == 80
        text = ['# Experiment F validation error analysis',
                'All scores measure agreement with UNVERIFIED rectangular references, not geological accuracy.',
                f'Checkpoint: `{CHECKPOINT}` ({tag}); SHA-256 `{before[CHECKPOINT]}`.',
                'Existing UNet, original loader, RGB /255, INTER_AREA 256x256; masks INTER_NEAREST >0. CPU, eval(), no_grad(), threshold >=0.5. No augmentation.',
                'Dice=2TP/(P+R), IoU=TP/(P+R-TP), no smoothing. Both-empty is N/A and excluded from overlap means. Precision/recall are N/A when their respective denominator is zero. Means are per-image, not pooled pixels.',
                'Extra/false-positive pixels mean predicted pixels outside the reference ONLY. Missed pixels mean reference pixels outside prediction. Neither establishes geological mistakes. Extra coverage uses total image area; missed-reference percentage uses reference area.',
                aggregate('All validation', rows), aggregate('Nonempty references', nonempty), aggregate('Empty references', empty),
                f"Empty-reference images with predictions: {sum(r['predicted_pixels']>0 for r in empty)}/61.",
                f"Mean precision/recall on nonempty references: {np.mean([r['precision_unverified'] for r in nonempty if r['precision_unverified'] is not None]):.6f} / {np.mean([r['recall_unverified'] for r in nonempty]):.6f}.",
                '## Ranked examples (deterministic filename tie-breaks)']
        groups = [('Lowest Dice, nonempty', sorted(nonempty, key=lambda r:(r['dice_unverified'],r['filename']))),
                  ('Highest Dice, nonempty', sorted(nonempty,key=lambda r:(-r['dice_unverified'],r['filename']))),
                  ('Highest extra/false-positive coverage, all', sorted(rows,key=lambda r:(-r['extra_image_percent'],r['filename']))),
                  ('Most missed reference fraction', sorted(nonempty,key=lambda r:(-r['missed_reference_percent'],r['filename']))),
                  ('Empty references with highest prediction coverage',sorted(empty,key=lambda r:(-r['predicted_foreground_percent'],r['filename'])))]
        for label, group in groups:
            text += ['### '+label, '| ID / filename | Dice | IoU | Extra image % | Missed reference % | Sheet / row |', '|---|---:|---:|---:|---:|---|']
            for r in group[:5]:
                text.append(f"| {r['coco_image_id']} / {r['filename']} | {fmt(r['dice_unverified'])} | {fmt(r['iou_unverified'])} | {r['extra_image_percent']:.3f} | {fmt(r['missed_reference_percent'])} | [{r['contact_sheet']}]({r['contact_sheet']}) / {r['sheet_row']} |")
        missed = sum(r['missed_pixels_unverified'] for r in nonempty)
        top5 = sum(r['missed_pixels_unverified'] for r in sorted(nonempty,key=lambda r:-r['missed_pixels_unverified'])[:5])
        text += ['## Measured failure patterns',
                 f"{sum(r['dice_unverified']<.3 for r in nonempty)}/19 nonempty references have Dice below 0.3. The five images with most missed pixels account for {100*top5/missed:.2f}% of all missed reference pixels.",
                 'Extra predictions on empty references are widespread, not proof of false geological detections. Incomplete labeling and model errors cannot be separated with these references alone.',
                 '## Visual interpretation limits',
                 'Contact sheets support human inspection; the script does not classify terrain or infer geological boundaries. Box-like edges, fragmentation and displaced regions need visual review; their cause cannot be inferred from overlap metrics.',
                 '## Recommended first improvement',
                 'Test annotation reliability first: obtain source-image context and independent review of the lowest-agreement examples plus high-coverage empty references. Keep F fixed. Do not automatically relabel or use empty masks as negatives. This is more informative than another loss or augmentation change while boundary validity is unresolved.',
                 '## Checks and reproducibility',
                 '80/80 inference calls succeeded; every saved F Dice/IoU/coverage reproduced within 1e-10. All 20 PNGs reopened and dimensions checked; CSV has 80 rows. No test split or training accessed. Sheets show nonempty references in ascending Dice, then empty references in descending prediction coverage. Native reference thumbnails are shown; metrics use the original 256x256 evaluation grid.',
                 'Protected checkpoint/configuration/source/validation-file hashes matched after rendering. See provenance.json. Existing assets were not rewritten.']
        verify_hashes(before)
        with (output/'report.md').open('x', encoding='utf-8') as f:
            f.write('\n\n'.join(text)+'\n')
        with (output/'provenance.json').open('x', encoding='utf-8') as f:
            json.dump(dict(checkpoint=str(CHECKPOINT), checkpoint_position=tag, threshold=THRESHOLD,
                           protected_sha256={str(k):v for k,v in before.items()}, sheets=sheets,
                           python=sys.version, torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),f,indent=2)
        print(aggregate('Nonempty references', nonempty))
        print(f'OUTPUT: {output}', flush=True)
        return 0
    finally:
        verify_hashes(before)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'STOP: {exc}. Any partial output is preserved; no repair attempted.', file=sys.stderr)
        raise
