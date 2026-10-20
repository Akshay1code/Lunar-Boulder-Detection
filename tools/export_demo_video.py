"""Export an existing completed PNG video demo as a labelled MP4 slideshow.

No model loading or inference. Original frame timing is NOT reproduced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run):
    run = Path(run).resolve()
    allowed = (ROOT/'video_demonstrations').resolve()
    if not run.is_relative_to(allowed) or 'test' in [x.lower() for x in run.parts]:
        raise ValueError('Select a completed run inside video_demonstrations')
    report_path = run/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    frames = report.get('frames', [])
    if report.get('status') != 'completed' or not 1 <= len(frames) <= 30:
        raise ValueError('Expected a completed run containing 1–30 sampled frames')
    paths = [report_path]
    for frame in frames:
        index = frame['requested_frame']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid frame index')
        for key in ('original', 'overlay'):
            p = (run/f'frame_{index:06d}_{key}.png').resolve()
            if not p.is_relative_to(run) or not p.is_file():
                raise ValueError(f'Missing or unsafe frame: {p}')
            paths.append(p)
    before = {str(p): sha(p) for p in paths}
    out = ROOT/'video_presentations'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
    out.mkdir(parents=True, exist_ok=False)
    target = out/'sampled_predictions.mp4'
    writer = None
    reader = None
    result = dict(status='incomplete', source_run=str(run), source_hashes=before,
                  sampled_frames=len(frames), playback_fps=10, seconds_per_sample=1,
                  interpretation='Sampled-frame slideshow, NOT source timing or real-time detection',
                  synthetic_source='SYNTHETIC' in str(report.get('source', '')).upper())
    try:
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1024, 640))
        if not writer.isOpened():
            raise RuntimeError('Installed OpenCV cannot encode mp4v. Existing PNGs remain usable.')
        for frame in frames:
            index = frame['requested_frame']
            canvas = np.full((640, 1024, 3), 245, np.uint8)
            lines = [
                'EXPERIMENTAL SEGMENTATION - NOT GEOLOGICALLY VERIFIED',
                f"Sample {index} | source t={frame['requested_seconds']:.2f}s | "
                f"threshold {report['threshold']:.2f} | coverage {frame['coverage_percent']:.3f}%",
                'Original sampled frame                         Prediction overlay',
            ]
            for n, text in enumerate(lines):
                cv2.putText(canvas, text, (16, 30+n*30), cv2.FONT_HERSHEY_SIMPLEX,
                            .60, (35, 35, 35), 1, cv2.LINE_AA)
            for j, key in enumerate(('original', 'overlay')):
                image = cv2.imread(str(run/f'frame_{index:06d}_{key}.png'))
                if image is None:
                    raise ValueError(f'Unreadable {key} frame {index}')
                h, w = image.shape[:2]
                scale = min(496/w, 440/h)
                tile = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                                  interpolation=cv2.INTER_AREA)
                x = j*512+8+(496-tile.shape[1])//2
                y = 112+(440-tile.shape[0])//2
                canvas[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            footer = ('SYNTHETIC CODEC TEST - not lunar footage' if result['synthetic_source']
                      else 'No motion analysis or geological activity detection')
            cv2.putText(canvas, footer, (16, 586), cv2.FONT_HERSHEY_SIMPLEX, .62, (30, 30, 160), 1)
            cv2.putText(canvas, '1 second per sample; slideshow timing, not original video speed',
                        (16, 620), cv2.FONT_HERSHEY_SIMPLEX, .60, (35, 35, 35), 1)
            for _ in range(10):
                writer.write(canvas)
        writer.release()
        reader = cv2.VideoCapture(str(target))
        if not reader.isOpened():
            raise RuntimeError('MP4 was written but cannot be decoded')
        count = 0
        while True:
            ok, image = reader.read()
            if not ok:
                break
            if image.shape != (640, 1024, 3):
                raise RuntimeError('Incorrect exported frame dimensions')
            count += 1
        if count != len(frames)*10:
            raise RuntimeError(f'MP4 validation failed: decoded {count} frames')
        result.update(status='completed', encoded_frames=count, output_sha256=sha(target))
        return target
    except BaseException as exc:
        result['error'] = str(exc)
        raise
    finally:
        if writer is not None:
            writer.release()
        if reader is not None:
            reader.release()
        changed = [p for p, h in before.items() if sha(Path(p)) != h]
        result['changed_source_files'] = changed
        if changed:
            result['status'] = 'integrity_failure'
        with (out/'export_report.json').open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        if changed:
            raise RuntimeError(f'Source files changed: {changed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    try:
        print(export(args.run_directory))
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        print(f'MP4 export failed: {exc}', file=sys.stderr)
        sys.exit(1)
