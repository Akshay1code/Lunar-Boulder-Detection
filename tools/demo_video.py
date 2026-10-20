"""Bounded offline video demonstration; no training, motion analysis or geological claims."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo
import cv2
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def process(source, fps=0.5, max_frames=20, threshold=0.5):
    source = Path(source).resolve()
    if any(p.lower() == 'test' for p in source.parts):
        raise ValueError('Test-split paths are not permitted')
    if source.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
        raise ValueError('Use MP4, AVI, MOV or MKV; decoder availability varies')
    if not source.is_file() or not 0 < source.stat().st_size <= 100*1024*1024:
        raise ValueError('Video must exist and be no larger than 100 MiB')
    if not np.isfinite(fps) or not 0.1 <= fps <= 2:
        raise ValueError('Sampling rate must be 0.1–2 frames/second')
    if not 1 <= max_frames <= 30:
        raise ValueError('Frame limit must be 1–30')
    if not np.isfinite(threshold) or not .2 <= threshold <= .8:
        raise ValueError('Threshold must be 0.20–0.80')
    protected = [source, demo.CHECKPOINT, demo.ROOT/'unet_model.py',
                 demo.ROOT/'dataset_loader.py']
    before = {str(p): digest(p) for p in protected}
    cap = cv2.VideoCapture(str(source))
    out = None
    report = dict(status='incomplete', source=str(source), threshold=threshold,
                  sampling_fps=fps, frame_limit=max_frames, frames=[],
                  checkpoint=str(demo.CHECKPOINT),
                  notice='Experimental predictions; geological confirmation unavailable. '
                  'Independent sampled frames, not motion or geological activity detection.')
    start = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be opened; unsupported codec or invalid file')
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(np.isfinite(x) and x > 0 for x in (native_fps, n, w, h)):
            raise ValueError('Missing/invalid video timing or dimensions')
        duration = n/native_fps
        if duration > 60 or w*h > 1920*1080:
            raise ValueError('Maximum duration 60 seconds and frame size 1920x1080 pixels')
        if fps > native_fps:
            raise ValueError('Requested sample rate exceeds source frame rate')
        out = demo.ROOT/'video_demonstrations'/(
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
        out.mkdir(parents=True, exist_ok=False)
        report.update(duration_seconds=duration, source_fps=native_fps,
                      source_frames=int(n), source_dimensions=[int(w), int(h)])
        torch.set_num_threads(2)
        model, tag = demo.load_model(demo.CHECKPOINT)
        indices = sorted(set(int(round(t*native_fps)) for t in
                             np.arange(0, duration, 1/fps)[:max_frames]))
        for index in indices:
            if index >= int(n):
                continue
            if time.perf_counter()-start > 120:
                raise RuntimeError('120-second processing budget reached; partial outputs retained')
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f'Codec cannot seek to requested frame {index}')
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f'Cannot decode requested frame {index}; not silently skipped')
            if frame.shape[0]*frame.shape[1] > 1920*1080:
                raise ValueError('Decoded frame exceeds pixel limit')
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                raise ValueError('Frame encoding failed')
            result = demo.render(demo.infer(encoded.tobytes(), model), tag, threshold)
            # One frame at a time, three PNGs; no video-sized RAM cache.
            for key in ('original', 'raw_mask', 'overlay'):
                with (out/f'frame_{index:06d}_{key}.png').open('xb') as f:
                    f.write(base64.b64decode(result[key].split(',')[1]))
            report['frames'].append(dict(requested_frame=index,
                requested_seconds=index/native_fps,
                decoder_position_after_read=cap.get(cv2.CAP_PROP_POS_FRAMES),
                coverage_percent=result['coverage'], regions=result['regions'],
                appearance=result['appearance']))
            print(f"Processed {len(report['frames'])}/{len(indices)} sampled frames; "
                  f"threshold {threshold:.2f}; coverage {result['coverage']:.3f}%", flush=True)
        if not report['frames']:
            raise ValueError('No frames processed')
        report['status'] = 'completed'
        return out
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        cap.release()
        report['elapsed_seconds'] = time.perf_counter()-start
        report['processed_frames'] = len(report['frames'])
        report['protected_sha256'] = before
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        report['changed_protected_files'] = changed
        if changed:
            report['status'] = 'integrity_failure'
        if out:
            with (out/'report.json').open('x', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'Outputs: {out}', flush=True)
        if changed:
            raise RuntimeError(f'Protected files changed: {changed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--fps', type=float, default=.5)
    parser.add_argument('--max-frames', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    try:
        process(args.video, args.fps, args.max_frames, args.threshold)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'Video demonstration failed: {exc}', file=sys.stderr)
        sys.exit(1)
