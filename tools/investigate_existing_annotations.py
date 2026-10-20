"""Descriptive train/valid COCO audit, never geological validation or training.

Only new CSV/TXT reports are written. --check-only writes nothing.
"""
import argparse
from collections import defaultdict
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
from audit_segmentation_dataset import ROOT, configured_paths
import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_details(flat):
    if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2:
        raise ValueError('Invalid polygon coordinate list')
    p = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite polygon coordinates')
    raw_count = len(p)
    closed = bool(np.array_equal(p[0], p[-1]))
    if closed:
        p = p[:-1]
    if len(p) < 3:
        raise ValueError('Fewer than three vertices after removing closure')
    area = float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))-np.dot(p[:, 1], np.roll(p[:, 0], -1)))/2)
    lo, hi = p.min(axis=0), p.max(axis=0)
    bbox_area = float(np.prod(hi-lo))
    corners = {(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])}
    edges = np.roll(p, -1, axis=0)-p
    axis_edges = bool(np.all(np.logical_xor(edges[:, 0] == 0, edges[:, 1] == 0)))
    rectangle = len(p) == 4 and bbox_area > 0 and set(map(tuple, p)) == corners and axis_edges
    # Canonical vertex sequence allows cyclic starts and reversed winding.
    vertices = list(map(tuple, p))
    forms = []
    for sequence in (vertices, vertices[::-1]):
        forms += [tuple(sequence[i:]+sequence[:i]) for i in range(len(sequence))]
    return dict(raw_vertices=raw_count, repeated_closing_vertex=closed, vertices=len(p),
                unique_vertices=len(set(vertices)), four_vertices=len(p) == 4, polygon_area=area,
                tight_bbox_area=bbox_area, axis_aligned_rectangle=bool(rectangle)), min(forms)


def verify(before):
    changed = [str(p) for p, h in before.items() if not p.is_file() or digest(p) != h]
    if changed:
        raise RuntimeError(f'Protected file discrepancy; no repairs attempted: {changed}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before, output = {}, None
    try:
        images_root, masks_root = configured_paths()
        for p in (ROOT/'create_masks.py', ROOT/'dataset_loader.py', ROOT/'tools/audit_segmentation_dataset.py'):
            before[p] = digest(p)
        rows, notes, failures = [], [], []
        for name in ('README.dataset.txt', 'README.roboflow.txt'):
            p = images_root/name
            if p.is_file():
                before[p] = digest(p)
                notes.append(f'RECORDED PROVENANCE — {p}\n'+p.read_text(encoding='utf-8-sig'))
        for split in ('train', 'valid'):
            source = images_root/split/'_annotations.coco.json'
            before[source] = digest(source)
            data = json.loads(source.read_text(encoding='utf-8-sig'))
            categories = {c['id']: c['name'] for c in data['categories']}
            ids = {key for key, name in categories.items() if name.lower() == 'landslide'}
            if not ids:
                raise ValueError(f'{split}: no landslide category found')
            images = {im['id']: im for im in data['images']}
            if len(images) != len(data['images']) or len({a['id'] for a in data['annotations']}) != len(data['annotations']):
                raise ValueError(f'{split}: duplicate image or annotation IDs')
            names = [im['file_name'] for im in images.values()]
            if len({Path(n).stem.casefold() for n in names}) != len(names):
                raise ValueError(f'{split}: ambiguous image-to-mask stems')
            grouped = defaultdict(list)
            for a in data['annotations']:
                if a['category_id'] in ids:
                    if a['image_id'] not in images:
                        raise ValueError('Annotation refers to missing image ID')
                    grouped[a['image_id']].append(a)
            count = rectangles = matches = duplicates = overlaps = 0
            for image_id, im in images.items():
                name = im['file_name']
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError(f'Unsafe filename: {name}')
                image_path = images_root/split/name
                mask_path = masks_root/split/(Path(name).stem+'_mask.png')
                for p in (image_path, mask_path):
                    before[p] = digest(p)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                existing = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                shape = (int(im['height']), int(im['width']))
                if image is None or existing is None or image.shape[:2] != shape or existing.shape != shape:
                    raise ValueError(f'Missing/unreadable/dimension-mismatched pair: {name}')
                if existing.dtype != np.uint8 or not set(np.unique(existing)) <= {0, 255}:
                    raise ValueError(f'Invalid reference mask values/format: {mask_path}')
                reconstructed = np.zeros(shape, dtype=np.uint8)
                ann_records = []
                for a in grouped[image_id]:
                    seg = a.get('segmentation')
                    if not isinstance(seg, list) or not seg:
                        raise ValueError(f"Unsupported/nonpolygon segmentation: {split} annotation {a['id']}")
                    details, keys = zip(*(polygon_details(poly) for poly in seg))
                    bbox = np.asarray(a['bbox'], dtype=float)
                    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] < 0):
                        raise ValueError('Invalid COCO bbox')
                    bbox_area = float(bbox[2]*bbox[3])
                    area = sum(d['polygon_area'] for d in details)
                    raster = np.zeros(shape, dtype=np.uint8)
                    for poly in seg:
                        points = np.round(np.asarray(poly, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
                        cv2.fillPoly(raster, [points], 255)
                    reconstructed |= raster
                    rect = all(d['axis_aligned_rectangle'] for d in details)
                    count += 1
                    rectangles += int(rect)
                    row = dict(record_type='annotation', split=split, filename=name, image_id=image_id,
                        annotation_id=a['id'], category_id=a['category_id'], category_name=categories[a['category_id']],
                        polygon_count=len(seg), raw_vertex_counts=json.dumps([d['raw_vertices'] for d in details]),
                        vertex_counts_excluding_closure=json.dumps([d['vertices'] for d in details]),
                        unique_vertex_counts=json.dumps([d['unique_vertices'] for d in details]),
                        four_vertices_per_polygon=json.dumps([d['four_vertices'] for d in details]),
                        all_polygons_four_vertices=all(d['four_vertices'] for d in details),
                        coco_bbox_xywh=json.dumps(bbox.tolist()), polygon_area_sum=area, bbox_area=bbox_area,
                        coco_recorded_area=a.get('area', 'N/A'), polygon_details=json.dumps(details),
                        area_approximately_bbox=bool(np.isclose(area, bbox_area, rtol=1e-4, atol=1e-3)),
                        all_polygons_axis_aligned_rectangles=rect, duplicate_annotation_ids=[], overlapping_annotation_ids=[],
                        overlap_raster_iou={}, reference_status='UNVERIFIED')
                    ann_records.append((row, tuple(sorted(keys)), raster))
                for i, (a, key_a, raster_a) in enumerate(ann_records):
                    for b, key_b, raster_b in ann_records[i+1:]:
                        if key_a == key_b:
                            duplicates += 1
                            a['duplicate_annotation_ids'].append(b['annotation_id'])
                            b['duplicate_annotation_ids'].append(a['annotation_id'])
                        intersection = int(np.count_nonzero(raster_a & raster_b))
                        if intersection:
                            overlaps += 1
                            iou = intersection/np.count_nonzero(raster_a | raster_b)
                            for first, second in ((a, b), (b, a)):
                                first['overlapping_annotation_ids'].append(second['annotation_id'])
                                first['overlap_raster_iou'][second['annotation_id']] = float(iou)
                mismatch = int(np.count_nonzero(existing != reconstructed))
                matches += int(mismatch == 0)
                if mismatch:
                    failures.append(f'{split}/{name}: {mismatch} mask pixels differ')
                for row, _, _ in ann_records:
                    for field in ('duplicate_annotation_ids', 'overlapping_annotation_ids', 'overlap_raster_iou'):
                        row[field] = json.dumps(row[field])
                    row['mask_mismatch_pixels'] = mismatch
                    rows.append(row)
                rows.append(dict(record_type='image_mask_check', split=split, filename=name, image_id=image_id,
                    landslide_annotation_count=len(ann_records), mask_path=str(mask_path), mask_mismatch_pixels=mismatch,
                    existing_foreground_percent=float(np.count_nonzero(existing)/existing.size*100),
                    image_sha256=before[image_path], mask_sha256=before[mask_path], reference_status='UNVERIFIED'))
            labeled = sum(bool(grouped[k]) for k in images)
            notes.append(f'{split}: landslide categories={[(k, categories[k]) for k in sorted(ids)]}\n'
                         f'Images={len(images)}; with landslide annotations={labeled}; without={len(images)-labeled}\n'
                         f'Annotations={count}; all parts axis-aligned rectangles={rectangles}\n'
                         f'Exact reconstructed-mask matches={matches}/{len(images)}; duplicate polygon-geometry pairs={duplicates}; raster-overlap pairs={overlaps}\n'
                         f'COCO info={json.dumps(data.get("info"))}; licenses={json.dumps(data.get("licenses"))}')
        verify(before)
        summary = '\n\n'.join([
            'DESCRIPTIVE ANNOTATION AUDIT — UNVERIFIED REFERENCES; TRAIN/VALID ONLY',
            'Polygon areas use the shoelace formula in original coordinate units. Repeated final closing vertices are removed for logical vertex counts.\n'
            'Four vertices alone does not establish a rectangle: four bounding-box corners and axis-aligned edges are checked directly.\n'
            'Area equality tolerance: relative 1e-4, absolute 1e-3. COCO bbox is [x,y,width,height].\n'
            'For multipart annotations, polygon areas are summed (not geometric union); per-part areas/counts are in CSV.\n'
            'Duplicates compare exact cyclic/reversed polygon geometry within each image. Overlap is measured on original-resolution rasters; touching edges may count.\n'
            'Mask comparison reproduces create_masks.py: float32 coordinates, np.round to int32, cv2.fillPoly(255), union by image. No masks are saved.\n'
            'Recorded README/COCO metadata describe provenance but do not independently establish labeling methodology or geological validity.\n'
            'Rectangular annotations are not automatically incorrect; empty references are not confirmed negatives.',
            *notes, 'DISCREPANCIES: '+ ('\n'.join(failures) if failures else 'None'),
            f'Protected-file SHA-256 verification passed for {len(before)} files. No checkpoint loaded; no training performed.',
        ])+'\n'
        buffer = io.StringIO(newline='')
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        if not args.check_only:
            parent = ROOT/'annotation_investigations'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            with (output/'annotation_audit.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                stream.write(buffer.getvalue())
            with (output/'summary.txt').open('x', encoding='utf-8') as stream:
                stream.write(summary)
        print(summary)
        print(f'Reports saved: {output}' if output else 'Checks complete; no reports written.')
        return 1 if failures else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 1
    finally:
        verify(before)


if __name__ == '__main__':
    raise SystemExit(main())
