"""Summarize saved train/valid annotation geometry; no image/mask/model reads.

Reads source audit CSV/TXT and COCO metadata for image dimensions.
--check-only creates no reports. All references remain UNVERIFIED.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_segmentation_dataset import ROOT, configured_paths

AUDIT = ROOT/'annotation_investigations/20261015T044234Z_9b9a9762ba9c4af2ae47098ecb655faf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return f'n={len(values)}, min={min(values):.6f}, median={statistics.median(values):.6f}, mean={statistics.mean(values):.6f}, max={max(values):.6f}' if values else 'N/A'


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    before = {}
    try:
        for p in (AUDIT/'annotation_audit.csv', AUDIT/'summary.txt', ROOT/'dataset_loader.py', ROOT/'tools/audit_segmentation_dataset.py'):
            before[p] = digest(p)
        prior_summary = (AUDIT/'summary.txt').read_text(encoding='utf-8')
        if 'DISCREPANCIES: None' not in prior_summary:
            raise ValueError('Source audit does not report a discrepancy-free result; inspect it first')
        with (AUDIT/'annotation_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        if any(r['split'] not in ('train', 'valid') for r in records):
            raise ValueError('Unexpected split in source audit')
        images_root, _ = configured_paths()
        metadata, annotations = {}, {}
        for split in ('train', 'valid'):
            p = images_root/split/'_annotations.coco.json'
            before[p] = digest(p)
            data = json.loads(p.read_text(encoding='utf-8-sig'))
            metadata[split] = {str(im['id']): im for im in data['images']}
            annotations[split] = {str(a['id']): a for a in data['annotations']}
        regions, image_rows, grouped = [], [], defaultdict(list)
        for r in records:
            if int(r['mask_mismatch_pixels']) != 0:
                raise ValueError('Source audit contains mask discrepancies')
            im = metadata[r['split']][r['image_id']]
            if im['file_name'] != r['filename']:
                raise ValueError('Image metadata no longer matches source audit')
            width, height = int(im['width']), int(im['height'])
            if width <= 0 or height <= 0:
                raise ValueError('Invalid image dimensions')
            if r['record_type'] == 'annotation':
                if r['all_polygons_axis_aligned_rectangles'] != 'True' or int(r['polygon_count']) != 1:
                    raise ValueError('Expected single rectangular polygons; cannot summarize changed geometry silently')
                recorded_bbox = json.loads(r['coco_bbox_xywh'])
                annotation = annotations[r['split']][r['annotation_id']]
                if str(annotation['image_id']) != r['image_id'] or annotation['bbox'] != recorded_bbox:
                    raise ValueError('Current COCO annotation differs from saved audit')
                flat = annotation['segmentation'][0]
                points = list(zip(flat[::2], flat[1::2]))
                x, y = min(p[0] for p in points), min(p[1] for p in points)
                w, h = max(p[0] for p in points)-x, max(p[1] for p in points)-y
                area = float(r['polygon_area_sum'])
                if abs(w*h-area) > max(.001, abs(area)*.0001):
                    raise ValueError('Current polygon dimensions disagree with audited polygon area')
                row = dict(split=r['split'], filename=r['filename'], image_id=r['image_id'],
                    annotation_id=r['annotation_id'], category_id=r['category_id'], category_name=r['category_name'],
                    image_width=width, image_height=height, rectangle_x=x, rectangle_y=y,
                    rectangle_width=w, rectangle_height=h, polygon_area=area, polygon_tight_bbox_area=w*h,
                    recorded_coco_bbox_xywh=r['coco_bbox_xywh'], recorded_coco_bbox_area=float(r['bbox_area']),
                    polygon_minus_recorded_bbox_area=area-float(r['bbox_area']),
                    audited_area_approximately_bbox=r['area_approximately_bbox'],
                    polygon_area_fraction=area/(width*height), raw_vertex_counts=r['raw_vertex_counts'],
                    vertices_excluding_closure=r['vertex_counts_excluding_closure'],
                    raster_overlap_annotation_ids=r['overlapping_annotation_ids'], reference_status='UNVERIFIED')
                regions.append(row)
                grouped[r['split'], r['image_id']].append(row)
            elif r['record_type'] == 'image_mask_check':
                image_rows.append(dict(split=r['split'], filename=r['filename'], image_id=r['image_id'],
                    annotation_count=int(r['landslide_annotation_count']), image_width=width, image_height=height,
                    raster_union_foreground_percent=float(r['existing_foreground_percent']), reference_status='UNVERIFIED'))
            else:
                raise ValueError('Unknown source audit record type')
        for row in image_rows:
            parts = grouped[row['split'], row['image_id']]
            if len(parts) != row['annotation_count']:
                raise ValueError('Annotation/image count mismatch')
            row['sum_polygon_area_fraction_not_union'] = sum(p['polygon_area_fraction'] for p in parts)
        lines = ['ORIGINAL LANDSLIDE ANNOTATION GEOMETRY — DESCRIPTIVE, UNVERIFIED',
            'Source audit: '+str(AUDIT),
            'Continuous COCO geometry: widths/heights and shoelace polygon areas in exported-image pixel coordinates.',
            'Rectangle dimensions come from original polygon bounds; recorded COCO bounding boxes are retained separately.',
            'Area fraction = polygon area/(image width*height). Repeated closing vertices do not add corners.',
            'Raster coverage is copied from the prior audit: union of rounded float32 polygons filled by OpenCV.',
            'Raster union coverage is not the sum of continuous polygon areas: rounding, inclusive boundary pixels,',
            'and overlaps differ. Per-image summed polygon fractions are explicitly NOT union coverage.',
            'Overlap lists below distinguish positive continuous rectangle intersection from raster-only contact.',
            'Rectangular annotations are not automatically incorrect. Geological validity remains unverified.',
            'Empty masks are not confirmed negatives. This report does not infer annotation intent or quality.',
            'No images, masks, checkpoints, or test split were opened. Saved raster results were not re-evaluated.', '']
        for split in ('train', 'valid'):
            rr = [r for r in regions if r['split'] == split]
            ii = [r for r in image_rows if r['split'] == split]
            expected = (191, 279) if split == 'train' else (42, 80)
            if (len(rr), len(ii)) != expected:
                raise ValueError(f'Unexpected {split} annotation/image counts')
            lines += [f'{split}: {len(rr)} single axis-aligned rectangular annotations; {len(ii)} images; '
                      f'{sum(r["annotation_count"] > 0 for r in ii)} annotated; {sum(r["annotation_count"] == 0 for r in ii)} unlabelled.',
                      'Categories: '+str(sorted({(r['category_id'], r['category_name']) for r in rr}))]
            for field in ('rectangle_width', 'rectangle_height', 'polygon_area', 'polygon_area_fraction'):
                lines.append(field+': '+stats([r[field] for r in rr]))
            lines.append(f"Polygon vs recorded bbox area outside prior audit tolerance (rtol=1e-4, atol=1e-3): {sum(r['audited_area_approximately_bbox'] == 'False' for r in rr)}. These are preserved, not repaired.")
            lines.append('Absolute polygon-minus-recorded-bbox area: '+stats([abs(r['polygon_minus_recorded_bbox_area']) for r in rr]))
            for label, subset in (('all images', ii), ('annotated images', [r for r in ii if r['annotation_count']])):
                lines.append(f'Raster union coverage % ({label}): '+stats([r['raster_union_foreground_percent'] for r in subset]))
            ordered = sorted(rr, key=lambda r: (r['polygon_area_fraction'], int(r['annotation_id'])))
            for label, selection in (('Smallest 3 regions', ordered[:3]), ('Largest 3 regions', list(reversed(ordered[-3:])))):
                lines.append(label+' (area fraction; annotation ID breaks ties):')
                lines += [f"  {r['filename']} | annotation {r['annotation_id']} | {r['rectangle_width']} x {r['rectangle_height']} | area={r['polygon_area']:.6f} | fraction={r['polygon_area_fraction']:.6f}" for r in selection]
            lines.append('Images with raster-overlapping annotations (unordered pairs, each listed once):')
            pairs = []
            index = {r['annotation_id']: r for r in rr}
            for a in rr:
                for bid in json.loads(a['raster_overlap_annotation_ids']):
                    if int(a['annotation_id']) >= int(bid):
                        continue
                    b = index[str(bid)]
                    if a['image_id'] != b['image_id']:
                        raise ValueError('Overlap references a different image')
                    dx = max(0, min(a['rectangle_x']+a['rectangle_width'], b['rectangle_x']+b['rectangle_width'])-max(a['rectangle_x'], b['rectangle_x']))
                    dy = max(0, min(a['rectangle_y']+a['rectangle_height'], b['rectangle_y']+b['rectangle_height'])-max(a['rectangle_y'], b['rectangle_y']))
                    pairs.append(f"  {a['filename']} | annotations {a['annotation_id']}, {bid} | continuous intersection area={dx*dy:.6f}"+(' (raster-only overlap/contact)' if dx*dy == 0 else ''))
            lines += pairs or ['  None']
            lines.append('')
        lines += ['SOURCE SHA-256 (checked unchanged before/after):']+[f'{p}: {h}' for p, h in before.items()]
        summary = '\n'.join(lines)+'\n'
        if not args.check_only:
            parent = ROOT/'annotation_geometry_reports'
            if parent.resolve() != parent:
                raise ValueError('Redirected output directory refused')
            parent.mkdir(exist_ok=True)
            output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
            output.mkdir(exist_ok=False)
            for name, text in (('summary.txt', summary), ('rectangle_geometry.csv', csv_text(regions)), ('image_coverage.csv', csv_text(image_rows))):
                with (output/name).open('x', encoding='utf-8-sig', newline='') as stream:
                    stream.write(text)
            print(f'Reports saved: {output}')
        print(summary)
        if args.check_only:
            print('Checks passed; no reports written.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 1
    finally:
        changed = [str(p) for p, h in before.items() if not p.is_file() or digest(p) != h]
        if changed:
            raise RuntimeError(f'Protected files changed; no repairs attempted: {changed}')


if __name__ == '__main__':
    raise SystemExit(main())
