"""Local manual SPACE review. No training, predictions, or original-file writes."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_loader import DATASET_ROOT
import cv2
import numpy as np
from PIL import Image

OUTPUT = ROOT/'space_annotation_reviews'
FORMAT = 'manual-space-review-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex


def write_json(p, data):
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format='PNG')
    return b.getvalue()


def uri(a):
    return 'data:image/png;base64,'+base64.b64encode(png(a)).decode()


def select():
    """Fixed dataset-only cases: known dark boundaries, irregular labels, terrain."""
    picks = {
        'train': [183, 136, 181, 271, 64],
        'valid': [73, 36,
                  'ohr_000_patch_0821_png.rf.819fe103479c93f50c0483739d6a840d.jpg',
                  'ohr_004_patch_0901_png.rf.cb53facbf3a9f1330c4e3c91f0ef19bd.jpg',
                  'ohr_008_patch_0229_png.rf.7dfe7a33d06c5844f9cb84002e164236.jpg']}
    records = []
    for split, targets in picks.items():
        cp = DATASET_ROOT/split/'_annotations.coco.json'
        data = json.loads(cp.read_text())
        assert next(c['name'] for c in data['categories'] if c['id'] == 4).lower() == 'space'
        for target in targets:
            matches = [i for i in data['images'] if i['id'] == target or i['file_name'] == target]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous/missing selection: {split}/{target}')
            im = matches[0]
            ip = (DATASET_ROOT/split/im['file_name']).resolve()
            if not ip.is_relative_to((DATASET_ROOT/split).resolve()):
                raise ValueError('Image path escapes selected split')
            aa = [a for a in data['annotations'] if a['image_id'] == im['id'] and a['category_id'] == 4]
            records.append(dict(key=f"{split}/{im['id']}", split=split, coco_id=im['id'],
                filename=im['file_name'], image=str(ip), image_sha256=sha(ip),
                coco=str(cp), coco_sha256=sha(cp), width=im['width'], height=im['height'],
                annotation_ids=[a['id'] for a in aa],
                polygons=[p for a in aa for p in a['segmentation']],
                selection_reason=('Irregular space polygons; inspect shadows' if target in (136, 181)
                    else 'Dark-region/space-boundary review' if aa else 'Terrain control; no space label is not proof of absence'),
                uncertainty='No verified horizon, geology, space completeness or shadow classification.',
                review_status='unreviewed'))
    assert len(records) == 10 and len({r['key'] for r in records}) == 10
    return records


class Workspace:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(OUTPUT.resolve()):
            raise ValueError('Workspace must be under space_annotation_reviews')
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if self.manifest['format'] != FORMAT or len(self.manifest['records']) != 10:
            raise ValueError('Invalid workspace manifest')
        self.records = self.manifest['records']

    def record(self, index):
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError('Invalid image index')
        return self.records[index]

    def revisions(self, index):
        return sorted(self.path.glob(f'image_{index:02d}_*.json'))

    def arrays(self, index):
        r = self.record(index)
        for path, expected in [(r['image'], r['image_sha256']), (r['coco'], r['coco_sha256'])]:
            if sha(Path(path)) != expected:
                raise ValueError('Source hash mismatch; review stopped, no automatic repair')
        bgr = cv2.imread(r['image'])
        if bgr is None or bgr.shape[:2] != (r['height'], r['width']):
            raise ValueError('Unreadable image or dimension mismatch')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ref = np.zeros(rgb.shape[:2], np.uint8)
        for polygon in r['polygons']:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError('Invalid COCO polygon')
            cv2.fillPoly(ref, [np.rint(np.array(polygon, np.float32).reshape(-1, 2)).astype(np.int32)], 1)
        labels = np.full(ref.shape, 255, np.uint8)
        labels[ref > 0] = 1  # Existing unverified proposal; outside remains UNKNOWN.
        return rgb, ref, labels

    def load(self, index, original=False):
        r = self.record(index)
        rgb, ref, labels = self.arrays(index)
        metadata = dict(reviewer='', notes='', evidence='', status='unreviewed')
        revisions = self.revisions(index)
        if revisions and not original:
            metadata = json.loads(revisions[-1].read_text())
            mask_path = self.path/metadata['mask_file']
            if mask_path.parent != self.path or sha(mask_path) != metadata['mask_sha256']:
                raise ValueError('Saved revision hash/path mismatch')
            with Image.open(mask_path) as im:
                labels = np.array(im)
            if labels.shape != ref.shape or not set(np.unique(labels)) <= {0, 1, 255}:
                raise ValueError('Invalid saved labels')
        overlay = rgb.copy()
        overlay[ref > 0] = (.65*rgb[ref > 0]+.35*np.array([0, 220, 255])).astype(np.uint8)
        return dict(record=r, original=uri(rgb), reference=uri(overlay),
                    labels=labels.ravel().tolist(), metadata=metadata,
                    revision_count=len(revisions))

    def summary(self):
        rows = []
        for i, r in enumerate(self.records):
            revisions = self.revisions(i)
            latest = json.loads(revisions[-1].read_text()) if revisions else {}
            rows.append(dict(index=i, key=r['key'], filename=r['filename'],
                             status=latest.get('status', 'unreviewed'),
                             revisions=len(revisions), pixel_counts=latest.get('pixel_counts'),
                             last_revision=revisions[-1].name if revisions else None))
        return rows

    def save(self, data):
        index = data['index']
        r = self.record(index)
        self.arrays(index)  # verify original hashes before writing
        if data.get('status') not in {'unreviewed', 'in_progress', 'reviewed'}:
            raise ValueError('Invalid review status')
        values = data['labels']
        if not isinstance(values, list) or len(values) != r['width']*r['height']:
            raise ValueError('Mask dimensions do not match original')
        if any(type(x) is not int or x not in (0, 1, 255) for x in values):
            raise ValueError('Allowed labels: 0 non-space, 1 space, 255 ignore')
        for key in ('reviewer', 'notes', 'evidence'):
            if not isinstance(data.get(key), str) or len(data[key]) > 10000:
                raise ValueError(f'Invalid {key}')
        labels = np.array(values, np.uint8).reshape(r['height'], r['width'])
        name = f'image_{index:02d}_{stamp()}'
        mask_path = self.path/(name+'.png')
        with mask_path.open('xb') as f:
            f.write(png(labels))
        metadata = {k: data[k] for k in ('status', 'reviewer', 'notes', 'evidence')}
        metadata.update(key=r['key'], mask_file=mask_path.name, mask_sha256=sha(mask_path),
            image_sha256=r['image_sha256'], coco_sha256=r['coco_sha256'],
            pixel_counts={str(k): int(np.count_nonzero(labels == k)) for k in (0, 1, 255)},
            interpretation='Manual proposal only; reviewed status is not geological certification.')
        write_json(self.path/(name+'.json'), metadata)
        write_json(self.path/('summary_'+stamp()+'.json'), self.summary())
        return dict(saved=name, directory=str(self.path), summary=self.summary())


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Manual space review</title>
<style>body{font:16px system-ui;margin:24px;background:#111b29;color:#edf3ff}
button,input,select,textarea{font:inherit;margin:5px;padding:7px}button{cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
canvas,img{width:100%;image-rendering:auto}canvas{touch-action:none;cursor:crosshair}
textarea{width:90%;height:55px}.notice{background:#3c3020;padding:14px}pre{white-space:pre-wrap}
@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Manual space annotation review</h1>
<p class="notice">Existing polygons and edited masks are UNVERIFIED proposals.
Darkness does not establish space. Use IGNORE for uncertainty. No training or geological classification is performed.
0 = manually reviewed non-space; 1 = proposed space; 255 = ignore/unknown.
Outside existing polygons starts as IGNORE, not background.</p>
<button id="prev">Previous</button><select id="images"></select><button id="next">Next</button>
<button id="load">Load latest saved</button><button id="reset">Load original proposal</button>
<p id="info"></p><div class="grid">
<div>Original image<img id="original"></div>
<div>Existing SPACE polygons — UNVERIFIED<img id="reference"></div>
<div>Editable proposal over image<canvas id="canvas"></canvas></div></div>
<label>Brush <select id="brushLabel"><option value="1">Space (cyan)</option><option value="0">Reviewed non-space (green)</option><option value="255">Ignore / uncertain (magenta)</option></select></label>
<label>Radius, source pixels <input id="radius" type="number" min="1" max="100" value="10"></label>
<button id="undo">Undo stroke</button><button id="ignore">Mark all uncertain</button>
<label><input id="showIgnore" type="checkbox" checked>Show ignored pixels</label>
<p>Draw on the right panel. Use small brushes near boundaries. All-uncertain clears a proposal safely.</p>
<label>Reviewer <input id="reviewer"></label>
<label>Review status <select id="status"><option>unreviewed</option><option>in_progress</option><option>reviewed</option></select></label>
<p>“Reviewed” records your decision only; it does not certify geological correctness.</p>
Evidence / rationale<textarea id="evidence"></textarea>Uncertainty / notes<textarea id="notes"></textarea>
<button id="save">Save NEW revision</button><p id="message" role="status"></p>
<details><summary>Review summary / workspace</summary><pre id="summary"></pre></details>
<script>
const $=id=>document.getElementById(id);let idx=0,labels,w,h,bg,dirty=false,history=[],down=false,last=null,busy=false;
const canvas=$('canvas'),ctx=canvas.getContext('2d');
async function api(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
const d=await r.json();if(!r.ok)throw Error(d.error);return d;}
function message(s){$('message').textContent=s;}
function confirmDiscard(){return !dirty||confirm('Discard unsaved changes? Save a revision first to keep them.');}
function draw(){if(!bg)return;ctx.drawImage(bg,0,0,w,h);let im=ctx.getImageData(0,0,w,h);
for(let i=0;i<labels.length;i++){const v=labels[i];if(v===255&&!$('showIgnore').checked)continue;
const c=v===1?[0,220,255]:v===0?[0,210,90]:[210,40,200];for(let k=0;k<3;k++)im.data[4*i+k]=Math.round(.72*im.data[4*i+k]+.28*c[k]);}ctx.putImageData(im,0,0);}
async function load(i,reset=false){if(busy)return;if(!confirmDiscard()){ $('images').value=idx;return;}busy=true;message('Loading...');
try{const d=await api('/image?index='+i+'&original='+(reset?'1':'0'));idx=i;w=d.record.width;h=d.record.height;
labels=new Uint8Array(d.labels);canvas.width=w;canvas.height=h;history=[];dirty=false;
$('images').value=idx;$('original').src=d.original;$('reference').src=d.reference;
bg=new Image();bg.src=d.original;await bg.decode();draw();
for(const key of ['reviewer','evidence','notes','status'])$(key).value=d.metadata[key]|| (key==='status'?'unreviewed':'');
$('info').textContent=d.record.key+' | '+d.record.filename+' | '+d.record.selection_reason+' | '+d.record.uncertainty;
message('Loaded '+(reset?'original unverified proposal':'latest saved revision, or original if none')+'. '+d.revision_count+' saved revisions.');
}catch(e){message('ERROR: '+e.message);}finally{busy=false;}}
function point(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*w/r.width,(e.clientY-r.top)*h/r.height];}
function paint(x,y){let r=Math.min(100,Math.max(1,Number($('radius').value)||10)),v=Number($('brushLabel').value);
for(let yy=Math.max(0,Math.floor(y-r));yy<Math.min(h,Math.ceil(y+r));yy++)for(let xx=Math.max(0,Math.floor(x-r));xx<Math.min(w,Math.ceil(x+r));xx++)
if((xx-x)**2+(yy-y)**2<=r*r)labels[yy*w+xx]=v;dirty=true;}
canvas.onpointerdown=e=>{if(busy||!labels)return;history.push(labels.slice());if(history.length>12)history.shift();down=true;last=point(e);paint(...last);draw();canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!down||busy)return;const p=point(e),n=Math.max(1,Math.ceil(Math.hypot(p[0]-last[0],p[1]-last[1])));
for(let k=1;k<=n;k++)paint(last[0]+(p[0]-last[0])*k/n,last[1]+(p[1]-last[1])*k/n);last=p;draw();};
canvas.onpointerup=()=>{down=false;last=null;};canvas.onpointercancel=canvas.onpointerup;
$('undo').onclick=()=>{if(!busy&&history.length){labels=history.pop();dirty=true;draw();}};
$('ignore').onclick=()=>{if(!busy&&labels&&confirm('Set every pixel to uncertain?')){history.push(labels.slice());if(history.length>12)history.shift();labels.fill(255);dirty=true;draw();}};
$('showIgnore').onchange=draw;
for(const key of ['reviewer','notes','evidence','status'])$(key).oninput=()=>dirty=true;
$('prev').onclick=()=>load(Math.max(0,idx-1));$('next').onclick=()=>load(Math.min(9,idx+1));
$('images').onchange=()=>load(Number($('images').value));$('load').onclick=()=>load(idx);$('reset').onclick=()=>load(idx,true);
$('save').onclick=async()=>{if(busy||!labels)return;busy=true;try{
const d=await api('/save',{index:idx,labels:Array.from(labels),status:$('status').value,reviewer:$('reviewer').value,evidence:$('evidence').value,notes:$('notes').value});
dirty=false;message('Saved '+d.saved+' in '+d.directory);$('summary').textContent=JSON.stringify(d,null,2);
}catch(e){message('SAVE FAILED: '+e.message);}finally{busy=false;}};
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue='';}};
(async()=>{try{const d=await api('/summary');$('summary').textContent=JSON.stringify(d,null,2);for(const r of d.rows){const o=document.createElement('option');o.value=r.index;o.textContent=(r.index+1)+': '+r.key;$('images').appendChild(o);}await load(0);}catch(e){message('ERROR: '+e.message);}})();
</script>'''


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, code=200):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        p = urlparse(self.path)
        try:
            if p.path == '/':
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p.path == '/summary':
                self.reply(dict(workspace=str(self.server.workspace.path), rows=self.server.workspace.summary()))
            elif p.path == '/image':
                q = parse_qs(p.query)
                self.reply(self.server.workspace.load(int(q['index'][0]), q.get('original', ['0'])[0] == '1'))
            else:
                self.reply(dict(error='Not found'), 404)
        except (ValueError, OSError, KeyError, IndexError) as e:
            self.reply(dict(error=str(e)), 400)

    def do_POST(self):
        try:
            if self.path != '/save' or self.headers.get('Content-Type') != 'application/json':
                raise ValueError('Invalid request')
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 8*1024*1024:
                raise ValueError('Invalid request size')
            self.reply(self.server.workspace.save(json.loads(self.rfile.read(size))))
        except (ValueError, OSError, KeyError, TypeError) as e:
            self.reply(dict(error=str(e)), 400)


def create(path):
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'manifest.json', dict(format=FORMAT, records=select(),
        labels={'0': 'reviewed non-space', '1': 'proposed space', '255': 'ignore/uncertain'},
        note='No verified horizon positions claimed; representative annotation/terrain review only.'))
    ws = Workspace(path)
    write_json(path/'summary_initial.json', ws.summary())
    return ws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, help='Resume the exact workspace printed on first launch')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        import tempfile
        OUTPUT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='self_test_', dir=OUTPUT) as tmp:
            ws = create(Path(tmp)/'workspace')
            for i in range(10):
                d = ws.load(i)
                assert len(d['labels']) == d['record']['width']*d['record']['height']
                assert set(d['labels']) <= {1, 255}
            d = ws.load(0)
            labels = d['labels']
            labels[0:3] = [0, 1, 255]
            payload = dict(index=0, labels=labels, status='unreviewed', reviewer='TEST ONLY', notes='test', evidence='')
            ws.save(payload)
            assert ws.load(0)['labels'] == labels
            ws.save(payload)
            assert len(ws.revisions(0)) == 2
            payload['labels'] = [2]*len(labels)
            try:
                ws.save(payload)
                raise AssertionError('Invalid label accepted')
            except ValueError:
                pass
            assert ws.summary()[1]['status'] == 'unreviewed'
            import threading
            from urllib.request import urlopen, Request
            from urllib.error import HTTPError
            from html.parser import HTMLParser
            import re
            class IDs(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.ids = []
                def handle_starttag(self, tag, attrs):
                    self.ids.extend(v for k, v in attrs if k == 'id')
            ids = IDs()
            ids.feed(PAGE)
            assert len(ids.ids) == len(set(ids.ids))
            assert set(re.findall(r"\$\('([^']+)'\)", PAGE)) <= set(ids.ids)
            server = HTTPServer(('127.0.0.1', 0), Handler)
            server.workspace = ws
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base) as response:
                    assert b'Editable proposal' in response.read()
                with urlopen(base+'/image?index=0') as response:
                    assert json.load(response)['labels'] == labels
                payload['labels'] = labels
                with urlopen(Request(base+'/save', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})) as response:
                    assert json.load(response)['summary'][0]['revisions'] == 3
                try:
                    urlopen(base+'/image?index=10')
                    raise AssertionError('Invalid index accepted')
                except HTTPError as exc:
                    assert exc.code == 400
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
            for r in ws.records:
                assert sha(Path(r['image'])) == r['image_sha256']
                assert sha(Path(r['coco'])) == r['coco_sha256']
            print('PASS: all 10 originals/annotations load with matching hashes and dimensions; '
                  '0/1/255 PNG save/load; unique revisions; invalid labels rejected; '
                  'HTTP page/load/save/navigation errors; DOM IDs; source hashes unchanged.')
        return
    ws = Workspace(args.workspace) if args.workspace else create(OUTPUT/stamp())
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.workspace = ws
    print(f'Open http://127.0.0.1:{args.port}\nWorkspace: {ws.path}\n'
          f'Resume: .\\.venv311\\Scripts\\python.exe -B tools\\review_space_annotations.py --workspace "{ws.path}"', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
