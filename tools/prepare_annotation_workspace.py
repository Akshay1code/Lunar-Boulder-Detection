"""Build offline manual-review workspaces from the existing five-image pilot.

No automatic boundaries, model inference, or training. Source assets read-only.
"""
import base64
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_segmentation_dataset import ROOT, configured_paths
from PIL import Image

MANIFEST = ROOT/'annotation_pilot/20261016T130040Z_acf2a154f6df4a7e9c6f6f31d4625559/pilot_manifest.json'

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>@@ID@@ — UNREVIEWED workspace</title>
<style>body{font:16px Arial;max-width:1400px;margin:24px auto;padding:16px;background:#f5f5f5} .row{display:flex;gap:20px}figure{margin:0;flex:1;min-width:0}img{width:100%;max-width:640px}canvas{width:100%;max-width:640px;touch-action:none;border:2px solid #777;image-rendering:pixelated}label{display:block;margin:16px 0}textarea{display:block;width:95%;min-height:80px}button,input,select{font:inherit;margin:6px;padding:6px}.warning{background:#fff1c4;padding:12px}h1{font-size:24px}</style>
<h1>@@ID@@ — UNREVIEWED manual annotation draft</h1><p>@@NAME@@ | TRAIN | COCO ID @@COCO@@</p>
<p class="warning">Existing reference: UNVERIFIED. New draft: UNREVIEWED. Empty masks do not establish absence of landslides.
Do not identify shadows, craters, or dark terrain as landslides without supporting evidence. Draw only a proposal you can explain; uncertainty may remain unresolved.</p>
<div class="row"><figure><figcaption>Original image</figcaption><img src="@@IMAGE@@" alt="Original training image"></figure>
<figure><figcaption>Existing reference mask — UNVERIFIED, read-only</figcaption><img src="@@MASK@@" alt="Unverified reference mask"></figure></div>
<h2>Editable mask canvas — UNREVIEWED</h2>
<p>Initially blank. Black means unmarked, NOT a confirmed negative. White means a manual proposal, NOT confirmed ground truth.
Canvas resolution: @@WIDTH@@ × @@HEIGHT@@, matching the original image. No reference boundaries are copied.</p>
<label>Tool <select id="tool"><option value="white">Draw proposal</option><option value="black">Erase proposal</option></select>
Brush diameter (original pixels) <input id="brush" type="number" min="1" max="100" value="8"></label>
<button id="undo">Undo last stroke</button><button id="clear">Clear draft canvas</button><br>
<canvas id="canvas" width="@@WIDTH@@" height="@@HEIGHT@@" aria-label="UNREVIEWED editable draft mask"></canvas>
<label>Reviewer<input id="reviewer" placeholder="Not assigned"></label>
<label>Visual evidence<textarea id="evidence" placeholder="Separate observations from interpretations; cite supporting evidence."></textarea></label>
<label>Uncertainty<textarea id="uncertainty" placeholder="Describe uncertain regions and alternative explanations."></textarea></label>
<label>Notes<textarea id="notes" placeholder="Boundary rationale, limitations, or additional review needed."></textarea></label>
<p class="warning">Edits are held in browser memory only. Save a draft before closing. Downloads do not update this workspace or original assets.
Use a new filename and do not replace an existing dataset mask. Status remains UNREVIEWED after export.</p>
<button id="save">Download UNREVIEWED draft (mask + fields as JSON)</button>
<button id="png">Download UNREVIEWED mask PNG</button>
<script>
const meta=@@META@@;
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d',{willReadFrequently:true});
ctx.fillStyle='black';ctx.fillRect(0,0,canvas.width,canvas.height);
let active=false,last=null,undo=[],dirty=false;
function snapshot(){undo.push(ctx.getImageData(0,0,canvas.width,canvas.height));if(undo.length>15)undo.shift();}
function point(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height];}
function paint(p){let size=Math.max(1,Math.min(100,Number(document.getElementById('brush').value)||8));
ctx.strokeStyle=ctx.fillStyle=document.getElementById('tool').value;ctx.lineWidth=size;ctx.lineCap='round';
ctx.beginPath();ctx.moveTo(...last);ctx.lineTo(...p);ctx.stroke();ctx.beginPath();ctx.arc(...p,size/2,0,Math.PI*2);ctx.fill();last=p;dirty=true;}
canvas.onpointerdown=e=>{if(e.button!==0)return;snapshot();active=true;last=point(e);canvas.setPointerCapture(e.pointerId);paint(last);};
canvas.onpointermove=e=>{if(active)paint(point(e));};
canvas.onpointerup=canvas.onpointercancel=()=>{active=false;};
document.getElementById('undo').onclick=()=>{if(undo.length){ctx.putImageData(undo.pop(),0,0);dirty=true;}};
document.getElementById('clear').onclick=()=>{if(confirm('Clear only this UNREVIEWED draft canvas?')){snapshot();ctx.fillStyle='black';ctx.fillRect(0,0,canvas.width,canvas.height);dirty=true;}};
// Threshold manually drawn antialiased brush edges only; never infer boundaries from the image.
function binaryCanvas(){const out=document.createElement('canvas');out.width=canvas.width;out.height=canvas.height;const c=out.getContext('2d');const d=ctx.getImageData(0,0,canvas.width,canvas.height);
for(let i=0;i<d.data.length;i+=4){let v=d.data[i]>=128?255:0;d.data[i]=d.data[i+1]=d.data[i+2]=v;d.data[i+3]=255;}c.putImageData(d,0,0);return out;}
function download(blob,suffix){const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=meta.review_id+'_UNREVIEWED_'+Date.now()+'_'+crypto.getRandomValues(new Uint32Array(1))[0]+suffix;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
document.getElementById('save').onclick=()=>{const fields={};for(const id of ['reviewer','evidence','uncertainty','notes'])fields[id]=document.getElementById(id).value;
download(new Blob([JSON.stringify({...meta,status:'UNREVIEWED',fields,mask_png_data_url:binaryCanvas().toDataURL('image/png')},null,2)],{type:'application/json'}),'.json');dirty=false;};
document.getElementById('png').onclick=()=>binaryCanvas().toBlob(b=>download(b,'.png'),'image/png');
for(const id of ['reviewer','evidence','uncertainty','notes'])document.getElementById(id).oninput=()=>dirty=true;
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue='';}};
</script></html>'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_uri(path):
    with Image.open(path) as source:
        source.load()
        buffer = io.BytesIO()
        source.save(buffer, format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'), source.size


def main():
    before = {}
    try:
        images_root, masks_root = configured_paths()
        for path in (MANIFEST, ROOT/'dataset_loader.py', ROOT/'tools/audit_segmentation_dataset.py'):
            before[path] = digest(path)
        pilot = json.loads(MANIFEST.read_text(encoding='utf-8'))
        records = pilot['images']
        if len(records)!=5 or len({r['image_id'] for r in records})!=5:
            raise ValueError('Expected five distinct pilot images')
        pages=[]
        for number,r in enumerate(records,1):
            if r['split']!='train' or Path(r['filename']).name!=r['filename']:
                raise ValueError('Pilot contains a nontraining or unsafe record')
            image_path=(images_root/'train'/r['filename']).resolve()
            mask_path=(masks_root/'train'/(Path(r['filename']).stem+'_mask.png')).resolve()
            if image_path.parent!=(images_root/'train').resolve() or mask_path.parent!=(masks_root/'train').resolve():
                raise ValueError('Redirected asset outside training directory')
            if Path(r['image_path'])!=image_path or Path(r['mask_path'])!=mask_path:
                raise ValueError('Pilot path association mismatch')
            for path,key in ((image_path,'image_sha256'),(mask_path,'mask_sha256')):
                before[path]=digest(path)
                if before[path]!=r[key]:
                    raise ValueError(f'Pilot hash mismatch: {path}')
            original,size=png_uri(image_path)
            mask,mask_size=png_uri(mask_path)
            if size!=mask_size:
                raise ValueError('Image/mask dimension mismatch')
            meta=dict(r,status='UNREVIEWED',source_manifest=str(MANIFEST),source_manifest_sha256=before[MANIFEST],
                      canvas_width=size[0],canvas_height=size[1],canvas_initial_state='blank_unmarked_not_negative')
            replacements={'ID':html.escape(r['review_id']),'NAME':html.escape(r['filename']),'COCO':str(r['image_id']),
                          'IMAGE':original,'MASK':mask,'WIDTH':str(size[0]),'HEIGHT':str(size[1]),
                          'META':json.dumps(meta).replace('<','\\u003c')}
            page=PAGE
            for key,value in replacements.items():
                page=page.replace('@@'+key+'@@',value)
            pages.append((f'image_{number:02d}',page,meta))
        parent=ROOT/'annotation_workspaces'
        if parent.resolve()!=parent:
            raise ValueError('Redirected output directory refused')
        parent.mkdir(exist_ok=True)
        output=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
        output.mkdir(exist_ok=False)
        links=[]
        for folder,page,meta in pages:
            directory=output/folder
            directory.mkdir(exist_ok=False)
            with (directory/'workspace.html').open('x',encoding='utf-8') as stream: stream.write(page)
            with (directory/'provenance.json').open('x',encoding='utf-8') as stream: json.dump(meta,stream,indent=2)
            links.append(f'<li><a href="{folder}/workspace.html">{html.escape(meta["review_id"])} — COCO {meta["image_id"]} — UNREVIEWED</a></li>')
        with (output/'index.html').open('x',encoding='utf-8') as stream:
            stream.write('<!doctype html><meta charset="utf-8"><h1>Manual annotation workspaces — UNREVIEWED</h1><p>Open an image workspace. Draft edits are in browser memory; export JSON to retain mask and review fields. Downloads go to your browser download location. Existing references are UNVERIFIED. No automated boundaries were created.</p><ul>'+''.join(links)+'</ul>')
        print(f'Created five workspaces: {output / "index.html"}')
        print('All canvases start blank; no mask PNGs generated. Existing files preserved.')
    finally:
        changed=[str(p) for p,h in before.items() if not p.is_file() or digest(p)!=h]
        if changed: raise RuntimeError(f'Protected source files changed: {changed}')


if __name__=='__main__':
    main()
