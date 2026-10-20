"""Local CPU U-Net upload demo. No training, dataset scanning, or disk outputs.

Launch: python -B demo.py. Test: python -B demo.py --self-test.
"""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tools'))
from review_landslide_candidates import preprocess
from preview_unet_predictions import load_model
import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_CHECKPOINT = ROOT/'experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth'
CHECKPOINT = Path(os.environ.get('LUNAR_CHECKPOINT_PATH', DEFAULT_CHECKPOINT))
THRESHOLD = 0.5
MAX_BYTES = 15*1024*1024
PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lunar segmentation prototype</title><style>
a{color:#9dcaff;display:inline-block;padding:12px;border:1px solid #9dcaff;border-radius:8px;text-decoration:none}body{font:17px system-ui;margin:0;background:#101723;color:#edf3ff}main{max-width:1400px;margin:auto;padding:32px}h1{margin-bottom:8px}.notice{padding:16px;border-left:4px solid #efbc65;background:#252b36;line-height:1.6}.controls{padding:24px 0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}button{padding:12px 24px;background:#9dcaff;color:#10213a;border:0;border-radius:8px;font:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}figure{margin:0;background:#1c2736;padding:12px;border-radius:10px}img{width:100%;display:block;object-fit:contain}figcaption{padding:8px 0}#mask{image-rendering:pixelated}#status{min-height:28px}small{color:#b9c8dd}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style>
<main><h1>Lunar segmentation prototype</h1><p>Experiment I · existing U-Net · CPU inference · threshold 0.5</p>
<p class="notice"><strong>Experimental predictions — not geologically verified.</strong><br>Trained on unverified rectangular reference annotations. Highlighted pixels are not confirmed landslides. Foreground coverage is not an accuracy score. <strong>Not suitable for certifying terrain safety.</strong></p>
<div class="controls"><label>Upload a lunar image <input id="file" type="file" accept="image/png,image/jpeg,image/bmp,image/tiff"></label><button id="run">Generate prediction</button></div>
<div class="controls"><label>Prediction threshold <input id="threshold" type="range" min="0.20" max="0.80" step="0.01" value="0.50" disabled> <output id="thresholdValue">0.50</output></label></div>
<small>Use a validation or external demo image, not test-split data. Uploads stay in memory; downloads are saved only when you choose. Maximum 15 MB and 16 million pixels.</small>
<p id="status" role="status" aria-live="polite">Choose an image to begin.</p>
<section id="results" hidden><h2 id="metrics"></h2><p id="predictionWarning" class="notice" role="alert" hidden></p><p><strong>Horizon/space scenes are not reliably supported.</strong> Dark pixels may be space or shadowed terrain; no automatic background removal is performed. Small predicted coverage does not mean no landslide.</p><p id="filename"></p><div class="grid">
<figure><figcaption>Original image</figcaption><img id="original" alt="Original uploaded lunar image"></figure>
<figure><figcaption id="maskCaption">Predicted mask · threshold 0.5</figcaption><img id="mask" alt="Experimental binary prediction"></figure>
<figure><figcaption>Experimental prediction overlay · red</figcaption><img id="overlay" alt="Red experimental prediction over the original image"></figure></div>
<p id="regions"></p><details><summary>Optional visual terrain appearance (not geology)</summary><p id="appearance"></p><p id="appearanceMethod"></p><p id="appearanceLimit"></p><p>Small predicted coverage does not establish absence of a landslide.</p></details><p><a id="downloadMask" download="experimental_mask_256.png">Download binary mask (256 x 256)</a> &nbsp; <a id="downloadOverlay" download="experimental_overlay.png">Download overlay</a></p><p>Percentage of model-output pixels above the threshold; not a measured physical landslide area.</p><p id="checkpoint"></p><small>Inference uses the existing 256×256 preprocessing. Mask and overlay are enlarged for display using nearest-neighbor interpolation.</small></section></main>
<script>
const el=id=>document.getElementById(id), button=el('run'), status=el('status'), slider=el('threshold');
let token=null, generation=0, requestNumber=0, timer=null;
function clearResults(){
  generation++; requestNumber++; token=null; clearTimeout(timer);
  el('results').hidden=true; slider.disabled=true; slider.value='0.50'; el('thresholdValue').textContent='0.50';
  for(const key of ['original','mask','overlay'])el(key).removeAttribute('src');
  for(const key of ['downloadMask','downloadOverlay'])el(key).removeAttribute('href');
  el('filename').textContent=''; el('metrics').textContent=''; el('checkpoint').textContent='';
}
el('file').onchange=()=>{clearResults();status.textContent='Choose Generate prediction for the selected image.';};
function show(result){
  for(const key of ['original','mask','overlay'])el(key).src=result[key];
  el('metrics').textContent='Predicted pixel coverage: '+result.coverage.toFixed(3)+'% | Threshold: '+result.threshold.toFixed(2);
  el('maskCaption').textContent='Predicted mask - threshold '+result.threshold.toFixed(2);
  el('checkpoint').textContent='Experiment I checkpoint: '+result.checkpoint+' ('+result.position+')';
  el('downloadMask').href=result.raw_mask; el('downloadOverlay').href=result.overlay;
  el('downloadMask').download='experimental_mask_t'+result.threshold.toFixed(2)+'.png';
  el('downloadOverlay').download='experimental_overlay_t'+result.threshold.toFixed(2)+'.png';
  el('regions').textContent='Connected predicted regions: '+result.regions.count+'. Largest sizes (up to 10): '+(result.regions.areas_pixels.slice(0,10).join(', ')||'none')+' output pixels. '+result.regions.definition;
  el('appearance').textContent=result.appearance.label+' | edge density '+(100*result.appearance.edge_fraction).toFixed(2)+'%';
  el('appearanceMethod').textContent=result.appearance.method;
  el('appearanceLimit').textContent=result.appearance.limitation;
  el('predictionWarning').textContent=result.warning;
  el('predictionWarning').hidden=!result.warning;
  el('results').hidden=false;
}
async function post(path,body,type){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':type},body});
  const result=await response.json();if(!response.ok)throw Error(result.error);return result;
}
button.onclick=async()=>{
  const file=el('file').files[0];clearResults();
  if(!file){status.textContent='Please choose an image first.';return;}
  if(file.size>15*1024*1024){status.textContent='Image exceeds 15 MB.';return;}
  const current=generation;button.disabled=true;status.textContent='Running CPU inference...';
  try{
    const result=await post('/predict',file,'application/octet-stream');
    if(current!==generation)return;
    token=result.token;show(result);el('filename').textContent=file.name;slider.disabled=false;
    status.textContent='Experimental prediction ready; not geologically verified.';
  }catch(error){if(current===generation)status.textContent='Could not process image: '+error.message;}
  finally{button.disabled=false;}
};
slider.oninput=()=>{
  el('thresholdValue').textContent=Number(slider.value).toFixed(2);
  clearTimeout(timer);const current=generation, serial=++requestNumber, threshold=Number(slider.value);
  status.textContent='Updating binarisation of cached probabilities...';
  timer=setTimeout(async()=>{
    try{
      const result=await post('/threshold',JSON.stringify({token,threshold}),'application/json');
      if(current!==generation||serial!==requestNumber)return;
      show(result);status.textContent='Threshold updated; model weights and probabilities unchanged.';
    }catch(error){if(current===generation&&serial===requestNumber){el('results').hidden=true;status.textContent='Could not update threshold: '+error.message;}}
  },120);
};
</script></html>'''


def data_uri(array):
    ok, encoded = cv2.imencode('.png', array)
    if not ok:
        raise ValueError('Could not encode preview')
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode('ascii')


def array_statistics(values, quantiles=False):
    result = dict(minimum=float(np.min(values)), maximum=float(np.max(values)),
                  mean=float(np.mean(values)), standard_deviation=float(np.std(values)))
    if quantiles:
        result.update(zip(('median', 'p90', 'p95', 'p99'),
                          map(float, np.percentile(values, (50, 90, 95, 99)))))
    return result


def infer(payload, model):
    # Inspect dimensions before allocating the OpenCV image.
    with Image.open(io.BytesIO(payload)) as header:
        metadata = dict(width=header.width, height=header.height, mode=header.mode, format=header.format)
        if header.width*header.height > 16_000_000:
            raise ValueError('Image exceeds 16 million pixels')
    original = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError('Unreadable image; upload a JPEG, PNG, BMP, or TIFF')
    tensor = preprocess(original).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        if logits.shape != (1,1,256,256) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Invalid model output')
        probabilities = logits.sigmoid()[0,0].numpy().copy()
    print('UPLOAD_DIAGNOSTICS ' + json.dumps(dict(
        upload_sha256=hashlib.sha256(payload).hexdigest(),
        source=metadata,
        decoded_bgr_shape=list(original.shape),
        decoded_dtype=str(original.dtype),
        decoded_pixel_statistics_0_255=array_statistics(original),
        tensor_shape=list(tensor.shape),
        tensor_dtype=str(tensor.dtype),
        normalized_rgb_statistics=array_statistics(tensor.numpy()),
        normalized_per_channel={name: array_statistics(tensor[0, i].numpy())
                                for i, name in enumerate(('R', 'G', 'B'))},
        aspect_ratio=float(original.shape[1]/original.shape[0]),
        resize='Direct INTER_AREA resize to 256x256; no padding or crop; nonsquare inputs are stretched',
        probability_statistics=array_statistics(probabilities, quantiles=True),
        interpretation='Experimental model scores, not calibrated geological probabilities'
    ), allow_nan=False), flush=True)
    # Bound browser previews without changing model input preprocessing.
    h,w = original.shape[:2]
    scale = min(1, 1024/max(h,w))
    display = cv2.resize(original, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA) if scale<1 else original
    return display, probabilities


def appearance_assessment(display):
    """Descriptive 2-D texture heuristic, independent of model predictions."""
    gray = cv2.resize(cv2.cvtColor(display, cv2.COLOR_BGR2GRAY),
                      (256, 256), interpolation=cv2.INTER_AREA)
    contrast = float(gray.std())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = float(np.mean(cv2.Canny(smooth, 50, 100) > 0))
    if contrast < 10 or clipped > .40:
        label = 'Inconclusive'
    elif edges < .04:
        label = 'Relatively smooth-looking'
    elif edges < .12:
        label = 'Textured-looking'
    else:
        label = 'Highly textured-looking'
    return dict(label=label, edge_fraction=edges, grayscale_std=contrast,
                clipped_fraction=clipped,
                method='256x256 grayscale; 3x3 Gaussian blur; Canny 50/100. '
                'Edge density <4%: smooth-looking; 4-<12%: textured-looking; '
                '>=12%: highly textured-looking. Inconclusive if grayscale std <10 '
                'or >40% of pixels are <=5 or >=250. Heuristic thresholds, not calibrated.',
                limitation='Lighting, shadows, compression, resolution and stretching affect this '
                'appearance score. Not elevation, slope, rock identification or a geological class. '
                'Geological confirmation is unavailable.')


def region_statistics(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8)
    areas = sorted((int(x) for x in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    return dict(count=count-1, areas_pixels=areas,
                largest_percent=[100*a/binary.size for a in areas[:10]],
                definition='8-connected foreground regions on the 256x256 output; '
                'all regions included, no cleanup. Sizes are pixel counts, not physical areas.')


def prediction_warning(coverage):
    # UI caution threshold only, NOT a classifier or mask filter.
    if coverage >= 40:
        return ('Broad foreground activation: 40% or more of output pixels are marked. '
                'This model can incorrectly highlight black space, shadows or other background. '
                'Do not interpret this result as a landslide extent. No automatic space mask '
                'has been applied; inspect the original image and raw prediction.')
    return ''


def render(cache, tag, threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not .20 <= threshold <= .80:
        raise ValueError('Threshold must be a finite number from 0.20 to 0.80')
    display, probabilities = cache
    binary = (probabilities >= threshold).astype(np.uint8)*255
    coverage = float(np.mean(binary > 0)*100)
    mask = cv2.resize(binary,(display.shape[1],display.shape[0]),interpolation=cv2.INTER_NEAREST)
    overlay = display.copy()
    overlay[mask>0] = np.rint(.65*display[mask>0]+.35*np.array([0,0,255])).astype(np.uint8)
    return dict(original=data_uri(display),mask=data_uri(mask),raw_mask=data_uri(binary),overlay=data_uri(overlay),coverage=coverage,
                threshold=threshold,checkpoint=str(CHECKPOINT.relative_to(ROOT)),position=tag,
                regions=region_statistics(binary),appearance=appearance_assessment(display),
                warning=prediction_warning(coverage))


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type):
        self.send_response(code)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/':
            self.reply(404,b'Not found','text/plain'); return
        self.reply(200,PAGE.encode('utf-8'),'text/html; charset=utf-8')

    def do_POST(self):
        if self.path not in ('/predict', '/threshold'):
            self.reply(404,b'Not found','text/plain'); return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_BYTES:
                raise ValueError('Upload must be between 1 byte and 15 MB')
            expected = 'application/octet-stream' if self.path == '/predict' else 'application/json'
            if self.headers.get('Content-Type') != expected:
                raise ValueError('Unsupported upload request')
            payload=self.rfile.read(size)
            if len(payload)!=size:
                raise ValueError('Incomplete upload')
            if self.path == '/predict':
                self.server.cache = None
                self.server.token = None
                self.server.cache = infer(payload,self.server.model)
                self.server.token = uuid.uuid4().hex
                self.server.upload_sha256 = hashlib.sha256(payload).hexdigest()
                threshold = THRESHOLD
            else:
                params = json.loads(payload)
                if not isinstance(params, dict) or self.server.cache is None or params.get('token') != self.server.token:
                    raise ValueError('No matching cached image; upload the image again')
                threshold = params.get('threshold')
            result=render(self.server.cache,self.server.tag,threshold)
            result['token']=self.server.token
            print('PREDICTION_DIAGNOSTICS ' + json.dumps(dict(
                upload_sha256=self.server.upload_sha256,
                operation='new_upload' if self.path == '/predict' else 'cached_threshold_change',
                threshold=result['threshold'],
                predicted_pixel_coverage_percent=result['coverage'],
                output_shape=[256, 256],
                interpretation='Model-output pixel percentage, not physical landslide area'
            ), allow_nan=False), flush=True)
            self.reply(200,json.dumps(result).encode(),'application/json')
        except (OSError,ValueError,RuntimeError) as exc:
            self.reply(400,json.dumps({'error':str(exc)}).encode(),'application/json')


def self_test(server):
    # Check the embedded frontend contract without requiring a browser.
    known=array_statistics(np.array([0., .25, .5, .75, 1.]), quantiles=True)
    assert known['minimum']==0. and known['maximum']==1. and known['mean']==.5
    assert np.allclose([known[k] for k in ('median','p90','p95','p99')], [.5,.9,.95,.99])
    assert prediction_warning(40) and prediction_warning(100) and not prediction_warning(39.999)
    print('PASS: diagnostic probability statistics, quantiles and broad-activation warning boundaries.')
    import re
    from html.parser import HTMLParser
    class ElementIds(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids=[]
        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for name,value in attrs if name=='id')
    elements=ElementIds()
    elements.feed(PAGE)
    script=PAGE.split('<script>',1)[1].split('</script>',1)[0]
    referenced=set(re.findall(r"el\('([^']+)'\)",script))
    referenced.update(('original','mask','overlay'))
    assert len(elements.ids)==len(set(elements.ids)), 'Duplicate HTML IDs'
    assert not referenced-set(elements.ids), f'Missing HTML IDs: {referenced-set(elements.ids)}'
    assert PAGE.index('id="maskCaption"') < PAGE.index('<script>')
    print('PASS: all frontend element references exist; HTML IDs are unique and precede the script.')
    import threading
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    calls=[]
    hook=server.model.register_forward_pre_hook(lambda model, args: calls.append(args[0].detach().clone()))
    weights_before={k:v.detach().clone() for k,v in server.model.state_dict().items()}
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,checkpoint['model_state_dict'][k]) for k,v in weights_before.items())
    assert not server.model.training and next(server.model.parameters()).device.type=='cpu'
    del checkpoint
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base) as response:
            assert b'Upload a lunar image' in response.read()
        # Exercise the real loader's preprocessing with mocked reads only.
        from unittest.mock import patch
        from dataset_loader import LunarLandslideDataset
        color=np.random.default_rng(42).integers(0,256,(173,311,3),dtype=np.uint8)
        dataset=object.__new__(LunarLandslideDataset)
        dataset.image_size=256
        dataset.pairs=[(Path('synthetic.png'),Path('synthetic_mask.png'))]
        with patch.object(cv2,'imread',side_effect=[color,np.zeros(color.shape[:2],dtype=np.uint8)]):
            expected,_=dataset[0]
        assert torch.equal(preprocess(color),expected)
        # Synthetic image only: no dataset access and no fabricated labels.
        sample=np.tile(np.arange(256,dtype=np.uint8),(256,1))
        sample=cv2.cvtColor(sample,cv2.COLOR_GRAY2BGR)
        ok,encoded=cv2.imencode('.png',sample)
        assert ok
        request=Request(base+'/predict',data=encoded.tobytes(),headers={'Content-Type':'application/octet-stream'})
        with urlopen(request) as response:
            result=json.load(response)
        assert cv2.imdecode(np.frombuffer(base64.b64decode(result['raw_mask'].split(',')[1]),dtype=np.uint8),cv2.IMREAD_GRAYSCALE).shape==(256,256)
        assert result['threshold']==.5 and 0<=result['coverage']<=100
        assert result['checkpoint']==str(CHECKPOINT.relative_to(ROOT))
        assert len(calls)==1 and torch.equal(calls[0][0],preprocess(sample))
        cached_before=server.cache[1].copy()
        coverages=[]
        for threshold in (.2,.5,.8):
            request=Request(base+'/threshold',data=json.dumps(dict(token=result['token'],threshold=threshold)).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request) as response:
                updated=json.load(response)
            raw=base64.b64decode(updated['mask'].split(',')[1])
            mask=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            expected_binary=(cached_before>=threshold).astype(np.uint8)*255
            assert np.array_equal(mask,expected_binary)
            assert updated['coverage']==float(np.mean(expected_binary>0)*100)
            coverages.append(updated['coverage'])
        assert coverages[0]>=coverages[1]>=coverages[2]
        assert len(calls)==1 and np.array_equal(server.cache[1],cached_before)
        assert all(torch.equal(v,weights_before[k]) for k,v in server.model.state_dict().items())
        # Known synthetic probabilities check inclusive threshold boundaries,
        # even when the actual model's output happens to be nearly constant.
        synthetic=np.array([[.1,.2,.5,.8,.9]],dtype=np.float32)
        for threshold,coverage in ((.2,80.),(.5,60.),(.8,40.)):
            assert render((np.zeros((1,5,3),dtype=np.uint8),synthetic),'synthetic',threshold)['coverage']==coverage
        for params in [dict(token=result['token'],threshold=t) for t in (.19,.81,None,'0.5',True,float('nan'))]+[dict(token='stale',threshold=.5)]:
            try:
                urlopen(Request(base+'/threshold',data=json.dumps(params).encode(),headers={'Content-Type':'application/json'}))
                raise AssertionError('Invalid threshold/token accepted')
            except HTTPError as exc:
                assert exc.code==400
        for key in ('original','mask','overlay'):
            raw=base64.b64decode(result[key].split(',')[1])
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            assert image.shape[:2]==(256,256)
            if key=='mask': assert set(np.unique(image)) <= {0,255}
        try:
            urlopen(Request(base+'/predict',data=b'not an image',headers={'Content-Type':'application/octet-stream'}))
            raise AssertionError('Invalid image was accepted')
        except HTTPError as exc:
            assert exc.code==400 and 'error' in json.load(exc)
        assert server.cache is None and server.token is None
        assert len(calls)==1
        print('PASS: Experiment I checkpoint weights loaded exactly; CPU/eval mode.')
        print('PASS: preprocessing exactly matches the existing loader using mocked synthetic reads.')
        print('PASS: HTTP upload, three PNG previews, native-resolution mask download, threshold masks/coverage and boundary values.')
        print('PASS: slider endpoint performs no extra inference; cached probabilities and model weights unchanged.')
        print('PASS: invalid images, invalid thresholds and stale cache tokens rejected.')
    finally:
        server.shutdown()
        thread.join()
        hook.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    protected=[CHECKPOINT,ROOT/'unet_model.py',ROOT/'dataset_loader.py',ROOT/'tools/review_landslide_candidates.py',ROOT/'tools/preview_unet_predictions.py']
    before={}
    server=None
    try:
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        torch.set_num_threads(2)
        model,tag=load_model(CHECKPOINT)
        server=HTTPServer(('127.0.0.1',0 if args.self_test else args.port),Handler)
        server.model,server.tag=model,tag
        server.cache,server.token=None,None
        if args.self_test:
            self_test(server)
        else:
            print(f'Open http://127.0.0.1:{server.server_port} in your browser. Stop with Ctrl+C.',flush=True)
            print('Experiment I; threshold 0.5; experimental and not geologically verified.',flush=True)
            server.serve_forever()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Demo could not start or continue: {exc}. Check the checkpoint path and whether the port is available.', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nDemo stopped.')
    finally:
        if server: server.server_close()
        changed=[str(p) for p,h in before.items() if hashlib.sha256(p.read_bytes()).hexdigest()!=h]
        if changed: raise RuntimeError(f'Protected-file hash discrepancy: {changed}')
        print('Protected source/checkpoint hashes unchanged.')


if __name__=='__main__':
    main()
