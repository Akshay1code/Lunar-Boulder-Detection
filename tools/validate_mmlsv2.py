"""Validate downloaded Mars train/val files and write a unique review. No training."""
import sys
sys.dont_write_bytecode = True
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from PIL import Image, ImageDraw
import tifffile
import torch
from mmlsv2_dataset import MMLSv2RGB, read_pair, preprocess, ROOT
from prepare_mmlsv2 import verify

PROJECT = ROOT.parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protected = [PROJECT/'unet_model.py', PROJECT/'dataset_loader.py', PROJECT/'train_unet.py',
                 PROJECT/'experiments/experiment_i_rotation/train_experiment_i.py']
    # Only checkpoint files, never dataset/test directories.
    protected += list((PROJECT/'experiments').glob('*/runs/*/*.pth'))
    protected += list((PROJECT/'checkpoints').glob('*.pth'))
    before = {str(p): digest(p) for p in protected}
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    entries = {e['path']:e for e in manifest['files']}
    records=[]
    for split in ['train','val']:
        dataset=MMLSv2RGB(split)
        for index,(ip,mp) in enumerate(dataset.pairs):
            for p in (ip,mp): verify(p.read_bytes(),entries[p.relative_to(ROOT).as_posix()])
            image,mask=read_pair(ip,mp)
            x,y=dataset[index]
            assert x.shape==(3,256,256) and y.shape==(1,256,256)
            assert torch.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
            # Exact 2x upsampling must preserve every source label and RGB sample.
            assert np.array_equal(y[0].numpy()[::2,::2],mask)
            assert np.allclose(x.numpy().transpose(1,2,0)[::2,::2],image[:,:,:3],atol=1e-7)
            assert float(y.mean())==float(mask.mean())
            with tifffile.TiffFile(ip) as f: ig=f.geotiff_metadata
            with tifffile.TiffFile(mp) as f: mg=f.geotiff_metadata
            records.append(dict(split=split,index=index,filename=ip.name,image_sha256=digest(ip),mask_sha256=digest(mp),
                image_shape=str(image.shape),image_dtype=str(image.dtype),mask_dtype=str(mask.dtype),mask_values=str(np.unique(mask).tolist()),
                image_min=float(image.min()),image_max=float(image.max()),rgb_min=float(image[:,:,:3].min()),rgb_max=float(image[:,:,:3].max()),
                foreground_percent=float(mask.mean()*100),image_geotags=ig is not None,mask_geotags=mg is not None))
        print('Validated',split,len(dataset),'pairs',flush=True)
    # Synthetic test: catches BGR swaps, double normalization and mask interpolation.
    a=np.zeros((128,128,7),np.float32);a[:,:,0]=.2;a[:,:,1]=.4;a[:,:,2]=.8
    m=np.zeros((128,128),np.uint8);m[10:20,30:40]=1
    x,y=preprocess(a,m)
    assert torch.allclose(x[:,0,0],torch.tensor([.2,.4,.8]))
    assert int(y.sum())==400 and np.array_equal(y[0].numpy()[::2,::2],m)
    for invalid in ['test','valid','../train']:
        try: MMLSv2RGB(invalid)
        except ValueError: pass
        else: raise AssertionError('Unsafe split accepted')
    sys.path.insert(0,str(PROJECT))
    from unet_model import UNet
    ck=PROJECT/'experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth'
    state=torch.load(ck,map_location='cpu',weights_only=True)
    model=UNet();model.load_state_dict(state['model_state_dict'],strict=True)
    assert state['update']==3080
    # No forward/backward or optimizer calls.
    summary={}
    for split in ['train','val']:
        rr=[r for r in records if r['split']==split];v=[r['foreground_percent'] for r in rr]
        summary[split]=dict(pairs=len(rr),empty=sum(t==0 for t in v),coverage_min=min(v),coverage_mean=float(np.mean(v)),coverage_max=max(v),
            images_with_geotags=sum(r['image_geotags'] for r in rr),masks_with_geotags=sum(r['mask_geotags'] for r in rr))
    trainhash={r['image_sha256'] for r in records if r['split']=='train'}
    summary['exact_image_duplicates_across_train_val']=sum(r['image_sha256'] in trainhash for r in records if r['split']=='val')
    parent=PROJECT/'dataset_source_comparisons'
    out=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_mmlsv2_preparation_'+uuid.uuid4().hex)
    out.mkdir(exist_ok=False)
    with (out/'inventory.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for split in ['train','val']:
        rr=sorted([r for r in records if r['split']==split],key=lambda r:r['foreground_percent'])
        selected=[rr[0],rr[len(rr)//2],rr[-1]]
        for number,r in enumerate(selected):
            image,mask=read_pair(ROOT/split/'images'/r['filename'],ROOT/split/'masks'/r['filename'])
            rgb=np.rint(image[:,:,:3]*255).astype(np.uint8);overlay=rgb.copy()
            overlay[mask>0]=np.rint(.65*rgb[mask>0]+.35*np.array([255,0,0])).astype(np.uint8)
            canvas=Image.new('RGB',(810,335),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),f"MMLSv2 {split}: {r['filename']} | foreground {r['foreground_percent']:.3f}%",fill='black')
            for j,(data,label) in enumerate([(rgb,'Published RGB bands 1-3'),(mask*255,'Published Mars reference mask'),(overlay,'Reference overlay (red)')]):
                draw.text((8+j*268,30),label,fill='black')
                panel=Image.fromarray(data).resize((256,256),Image.Resampling.NEAREST)
                canvas.paste(panel,(8+j*268,50))
            draw.text((8,315),'Dataset annotation; not independently verified here. Not lunar ground truth.',fill='black')
            path=out/f'{split}_{number:02d}_{Path(r["filename"]).stem}.png'
            with path.open('xb') as f: canvas.save(f,format='PNG')
            with Image.open(path) as im: im.load();assert im.size==(810,335)
    assert before=={str(p):digest(p) for p in protected}, 'Protected project file changed'
    for entry in entries.values(): verify((ROOT/entry['path']).read_bytes(),entry)
    summary.update(checkpoint_update=state['update'],parameters=sum(p.numel() for p in model.parameters()),
                   synthetic_checks_passed=True,protected_files_unchanged=True,protected_hashes=before,
                   all_downloaded_files_reverified=True,tifffile_version=tifffile.__version__)
    with (out/'validation.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='protected_hashes'},indent=2),flush=True)
    print('REPORT_DIRECTORY',out,flush=True)

if __name__=='__main__':main()
