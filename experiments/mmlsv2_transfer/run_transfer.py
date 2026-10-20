"""Isolated Mars-to-Moon trial. Default validates only. Explicit --train required."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, copy, csv, ctypes, importlib.util, json, os, random, time, uuid
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from mmlsv2_dataset import MMLSv2RGB, read_pair, preprocess
from prepare_mmlsv2 import verify
from preview_unet_predictions import digest, font
from visualize_validation_errors import render_row
spec=importlib.util.spec_from_file_location('baseline_I',ROOT/'experiments/experiment_i_rotation/train_experiment_i.py')
I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)
F=I.F
BASE=ROOT/'experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040'

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)

def memory():
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    try:
        p=PMC();p.cb=ctypes.sizeof(p)
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        if not fn(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb):return None
        return dict(working_set_mb=p.WorkingSetSize/1024**2,peak_working_set_mb=p.PeakWorkingSetSize/1024**2)
    except Exception:return None

def path(run,name):
    run=Path(run).resolve();p=(run/name).resolve()
    if run.parent!=(HERE/'runs').resolve() or p.parent!=run or p.exists():
        raise ValueError('Unsafe/existing output: '+str(p))
    return p

def write(run,name,obj):
    with path(run,name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)

class Counter:
    def __init__(self,opt,budget):
        self.n=0;self.budget=budget
        self.handles=[opt.register_step_pre_hook(self.pre),opt.register_step_post_hook(self.post)]
    def pre(self,*args):
        if self.n>=self.budget:raise RuntimeError('Optimizer budget exceeded')
    def post(self,*args):self.n+=1
    def close(self):
        for h in self.handles:h.remove()

def save(run,name,model,opt,event,sampler,aug,cfg):
    state=copy.deepcopy(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
        update=event['update'],stage=event['stage'],configuration=cfg,evaluation=event,
        torch_rng=torch.get_rng_state(),sampling_rng=sampler.get_state(),augmentation_rng=aug.get_state(),
        python_rng=random.getstate(),label=I.LABEL))
    p=path(run,name)
    with p.open('xb') as f:torch.save(state,f)
    F.verify_checkpoint(torch.load(p,map_location='cpu',weights_only=True),state)
    return p

def preflight():
    cfg=json.loads((HERE/'config.json').read_text())
    assert (cfg['seed'],cfg['batch_size'],cfg['learning_rate'],cfg['mars_updates'],cfg['lunar_updates'],cfg['threshold'])==(42,2,.001,2330,5180,.5)
    seed();prepared=I.prepare(I.load_config())
    train,selected,tr,valid,vr,protected=prepared
    protected.update({str(p.resolve()):digest(p) for p in HERE.glob('*') if p.is_file()})
    for p in [ROOT/'tools/mmlsv2_dataset.py',ROOT/'tools/prepare_mmlsv2.py',ROOT/'datasets/mmlsv2/README.source.md',ROOT/'datasets/mmlsv2/LICENSE.github.txt']:
        protected[str(p.resolve())]=digest(p)
    entries=json.loads((ROOT/'datasets/mmlsv2/source_manifest.json').read_text())['files']
    for entry in entries:
        p=ROOT/'datasets/mmlsv2'/entry['path'];verify(p.read_bytes(),entry);protected[str(p.resolve())]=digest(p)
    mars=[];meta={}
    import tifffile
    for split in ['train','val']:
        ds=MMLSv2RGB(split);rr=[];geo=[0,0];ranges=[]
        for j,(ip,mp) in enumerate(ds.pairs):
            a,m=read_pair(ip,mp);x,y=ds[j]
            assert np.array_equal(y[0].numpy()[::2,::2],m)
            assert np.allclose(x.numpy().transpose(1,2,0)[::2,::2],a[:,:,:3],atol=1e-7)
            for k,p in enumerate((ip,mp)):
                with tifffile.TiffFile(p) as tf:geo[k]+=tf.geotiff_metadata is not None
            rr.append(dict(image_key=f'mars/{split}/{ip.name}',reference_foreground_coverage=float(y.mean())))
            ranges.append(float(y.mean()))
        meta[split]=dict(pairs=len(ds),empty=sum(v==0 for v in ranges),mean_coverage=float(np.mean(ranges)),image_geotags=geo[0],mask_geotags=geo[1])
        mars.append((ds,rr))
    assert {p.name for p,_ in mars[0][0].pairs}.isdisjoint({p.name for p,_ in mars[1][0].pairs})
    state=torch.load(BASE/'best.pth',map_location='cpu',weights_only=True)
    model=F.UNet();model.load_state_dict(state['model_state_dict'],strict=True);assert state['update']==3080
    F.verify_unchanged(protected)
    return cfg,prepared,mars,meta,protected

def loader(ds,shuffle=False):
    gen=torch.Generator().manual_seed(42)
    return DataLoader(ds,batch_size=2,shuffle=shuffle,num_workers=0,drop_last=False,generator=gen),gen

def stage(run,name,model,ds,eval_ds,records,validation,validation_records,cfg,steps,interval,rotate):
    seed()  # Fresh optimizer/sampling/augmentation, model weights intentionally retained.
    training,sampler=loader(ds,True);evaluation,_=loader(eval_ds);vl,_=loader(validation)
    opt=torch.optim.Adam(model.parameters(),lr=.001);aug=torch.Generator().manual_seed(42)
    count=Counter(opt,steps);start=time.perf_counter();cpu=time.process_time();best=None;best_path=None;last_event=None;events=[]
    with path(run,name+'_training.jsonl').open('x',encoding='utf-8') as th, path(run,name+'_evaluation.jsonl').open('x',encoding='utf-8') as eh:
        def record():
            nonlocal best,best_path,last_event
            event=dict(stage=name,update=count.n,elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                resources=memory(),train=I.evaluate(model,evaluation,records),validation=I.evaluate(model,vl,validation_records))
            assert all(p.grad is None for p in model.parameters()), 'Evaluation must not create gradients'
            eh.write(json.dumps(event,allow_nan=False)+'\n');eh.flush();events.append(count.n);last_event=event
            score=event['validation']['nonempty']['mean']['dice']
            # Every scheduled state is immutable and verified; no overwritten recovery files.
            cp=save(run,f'{name}_update_{count.n:05d}.pth',model,opt,event,sampler,aug,cfg)
            if I.improves(score,best):best=score;best_path=cp
            print(json.dumps(dict(stage=name,update=count.n,train_loss=event['train']['all']['mean']['total_loss'],
                validation_dice=score,validation_iou=event['validation']['nonempty']['mean']['iou'],
                empty_coverage=event['validation']['empty_unverified']['mean']['predicted_foreground_coverage'],
                elapsed_seconds=event['elapsed_seconds'],resources=event['resources'])),flush=True)
        try:
            record();passes=0;presentations=0
            while count.n<steps:
                passes+=1
                for images,targets in training:
                    if count.n>=steps:break
                    if rotate:images,targets,turns=I.joint_rotate(images,targets,aug)
                    else:turns=None
                    model.train();opt.zero_grad(set_to_none=True)
                    loss=F.loss_components(model(images),targets);loss['total_loss'].backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                    opt.step();opt.zero_grad(set_to_none=True);presentations+=len(images)
                    th.write(json.dumps(dict(update=count.n,pass_number=passes,batch_size=len(images),quarter_turns=turns,
                        bce=float(loss['bce'].detach()),dice_loss=float(loss['dice_loss'].detach()),total_loss=float(loss['total_loss'].detach()),elapsed_seconds=time.perf_counter()-start))+'\n');th.flush()
                    if count.n%interval==0 or count.n==steps:record()
            assert count.n==steps and all(int(v['step'])==steps for v in opt.state.values())
            assert events==list(range(0,steps+1,interval))
            assert presentations==({'mars':465*10,'lunar':73*140}[name])
            final_path=run/f'{name}_update_{steps:05d}.pth'
            result=dict(status='completed',updates=count.n,passes=passes,presentations=presentations,
                elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,resources=memory(),
                best_checkpoint=str(best_path),last_checkpoint=str(final_path),best_validation_dice=best,final_evaluation=last_event)
            write(run,name+'_summary.json',result)
            return result
        except BaseException as exc:
            write(run,name+'_incomplete.json',dict(status='incomplete',successful_updates=count.n,error=repr(exc)))
            raise
        finally:count.close()

def compare(run,modelA,modelB,valid,vr,cfg):
    vl,_=loader(valid)
    a=I.evaluate(modelA,vl,vr);b=I.evaluate(modelB,vl,vr)
    write(run,'comparison.json',dict(A=a,B=b))
    with path(run,'per_image.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['arm']+list(a['per_image'][0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,res in [('A',a),('B',b)]:
            writer.writerows(dict(arm=arm,**row) for row in res['per_image'])
    nonempty=[j for j,r in enumerate(vr) if r['reference_nonempty']]
    gains=sorted(nonempty,key=lambda j:b['per_image'][j]['dice']-a['per_image'][j]['dice'])
    empty=sorted([j for j,r in enumerate(vr) if not r['reference_nonempty']],key=lambda j:b['per_image'][j]['predicted_foreground_coverage'],reverse=True)
    chosen=list(dict.fromkeys(gains[:3]+gains[-3:]+empty[:3]))
    for j in chosen:
        ip,mp=valid.pairs[j];x,y=valid[j]
        canvas=Image.new('RGB',(1080,744),'white');draw=ImageDraw.Draw(canvas)
        for k,(label,model,res) in enumerate([('A: lunar-only Experiment I',modelA,a),('B: Mars-pretrained, lunar fine-tuned',modelB,b)]):
            with torch.inference_mode():pred=(model(x[None]).sigmoid()[0,0].numpy()>=.5)
            r=res['per_image'][j]
            row=dict(coco_image_id=vr[j]['coco_image_id'],dice_unverified=r['dice'],iou_unverified=r['iou'],
                reference_foreground_percent=r['reference_foreground_coverage']*100,predicted_foreground_percent=r['predicted_foreground_coverage']*100)
            panel=render_row(ip,mp,pred,row);draw.text((8,k*372+4),label,font=font(16),fill='black');canvas.paste(panel,(0,k*372+25));panel.close()
        p=path(run,f'comparison_valid_{j:02d}.png')
        with p.open('xb') as f:canvas.save(f,format='PNG')
        with Image.open(p) as check:check.load()
        canvas.close()
    return a,b,chosen

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train',action='store_true');args=parser.parse_args()
    cfg,prepared,mars,meta,protected=preflight()
    print('PREFLIGHT PASS: Mars 465/66, Moon 73/80 (19 nonempty), protected hashes unchanged',flush=True)
    if not args.train:return
    train,selected,tr,valid,vr,_=prepared
    runs=HERE/'runs';runs.mkdir(exist_ok=True)
    run=runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex);run.mkdir(exist_ok=False)
    write(run,'configuration.json',cfg);write(run,'provenance.json',dict(mars=meta,train=tr,validation=vr,protected_hashes=protected,python=sys.version,torch=str(torch.__version__),numpy=np.__version__))
    print('RUN_DIRECTORY',run,flush=True)
    start=time.perf_counter()
    try:
        seed();modelA=F.UNet();state=torch.load(BASE/'best.pth',map_location='cpu',weights_only=True);modelA.load_state_dict(state['model_state_dict'],strict=True);modelA.eval()
        vl,_=loader(valid);fresh=I.evaluate(modelA,vl,vr)
        for name in ['dice','iou','precision','recall']:
            assert abs(fresh['nonempty']['mean'][name]-state['evaluation']['validation']['nonempty']['mean'][name])<1e-7
        write(run,'A_reproduced_validation.json',fresh)
        seed();model=F.UNet()
        # Check fresh seed/model setup against the original I update-zero loss.
        subset=torch.utils.data.Subset(train,selected);el,_=loader(subset)
        initial=I.evaluate(model,el,tr)
        first=json.loads((BASE/'evaluation_history.jsonl').open().readline())
        assert abs(initial['all']['mean']['total_loss']-first['train']['all']['mean']['total_loss'])<1e-7
        write(run,'initialization_check.json',dict(current=initial['all'],baseline=first['train']['all']))
        m=stage(run,'mars',model,mars[0][0],mars[0][0],mars[0][1],mars[1][0],mars[1][1],cfg,2330,233,False)
        # Fixed-budget final Mars weights retained; no lunar validation-driven Mars selection.
        l=stage(run,'lunar',model,subset,subset,tr,valid,vr,cfg,5180,140,True)
        best=torch.load(l['best_checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(best['model_state_dict'],strict=True);model.eval()
        a,b,chosen=compare(run,modelA,model,valid,vr,cfg)
        assert b==best['evaluation']['validation'], 'Best weights do not reproduce recorded validation'
        for alias,source in [('best.pth',Path(l['best_checkpoint'])),('last.pth',Path(l['last_checkpoint']))]:
            with path(run,alias).open('xb') as f:f.write(source.read_bytes())
            assert digest(run/alias)==digest(source)
        F.verify_unchanged(protected)
        write(run,'summary.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,mars=m,lunar=l,
            A_checkpoint=str(BASE/'best.pth'),B_best_update=best['update'],A=a['nonempty'],B=b['nonempty'],
            A_empty=a['empty_unverified'],B_empty=b['empty_unverified'],comparison_indices=chosen,
            protected_files_unchanged=True,resources=memory()))
        print('COMPLETED',run,'A_DICE',a['nonempty']['mean']['dice'],'B_DICE',b['nonempty']['mean']['dice'],flush=True)
    except BaseException as exc:
        write(run,'incomplete.json',dict(status='incomplete',error=repr(exc),elapsed_seconds=time.perf_counter()-start));raise
    finally:F.verify_unchanged(protected)

if __name__=='__main__':main()
