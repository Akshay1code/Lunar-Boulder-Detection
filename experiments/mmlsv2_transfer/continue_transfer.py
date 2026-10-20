"""Recover the interrupted Mars stage into a new run; never overwrite old output."""
import sys
sys.dont_write_bytecode=True
from run_transfer import *

OLD=HERE/'runs/20261019T160215Z_5de7d6a8ba254653bd3174a6db6c05ae'

def recover_mars(run,model,mars,cfg,state):
    ds,records=mars[0];vd,vr=mars[1]
    # Replay indices only: reproduce RandomSampler's exhausted-iterator RNG state.
    index_loader,g=loader(list(range(len(ds))),True)
    for epoch in range(9):
        it=iter(index_loader)
        for _ in range(233):next(it)
        if epoch==8:assert torch.equal(g.get_state(),state['sampling_rng']), 'Sampler state not reproducible'
        try:next(it)
        except StopIteration:pass
        else:raise AssertionError('Unexpected sampler length')
    training,sampler=loader(ds,True);sampler.set_state(g.get_state())
    evaluation,_=loader(ds);validation,_=loader(vd)
    model.load_state_dict(state['model_state_dict'],strict=True)
    opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(state['optimizer_state_dict'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    aug=torch.Generator();aug.set_state(state['augmentation_rng'])
    count=Counter(opt,2330);count.n=2097
    previous={x['update']:x for x in [json.loads(line) for line in (OLD/'mars_training.jsonl').read_text().splitlines()]}
    start=time.perf_counter();checks=[]
    try:
        with path(run,'mars_continued_training.jsonl').open('x') as f:
            for images,targets in training:
                if count.n>=2330:break
                model.train();opt.zero_grad(set_to_none=True)
                loss=F.loss_components(model(images),targets);loss['total_loss'].backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                opt.step();opt.zero_grad(set_to_none=True)
                row=dict(update=count.n,pass_number=10,batch_size=len(images),bce=float(loss['bce'].detach()),dice_loss=float(loss['dice_loss'].detach()),total_loss=float(loss['total_loss'].detach()),elapsed_seconds=time.perf_counter()-start)
                if count.n in previous:
                    differences={k:abs(row[k]-previous[count.n][k]) for k in ['bce','dice_loss','total_loss']}
                    assert max(differences.values())<1e-7, ('Recovery mismatch',count.n,differences)
                    checks.append(dict(update=count.n,differences=differences))
                    if count.n==2104:
                        write(run,'recovery_verified.json',dict(checkpoint=str(OLD/'mars_update_02097.pth'),checks=checks,verified=True))
                        print('RECOVERY VERIFIED: updates 2098-2104 match saved losses',flush=True)
                f.write(json.dumps(row)+'\n');f.flush()
                if count.n%50==0:print('Mars update',count.n,'loss',row['total_loss'],flush=True)
        assert count.n==2330 and len(checks)==7
        assert all(int(v['step'])==2330 for v in opt.state.values())
        event=dict(stage='mars',update=2330,elapsed_seconds=time.perf_counter()-start,resources=memory(),train=I.evaluate(model,evaluation,records),validation=I.evaluate(model,validation,vr))
        cp=save(run,'mars_update_02330.pth',model,opt,event,sampler,aug,cfg)
        result=dict(status='completed',updates=2330,source_run=str(OLD),rollback_replayed_updates=7,continuation_seconds=time.perf_counter()-start,final_checkpoint=str(cp),final_evaluation=event)
        write(run,'mars_summary.json',result)
        return result
    finally:count.close()

def main():
    cfg,prepared,mars,meta,protected=preflight()
    F.verify_unchanged(json.loads((OLD/'provenance.json').read_text())['protected_hashes'])
    state=torch.load(OLD/'mars_update_02097.pth',map_location='cpu',weights_only=True)
    assert state['update']==2097 and state['stage']=='mars'
    assert all(int(v['step'])==2097 for v in state['optimizer_state_dict']['state'].values())
    run=HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_continued_'+uuid.uuid4().hex);run.mkdir(exist_ok=False)
    write(run,'configuration.json',cfg);write(run,'provenance.json',dict(protected_hashes=protected,source_run=str(OLD),mars=meta))
    print('CONTINUATION_DIRECTORY',run,flush=True)
    start=time.perf_counter()
    try:
        seed();model=F.UNet();m=recover_mars(run,model,mars,cfg,state)
        train,selected,tr,valid,vr,_=prepared
        subset=torch.utils.data.Subset(train,selected)
        l=stage(run,'lunar',model,subset,subset,tr,valid,vr,cfg,5180,140,True)
        best=torch.load(l['best_checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(best['model_state_dict'],strict=True);model.eval()
        modelA=F.UNet();a_state=torch.load(BASE/'best.pth',map_location='cpu',weights_only=True)
        modelA.load_state_dict(a_state['model_state_dict'],strict=True);modelA.eval()
        a,b,chosen=compare(run,modelA,model,valid,vr,cfg)
        assert b==best['evaluation']['validation']
        for k in ['dice','iou','precision','recall']:
            assert abs(a['nonempty']['mean'][k]-a_state['evaluation']['validation']['nonempty']['mean'][k])<1e-7
        for alias,source in [('best.pth',Path(l['best_checkpoint'])),('last.pth',Path(l['last_checkpoint']))]:
            with path(run,alias).open('xb') as f:f.write(source.read_bytes())
            assert digest(run/alias)==digest(source)
        F.verify_unchanged(protected)
        write(run,'summary.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,mars=m,lunar=l,A_checkpoint=str(BASE/'best.pth'),B_best_update=best['update'],A=a['nonempty'],B=b['nonempty'],A_empty=a['empty_unverified'],B_empty=b['empty_unverified'],comparison_indices=chosen,protected_files_unchanged=True,resources=memory()))
        print('COMPLETED',run,flush=True)
    except BaseException as exc:
        write(run,'incomplete.json',dict(status='incomplete',error=repr(exc)));raise
    finally:F.verify_unchanged(protected)

if __name__=='__main__':main()
