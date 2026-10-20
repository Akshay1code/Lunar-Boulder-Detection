"""Isolated Experiment I. No training without --train; default is help."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from evaluate_experiment_e_validation import validation_dataset

F_PATH = ROOT / 'experiments/experiment_f_dice_loss/train_experiment_f.py'
spec = importlib.util.spec_from_file_location('experiment_i_baseline_f', F_PATH)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)  # Definitions only; never invoke F.main/train.
F_RUN = ROOT / 'experiments/experiment_f_dice_loss/runs/20261014T102910Z_262e760875cd4383a83f9d45a975d403'
LABEL = 'Agreement with unverified rectangular references; not geological accuracy'
SCHEDULE = list(range(0, 5181, 140))
OVERRIDES = dict(experiment='I', checkpoint_selection='strictly_higher_nonempty_validation_macro_dice_earliest_tie',
    evaluation_split='train_selected_and_valid', expected_valid_total=80, expected_valid_nonempty=19,
    augmentation=dict(kind='joint_quarter_turn', degrees=[0, 90, 180, 270], probabilities=[.25]*4,
                      seed=42, independent_generator=True, stage='after_preprocessing_train_only'))


def load_config():
    baseline = F.load_config()
    F.require(baseline == json.loads((F_RUN/'configuration.json').read_text()), 'Saved/current F config differs')
    cfg = json.loads((HERE/'config.json').read_text())
    expected = dict(baseline, **OVERRIDES)
    expected['baseline_integrity'] = cfg['baseline_integrity']
    F.require(cfg == expected, 'Experiment I settings differ from approved configuration')
    for path, digest in cfg['baseline_integrity'].items():
        F.require(F.digest(ROOT/path) == digest, 'Baseline changed: '+path)
    return cfg


def joint_rotate(images, targets, generator):
    F.require(images.ndim == targets.ndim == 4 and images.shape[0] == targets.shape[0]
              and images.shape[-2:] == targets.shape[-2:], 'Pair shape mismatch')
    turns = torch.randint(0, 4, (images.shape[0],), generator=generator).tolist()
    return (torch.stack([torch.rot90(x, k, (-2, -1)) for x,k in zip(images, turns)]).contiguous(),
            torch.stack([torch.rot90(y, k, (-2, -1)) for y,k in zip(targets, turns)]).contiguous(), turns)


def prepare(cfg):
    train, selected, records, before = F.prepare(cfg)
    valid = validation_dataset()  # Only valid; no test dataset construction.
    annotation = valid.image_dir/'_annotations.coco.json'
    paths = [annotation, HERE/'config.json', Path(__file__), HERE/'test_experiment_i.py', HERE/'README.md']
    paths += [p for pair in valid.pairs for p in pair]
    before.update({str(p.resolve()): F.digest(p) for p in paths})
    try:
        coco = json.loads(annotation.read_text())
        info = {r['file_name']: r for r in coco['images']}
        F.require(len(info) == 80 and set(info) == {p.name for p,_ in valid.pairs}, 'Validation COCO mismatch')
        land = {r['id'] for r in coco['categories'] if r['name'].lower() == 'landslide'}
        F.require(land == {3}, 'Unexpected validation category mapping')
        vr = []
        for i,(ip,mp) in enumerate(valid.pairs):
            image, target = valid[i]
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            anns = [a for a in coco['annotations'] if a['image_id'] == info[ip.name]['id']]
            F.verify_mask_rules(mask, info[ip.name], anns, land)
            nonempty = bool(target.any())
            F.require(nonempty == bool(mask.any()), 'Validation emptiness changed during resizing')
            vr.append(dict(image_key='valid/'+ip.name, coco_image_id=info[ip.name]['id'],
                           image_path=str(ip), mask_path=str(mp), reference_nonempty=nonempty,
                           reference_foreground_coverage=float(target.mean()),
                           image_sha256=F.digest(ip), mask_sha256=F.digest(mp)))
        F.require(len(vr)==80 and sum(r['reference_nonempty'] for r in vr)==19, 'Expected validation 80/19/61')
        F.require({p.resolve() for pair in train.pairs for p in pair}.isdisjoint(
                  {p.resolve() for pair in valid.pairs for p in pair}), 'Split paths overlap')
        saved = json.loads((F_RUN/'provenance.json').read_text())['selection']
        F.require(saved == records, 'Selection differs from F')
        return train, selected, records, valid, vr, before
    finally:
        F.verify_unchanged(before)


def aggregate(rows):
    fields = ('bce','dice_loss','total_loss','dice','iou','precision','recall',
              'predicted_foreground_coverage','reference_foreground_coverage')
    result = dict(images=len(rows), mean={}, defined_counts={})
    for name in fields:
        values = [r[name] for r in rows if r[name] is not None]
        result['mean'][name] = sum(values)/len(values) if values else None
        result['defined_counts'][name] = len(values)
    result['images_with_foreground'] = sum(r['predicted_foreground_coverage'] > 0 for r in rows)
    return result


def evaluate(model, loader, records):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            components = F.loss_components(logits, targets)
            for j,(pred,target) in enumerate(zip(logits.sigmoid() >= .5, targets > 0)):
                record = records[len(rows)]
                tp,fp,fn = int((pred & target).sum()),int((pred & ~target).sum()),int((~pred & target).sum())
                coverage = float(target.float().mean())
                F.require(coverage == record['reference_foreground_coverage'], 'Evaluation association mismatch')
                bce = float(components['bce_per_image'][j]); dl = float(components['dice_loss_per_image'][j])
                rows.append(dict(image_key=record['image_key'], bce=bce, dice_loss=dl, total_loss=bce+dl,
                    iou=tp/(tp+fp+fn) if tp+fp+fn else None, reference_foreground_coverage=coverage,
                    **F.metrics(tp,fp,fn,target.numel())))
    F.require(len(rows)==len(records), 'Evaluation count mismatch')
    return dict(per_image=rows, all=aggregate(rows),
        nonempty=aggregate([r for r in rows if r['reference_foreground_coverage']>0]),
        empty_unverified=aggregate([r for r in rows if r['reference_foreground_coverage']==0]))


def improves(score, previous):
    F.require(score is not None and math.isfinite(score), 'Undefined/nonfinite selection Dice')
    return previous is None or score > previous


def output_path(run, name):
    runs = HERE/'runs'
    F.require(runs.resolve().parent == HERE.resolve(), 'Redirected runs directory')
    run = Path(run).resolve()
    F.require(run.parent == runs.resolve(), 'Run outside Experiment I')
    p = (run/name).resolve()
    F.require(p.parent == run and p != run, 'Output path outside run')
    F.require(not p.exists(), 'Refusing overwrite: '+str(p))
    return p


def create_run():
    runs = HERE/'runs'
    F.require(runs.resolve().parent == HERE.resolve(), 'Redirected runs directory')
    runs.mkdir(exist_ok=True)
    run = runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
    run.mkdir(exist_ok=False)
    return run


def write_json(run, name, value):
    with output_path(run,name).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def save_checkpoint(run, name, model, optimizer, cfg, evaluation, generator):
    state = copy.deepcopy(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        update=evaluation['update'], configuration=cfg, evaluation=evaluation,
        augmentation_rng_state=generator.get_state(), label=LABEL))
    path = output_path(run,name)
    with path.open('xb') as f:
        torch.save(state,f)
    F.verify_checkpoint(torch.load(path,map_location='cpu',weights_only=True),state)
    return path


def train(cfg, train_ds, selected, tr, valid, vr, before):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    training, evaluation, sampler = F.make_loaders(train_ds, selected, 42)
    model = F.UNet().cpu()  # Same construction order as F, never load F weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    augmentation = torch.Generator().manual_seed(42)
    validation = DataLoader(valid,batch_size=2,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(42))
    counter = F.StepBudget(optimizer)
    run = create_run(); start = time.perf_counter()
    score = None; best_path = None; best_eval = None; evaluations=[]; presentations=0
    try:
        write_json(run,'configuration.json',cfg)
        write_json(run,'provenance.json',dict(train=tr,validation=vr,protected_hashes=before,
            python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,label=LABEL))
        with output_path(run,'training_history.jsonl').open('x') as th, output_path(run,'evaluation_history.jsonl').open('x') as eh:
            def record(step):
                nonlocal score,best_path,best_eval
                event=dict(update=step,elapsed_seconds=time.perf_counter()-start,
                    train=evaluate(model,evaluation,tr),validation=evaluate(model,validation,vr))
                eh.write(json.dumps(event,allow_nan=False)+'\n');eh.flush()
                evaluations.append(step)
                candidate=event['validation']['nonempty']['mean']['dice']
                if improves(candidate,score):
                    best_path=save_checkpoint(run,f'best_validation_update_{step:05d}.pth',model,optimizer,cfg,event,augmentation)
                    score=candidate;best_eval=event
                print(json.dumps(dict(update=step,validation=event['validation']['nonempty'],
                    empty_unverified=event['validation']['empty_unverified'],best_update=best_eval['update'])),flush=True)
                return event
            final=record(0)
            for step,pass_number,batch_number,(images,targets) in F.bounded_batches(training,5180):
                images,targets,turns=joint_rotate(images,targets,augmentation)
                model.train();optimizer.zero_grad(set_to_none=True)
                components=F.loss_components(model(images),targets)
                components['total_loss'].backward()
                F.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Invalid gradient')
                optimizer.step()
                F.require(counter.successful==step,'Optimizer counter mismatch')
                presentations+=len(images)
                th.write(json.dumps(dict(update=step,pass_number=pass_number,batch=batch_number,quarter_turns=turns,
                    bce=float(components['bce'].detach()),dice_loss=float(components['dice_loss'].detach()),
                    total_loss=float(components['total_loss'].detach()),elapsed_seconds=time.perf_counter()-start))+'\n');th.flush()
                if step%140==0: final=record(step)
        F.check_completion(counter,optimizer,evaluations,presentations,pass_number,batch_number)
        save_checkpoint(run,'last.pth',model,optimizer,cfg,final,augmentation)
        # Immutable improvement checkpoints preserve actual weights. Publish alias only once.
        with output_path(run,'best.pth').open('xb') as f: f.write(best_path.read_bytes())
        F.require(F.digest(run/'best.pth')==F.digest(best_path),'Best alias differs')
        after=F.verify_unchanged(before)
        write_json(run,'summary.json',dict(status='completed',successful_updates=counter.successful,
            best_update=best_eval['update'],best_validation=best_eval['validation'],final_validation=final['validation'],
            best_checkpoint_sha256=F.digest(run/'best.pth'),last_checkpoint_sha256=F.digest(run/'last.pth'),
            elapsed_seconds=time.perf_counter()-start,protected_hashes_after=after))
    except BaseException as exc:
        write_json(run,'incomplete.json',dict(status='incomplete',successful_updates=counter.successful,
            error=repr(exc),best_checkpoint=str(best_path) if best_path else None))
        raise
    finally:
        counter.close(); F.verify_unchanged(before)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--train',action='store_true')
    group.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv)
    if not (args.train or args.validate_only): parser.print_help();return
    cfg=load_config();torch.set_num_threads(2)
    prepared=prepare(cfg)
    if args.validate_only:
        print('PASS: 73 F-identical training records; validation 80/19/61; COCO/mask alignment; protected hashes unchanged. No model, optimizer or run created.')
        return
    train(cfg,*prepared)


if __name__=='__main__': main()
