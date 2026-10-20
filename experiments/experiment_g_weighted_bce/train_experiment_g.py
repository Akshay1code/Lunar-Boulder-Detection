"""Experiment G infrastructure. No training without explicit --train.

Reuses F selection, loader sampling, Dice loss, counters and serialization checks.
All metrics describe agreement with UNVERIFIED rectangular references.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('experiment_f_baseline', ROOT/'experiments/experiment_f_dice_loss/train_experiment_f.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

F_RUN = ROOT/'experiments/experiment_f_dice_loss/runs/20261014T102910Z_262e760875cd4383a83f9d45a975d403'
LABEL = 'Agreement with UNVERIFIED rectangular reference masks; not verified geological accuracy'
FOREGROUND, BACKGROUND = 733568, 4050560
POS_WEIGHT = round(BACKGROUND/FOREGROUND, 6)
BUDGET = 5180
SCHEDULE = list(range(0, BUDGET+1, 140))


def configuration():
    baseline = F.load_config()
    expected = dict(baseline)
    expected.update(experiment='G', loss='WeightedBCEWithLogitsLoss+SoftDiceLoss', pos_weight=POS_WEIGHT,
        description=LABEL, checkpoint_selection='strict_training_dice_and_strict_nonempty_validation_dice_earliest_ties',
        evaluation_split='train_selected_and_valid',
        weight_calculation={'foreground_pixels':FOREGROUND, 'background_pixels':BACKGROUND,
                            'ratio':BACKGROUND/FOREGROUND, 'round_decimal_places':6, 'split':'train_selected'},
        validation_counts={'total':80, 'nonempty':19, 'empty_unverified':61})
    actual = json.loads((EXPERIMENT/'config.json').read_text(encoding='utf-8'))
    F.require(actual == expected, 'G configuration differs from approved settings')
    F.require(baseline == json.loads((F_RUN/'configuration.json').read_text(encoding='utf-8')), 'F config differs from completed run')
    return actual, baseline


def loss_components(logits, targets):
    components = F.loss_components(logits, targets)  # Preserve F's exact soft Dice.
    weighted = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), pos_weight=logits.new_tensor(POS_WEIGHT, dtype=torch.float32), reduction='none')
    per_image = weighted.flatten(1).mean(1)
    components.update(weighted_bce=per_image.mean(), weighted_bce_per_image=per_image)
    components['total_loss'] = components['weighted_bce']+components['dice_loss']
    return components


def prepare(baseline):
    # F.prepare checks B/E provenance, original polygon mask rules, and hashes
    # existing source/config/history/checkpoint files without instantiating test.
    ds, selected, records, before = F.prepare(baseline)
    F.require(records == json.loads((F_RUN/'provenance.json').read_text())['selection'], 'Selection differs from F')
    fg = total = 0
    for index in selected:
        _, target = ds[index]
        fg += int(target.sum())
        total += target.numel()
    F.require((fg,total-fg) == (FOREGROUND,BACKGROUND), 'Training pixel counts changed')
    F.require(round((total-fg)/fg,6) == POS_WEIGHT == 5.521724, 'Training-derived weight mismatch')
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        valid = F.LunarLandslideDataset('valid', image_size=256)
        valid.pairs = [(a.resolve(),b.resolve()) for a,b in valid.pairs]
        valid.image_dir = valid.image_dir.resolve()
    finally:
        os.chdir(previous)
    source = valid.image_dir/'_annotations.coco.json'
    before[str(source)] = F.digest(source)
    metadata = json.loads(source.read_text(encoding='utf-8'))
    images = {i['file_name']:i for i in metadata['images']}
    lands = {c['id'] for c in metadata['categories'] if c['name'].lower()=='landslide'}
    F.require(len(images)==len(metadata['images'])==80 and set(images)=={a.name for a,b in valid.pairs}, 'Validation inventory mismatch')
    vr = []
    for image_path,mask_path in valid.pairs:
        before[str(image_path)],before[str(mask_path)] = F.digest(image_path),F.digest(mask_path)
    for index,(image_path,mask_path) in enumerate(valid.pairs):
        image,target = valid[index]
        info = images[image_path.name]
        annotations = [a for a in metadata['annotations'] if a['image_id']==info['id']]
        F.verify_mask_rules(cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE),info,annotations,lands)
        F.require(image.shape==(3,256,256) and target.shape==(1,256,256),'Validation shape mismatch')
        vr.append(dict(image_key='valid/'+image_path.name, coco_image_id=info['id'],
                       image_path=str(image_path),mask_path=str(mask_path),
                       image_sha256=before[str(image_path)],mask_sha256=before[str(mask_path)],
                       reference_foreground_coverage=float(target.mean()),
                       reference_nonempty=bool(target.any()), original_split='valid'))
    F.require(sum(r['reference_nonempty'] for r in vr)==19 and len(vr)==80,'Expected 19 nonempty and 61 empty validation references')
    F.verify_unchanged(before)
    return ds,selected,records,valid,vr,before


def aggregate(rows):
    names = ('bce','weighted_bce','dice_loss','total_loss','dice','iou','recall','precision',
             'predicted_foreground_coverage','reference_foreground_coverage')
    result,counts = {},{}
    for name in names:
        values=[r[name] for r in rows if r[name] is not None]
        result[name]=sum(values)/len(values) if values else None
        counts[name]=len(values)
    return dict(images=len(rows),macro_mean=result,defined_image_counts=counts,
                images_with_foreground=sum(r['predicted_foreground_coverage']>0 for r in rows))


def evaluate(model,loader,records):
    was_training = model.training
    rows=[]
    try:
        model.eval()
        with torch.inference_mode():
            for images,targets in loader:
                logits=model(images)
                components=loss_components(logits,targets)
                predictions=logits.sigmoid()>=.5
                for i,(prediction,target) in enumerate(zip(predictions,targets>0)):
                    r=records[len(rows)]
                    coverage=float(target.float().mean())
                    F.require(coverage==r['reference_foreground_coverage'],'Evaluation record association mismatch')
                    tp,fp,fn=int((prediction&target).sum()),int((prediction&~target).sum()),int((~prediction&target).sum())
                    bce=float(components['bce_per_image'][i]); wbce=float(components['weighted_bce_per_image'][i]); dice_loss=float(components['dice_loss_per_image'][i])
                    rows.append(dict(image_key=r['image_key'],coco_image_id=r['coco_image_id'],
                        bce=bce,weighted_bce=wbce,dice_loss=dice_loss,total_loss=wbce+dice_loss,
                        reference_foreground_coverage=coverage, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                        **F.metrics(tp,fp,fn,target.numel())))
        F.require(len(rows)==len(records),'Incomplete evaluation')
        return dict(label=LABEL,threshold=.5,per_image=rows,aggregate=aggregate(rows),
            nonempty=aggregate([r for r in rows if r['reference_foreground_coverage']>0]),
            empty_unverified=aggregate([r for r in rows if r['reference_foreground_coverage']==0]))
    finally:
        model.train(was_training)


def destination(run,name):
    root=EXPERIMENT/'runs'
    F.require(root.resolve()==root and Path(run).resolve().parent==root,'Output run escapes G')
    path=(Path(run)/name).resolve()
    F.require(path.parent==Path(run).resolve(),'Output path escapes run')
    return path


def write_json(run,name,value):
    with destination(run,name).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def state(model,optimizer,config,evaluation,selection):
    return copy.deepcopy(dict(label=LABEL,update=evaluation['update'],configuration=config,
        model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        evaluation=evaluation,selection=selection))


def save_verified(run,name,value):
    path=destination(run,name)
    with path.open('xb') as stream: torch.save(value,stream)
    F.verify_checkpoint(torch.load(path,map_location='cpu',weights_only=True),value)
    return F.digest(path)


class Best:
    def __init__(self,run,kind):
        F.require(kind in ('training','validation'),'Unknown checkpoint kind')
        self.run,self.kind,self.score,self.file_hash,self.update=run,kind,None,None,None

    def consider(self,model,optimizer,config,evaluation):
        report=evaluation[self.kind]
        score=report['aggregate' if self.kind=='training' else 'nonempty']['macro_mean']['dice']
        F.require(score is not None and math.isfinite(score),'Invalid selection Dice')
        if self.score is not None and score<=self.score: return False
        value=state(model,optimizer,config,evaluation,'strict_'+self.kind+'_dice_earliest_tie')
        path=destination(self.run,'best_'+self.kind+'.pth')
        if self.file_hash is None:
            F.require(not path.exists(),'Refusing pre-existing checkpoint')
        else:
            F.require(F.digest(path)==self.file_hash,'Owned best checkpoint changed externally')
        staging='staging_'+self.kind+'_'+uuid.uuid4().hex+'.tmp'
        save_verified(self.run,staging,value)
        if self.file_hash is None:
            destination(self.run,staging).rename(path)
        else:
            F.require(F.digest(path)==self.file_hash,'Checkpoint changed during staging')
            os.replace(destination(self.run,staging),path)
        self.score,self.update,self.file_hash=score,evaluation['update'],F.digest(path)
        return True


def train(config,ds,selected,records,valid,vr,before):
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config['cpu_threads'])
    training,training_eval,_=F.make_loaders(ds,selected,config['seed'])
    # Dedicated generator prevents validation iterator creation from advancing
    # the model or training sampler RNG. Validation never enters training loader.
    validation=DataLoader(valid,batch_size=2,shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(config['seed']))
    model=F.UNet().to('cpu')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    root=EXPERIMENT/'runs'
    F.require(root.resolve()==root,'Redirected run root')
    root.mkdir(exist_ok=True)
    run=root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
    run.mkdir(exist_ok=False)
    counter=F.StepBudget(optimizer)
    best_train,best_valid=Best(run,'training'),Best(run,'validation')
    start=time.perf_counter();evaluations=[];presentations=0
    try:
        write_json(run,'configuration.json',config)
        write_json(run,'provenance.json',dict(label=LABEL,training=records,validation=vr,baseline=str(F_RUN)))
        write_json(run,'protected_hashes_before.json',before)
        write_json(run,'environment.json',dict(python=sys.version,packages={n:F.importlib.metadata.version(n) for n in ('torch','numpy','opencv-python')},cpu_threads=torch.get_num_threads()))
        with destination(run,'training_history.jsonl').open('x',encoding='utf-8') as history, destination(run,'evaluation_history.jsonl').open('x',encoding='utf-8') as eh:
            def record(step):
                result=dict(update=step,successful_optimizer_steps=counter.successful,elapsed_seconds=time.perf_counter()-start,
                            training=evaluate(model,training_eval,records),validation=evaluate(model,validation,vr))
                eh.write(json.dumps(result,allow_nan=False)+'\n');eh.flush()
                best_train.consider(model,optimizer,config,result)
                best_valid.consider(model,optimizer,config,result)
                evaluations.append(step)
                print(json.dumps(dict(update=step,training=result['training']['aggregate'],validation_nonempty=result['validation']['nonempty'],validation_empty=result['validation']['empty_unverified'])),flush=True)
                return result
            final=record(0)
            for step,pass_number,batch_number,(images,targets) in F.bounded_batches(training,BUDGET):
                model.train();optimizer.zero_grad(set_to_none=True)
                components=loss_components(model(images),targets)
                loss=components['total_loss']
                F.require(bool(torch.isfinite(loss)),'Nonfinite loss')
                loss.backward()
                F.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Missing/nonfinite gradient')
                optimizer.step()
                F.require(counter.successful==step,'Successful step count mismatch')
                presentations+=len(images)
                history.write(json.dumps(dict(update=step,successful_optimizer_steps=counter.successful,pass_number=pass_number,
                    batch_in_pass=batch_number,batch_size=len(images),elapsed_seconds=time.perf_counter()-start,
                    **{k:float(components[k].detach()) for k in ('bce','weighted_bce','dice_loss','total_loss')}),allow_nan=False)+'\n');history.flush()
                if step%140==0: final=record(step)
        F.check_completion(counter,optimizer,evaluations,presentations,pass_number,batch_number)
        save_verified(run,'last.pth',state(model,optimizer,config,final,'final_update'))
        for best in (best_train,best_valid):
            p=destination(run,'best_'+best.kind+'.pth')
            F.require(F.digest(p)==best.file_hash,'Best checkpoint hash mismatch')
            loaded=torch.load(p,map_location='cpu',weights_only=True)
            F.require(loaded['update']==best.update,'Best update mismatch')
        after=F.verify_unchanged(before)
        write_json(run,'protected_hashes_after.json',after)
        write_json(run,'summary.json',dict(status='completed',label=LABEL,successful_optimizer_steps=counter.successful,
            evaluations=evaluations,complete_passes=140,presentations=presentations,elapsed_seconds=time.perf_counter()-start,
            best_training_update=best_train.update,best_training_dice=best_train.score,
            best_validation_update=best_valid.update,best_validation_dice=best_valid.score,final=final,
            checkpoint_hashes={name:F.digest(destination(run,name)) for name in ('best_training.pth','best_validation.pth','last.pth')}))
    except BaseException as exc:
        write_json(run,'incomplete.json',dict(status='incomplete',successful_optimizer_steps=counter.successful,error=repr(exc)))
        print(f'INCOMPLETE: {run}',file=sys.stderr)
        raise
    finally:
        counter.close();F.verify_unchanged(before)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only',action='store_true')
    mode.add_argument('--train',action='store_true',help='Start training only after separate user approval')
    args=parser.parse_args(argv)
    if not args.train and not args.validate_only:
        parser.print_help();return 0
    before={}
    try:
        config,baseline=configuration()
        torch.set_num_threads(2)
        ds,selected,records,valid,vr,before=prepare(baseline)
        if args.train: train(config,ds,selected,records,valid,vr,before)
        else: print(json.dumps(dict(status='validated_no_training_no_outputs',training_images=len(selected),validation_images=len(vr),foreground=FOREGROUND,background=BACKGROUND,pos_weight=POS_WEIGHT,protected_files=len(before)),indent=2))
        return 0
    except (OSError,ValueError,KeyError,RuntimeError,KeyboardInterrupt) as exc:
        print(f'SAFE STOP: {exc}',file=sys.stderr);return 1
    finally:
        F.verify_unchanged(before)


if __name__=='__main__':
    raise SystemExit(main())
