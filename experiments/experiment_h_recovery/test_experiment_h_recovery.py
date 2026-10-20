"""Non-training tests. No filesystem writes or real optimizer updates."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock
sys.dont_write_bytecode = True
import train_experiment_h_recovery as e


class FakeOptimizer:
    def __init__(self):
        self.pre, self.post = [], []
        self.param_groups = [{"params": ["parameter"]}]
        self.state = {"parameter": {"step": 0}}

    def register_step_pre_hook(self, fn):
        self.pre.append(fn)
        return Mock(remove=lambda: self.pre.remove(fn))

    def register_step_post_hook(self, fn):
        self.post.append(fn)
        return Mock(remove=lambda: self.post.remove(fn))

    def step(self, fail=False):
        for hook in self.pre:
            hook(self, (), {})
        if fail:
            raise KeyboardInterrupt()
        self.state["parameter"]["step"] += 1
        for hook in self.post:
            hook(self, (), {})


class InfrastructureTests(unittest.TestCase):
    def test_config_difference(self):
        c = e.load_config()
        self.assertEqual(c, e.expected_configuration())
        self.assertEqual(c["max_updates"], 5180)
        b = json.loads((e.BASELINE / "configuration.json").read_text())
        for k in ("seed", "device", "cpu_threads", "image_size", "batch_size", "drop_last", "num_workers",
                  "model", "initialization", "optimizer", "learning_rate", "pos_weight", "threshold"):
            self.assertEqual(c[k], b[k])

    def test_selection_matches_manifest(self):
        records = json.loads((e.BASELINE / "data_manifest.json").read_text())["splits"]["train"]
        selected = e.checked_selection([r["reference_nonempty"] for r in records], 279, 73, "train")
        e.verify_records(records, copy.deepcopy(records), selected)
        with self.assertRaises(ValueError):
            e.verify_records(records, records, list(reversed(selected)))
        changed = copy.deepcopy(records)
        changed[0]["mask_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            e.verify_records(changed, records, selected)

    def test_count_failure(self):
        with self.assertRaises(ValueError):
            e.checked_selection([True]*72+[False]*207, 279, 73, "train")

    def test_training_only_gate(self):
        with patch.object(e, "LunarLandslideDataset") as loader:
            for split in ("valid", "test"):
                with self.assertRaises(ValueError):
                    e.make_dataset(split, 256)
            loader.assert_not_called()

    def test_coco_mask_association(self):
        mask = e.np.zeros((8, 8), dtype=e.np.uint8)
        mask[2:6, 1:5] = 255
        annotation = {"category_id": 1, "segmentation": [[1,2,4,2,4,5,1,5]]}
        e.verify_mask_rules(mask, {"height": 8, "width": 8}, [annotation], {1})
        with self.assertRaises(ValueError):
            e.verify_mask_rules(mask, {"height": 8, "width": 8}, [], {1})

    def test_metrics(self):
        m = e.metrics(2, 1, 2, 10)
        self.assertAlmostEqual(m["dice"], 4/7)
        self.assertEqual(m["recall"], .5)
        self.assertAlmostEqual(m["precision"], 2/3)
        self.assertEqual(m["predicted_foreground_coverage"], .3)
        self.assertEqual(e.metrics(0, 0, 4, 10)["dice"], 0)
        self.assertIsNone(e.metrics(0, 0, 4, 10)["precision"])
        self.assertIsNone(e.metrics(0, 0, 0, 10)["dice"])

    def test_empty_unselected_mask_is_not_an_error(self):
        mask = e.np.zeros((8, 8), dtype=e.np.uint8)
        e.verify_mask_rules(mask, {"height": 8, "width": 8}, [], {1})
        self.assertEqual(e.checked_selection([True]*73+[False]*206, 279, 73, "train"), list(range(73)))

    def test_groups(self):
        self.assertEqual([e.coverage_group(x) for x in (.049, .05, .149, .15, .299, .30)],
                         ["lt_5_percent", "5_to_lt_15_percent", "5_to_lt_15_percent",
                          "15_to_lt_30_percent", "15_to_lt_30_percent", "ge_30_percent"])
        self.assertIsNone(e.aggregate([])["macro_mean"]["precision"])
        with self.assertRaises(ValueError):
            e.coverage_group(0)

    def test_evaluation_threshold_and_counts(self):
        targets = e.torch.ones(73, 1, 2, 2)
        images = e.torch.zeros(73, 3, 2, 2)
        records = [dict(image_key=str(i), coco_image_id=i, reference_foreground_coverage=1.,
                        coverage_group="ge_30_percent") for i in range(73)]
        class Model:
            def eval(self): pass
            def __call__(self, x): return e.torch.zeros(len(x), 1, 2, 2)
        result = e.evaluate(Model(), [(images, targets)], records)
        self.assertEqual(result["aggregate"]["macro_mean"]["dice"], 1.)
        self.assertEqual(result["aggregate"]["images"], 73)
        self.assertEqual(result["coverage_groups"]["ge_30_percent"]["images"], 73)

    def test_sampling_identical_to_b_and_evaluation_independent(self):
        ds = e.torch.utils.data.TensorDataset(e.torch.arange(279))
        selected = list(range(73))
        actual, evaluation, _ = e.make_loaders(ds, selected, 42)
        baseline = e.DataLoader(e.Subset(ds, selected), batch_size=2, shuffle=True,
            generator=e.torch.Generator().manual_seed(42), num_workers=0, drop_last=False)
        for _ in range(3):
            list(evaluation)
            self.assertEqual([x[0].tolist() for x in actual], [x[0].tolist() for x in baseline])

    def test_full_exposure_and_limit(self):
        ds = e.torch.utils.data.TensorDataset(e.torch.arange(73))
        loader, _, _ = e.make_loaders(ds, list(range(73)), 42)
        counts = [0]*73
        opt = FakeOptimizer(); counter = e.StepBudget(opt)
        evaluations = list(range(0, 5181, 140))
        for step, epoch, batch, values in e.bounded_batches(loader, 5180):
            opt.step()
            for i in values[0].tolist(): counts[i] += 1
        self.assertEqual(counts, [140]*73)
        e.check_completion(counter, opt, evaluations, sum(counts), epoch, batch)
        with self.assertRaises(ValueError): opt.step()
        self.assertEqual(counter.successful, 5180)
        self.assertEqual(opt.state["parameter"]["step"], 5180)
        counter.close()

    def test_failed_step_not_counted(self):
        opt = FakeOptimizer(); counter = e.StepBudget(opt)
        with self.assertRaises(KeyboardInterrupt): opt.step(fail=True)
        self.assertEqual(counter.successful, 0)
        with self.assertRaises(ValueError):
            e.check_completion(counter, opt, [], 0, 0, 0)
        counter.close()

    def test_adam_counter_mismatch(self):
        opt = FakeOptimizer(); counter = e.StepBudget(opt); counter.successful = 5180
        with self.assertRaises(ValueError):
            e.check_completion(counter, opt, list(range(0, 5181, 140)), 73*140, 140, 37)
        counter.close()

    def test_best_strict_improvement_and_snapshot(self):
        model = e.torch.nn.Linear(1, 1)
        opt = Mock(state_dict=lambda: {})
        best = e.BestCheckpoint(e.EXPERIMENT / "runs" / "fake")
        published = []
        def publish(state): published.append(state); return "hash"
        with patch.object(best, "publish", side_effect=publish):
            for step, dice in [(0,0.), (140,0.), (280,.8), (420,.7)]:
                best.consider(model, opt, {}, dict(update=step, aggregate={"macro_mean":{"dice":dice}}))
        self.assertEqual([s["update"] for s in published], [0,280])
        saved = published[-1]["model_state_dict"]["weight"].clone()
        with e.torch.no_grad(): model.weight.add_(1)
        self.assertTrue(e.torch.equal(saved, published[-1]["model_state_dict"]["weight"]))
        self.assertEqual(best.evaluation["update"], 280)
        buffer = io.BytesIO(); e.torch.save(published[-1], buffer); buffer.seek(0)
        restored = e.torch.load(buffer, weights_only=True)
        e.verify_checkpoint(restored, published[-1])
        restored["model_state_dict"]["weight"].add_(1)
        with self.assertRaises(ValueError): e.verify_checkpoint(restored, published[-1])

    def test_failed_publication_preserves_best(self):
        best = e.BestCheckpoint(e.EXPERIMENT / "runs" / "fake")
        with patch.object(best, "publish", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                best.consider(Mock(state_dict=lambda:{}), Mock(state_dict=lambda:{}), {},
                    dict(update=0,aggregate={"macro_mean":{"dice":0.}}))
        self.assertIsNone(best.evaluation)

    def test_existing_best_rejected(self):
        best = e.BestCheckpoint(e.EXPERIMENT / "runs" / "fake")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError): best.publish({"update":0})

    def test_changed_best_rejected(self):
        best = e.BestCheckpoint(e.EXPERIMENT / "runs" / "fake"); best.file_hash = "old"
        with patch.object(Path, "is_file", return_value=True), patch.object(e, "digest", return_value="changed"):
            with self.assertRaises(ValueError): best.publish({"update":0})

    def test_output_escape(self):
        run = e.EXPERIMENT / "runs" / "fake"
        self.assertEqual(e.output_path(run, "last.pth"), run / "last.pth")
        for destination in ("../other.pth", str(e.ORIGINAL_CHECKPOINT)):
            with self.assertRaises(ValueError): e.output_path(run, destination)
        with self.assertRaises(ValueError): e.output_path(e.BASELINE, "last.pth")

    def test_exclusive_output_modes(self):
        run = e.EXPERIMENT / "runs" / "fake"
        for fn, args, mode in [(e.write_json,(run,"summary.json",{}),"x"), (e.save_checkpoint,(run,{}),"xb")]:
            with patch.object(Path,"open",side_effect=FileExistsError) as opening:
                with self.assertRaises(FileExistsError): fn(*args)
                self.assertEqual(opening.call_args.args[0],mode)

    def test_unique_run_refuses_collision(self):
        with patch.object(Path, "mkdir") as mkdir:
            run1, run2 = e.create_run(), e.create_run()
            self.assertNotEqual(run1, run2)
            self.assertFalse(mkdir.call_args.kwargs["exist_ok"])

    def test_default_no_work(self):
        with patch.object(e, "prepare") as prep, patch.object(e, "train") as train, patch("sys.stdout",new=io.StringIO()):
            self.assertEqual(e.main([]),0)
            prep.assert_not_called(); train.assert_not_called()

    def test_validate_only_no_training_or_outputs(self):
        records=[dict(coco_image_id=i,coverage_group="lt_5_percent",reference_foreground_coverage=.01) for i in range(73)]
        with patch.object(e,"load_config",return_value={"cpu_threads":2}), \
             patch.object(e,"prepare",return_value=(None,list(range(73)),records,{})), \
             patch.object(e,"train") as training, patch.object(e,"create_run") as create, \
             patch.object(e,"UNet") as model, patch("sys.stdout",new=io.StringIO()):
            self.assertEqual(e.main(["--validate-only"]),0)
            training.assert_not_called(); create.assert_not_called(); model.assert_not_called()

    def test_interrupted_run_has_no_completion_summary(self):
        opt=FakeOptimizer()
        model=Mock(); model.to.return_value=model
        with patch.object(e,"make_loaders",return_value=([],[],None)), \
             patch.object(e,"UNet",return_value=model), patch.object(e.torch.optim,"Adam",return_value=opt), \
             patch.object(e,"create_run",return_value=e.EXPERIMENT/"runs"/"fake"), \
             patch.object(e,"write_json",side_effect=KeyboardInterrupt) as writes, \
             patch.object(e,"verify_unchanged") as verify, patch("sys.stderr",new=io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                e.train({"seed":42,"cpu_threads":2,"learning_rate":.001},None,[],[],{})
            self.assertFalse(any(call.args[1]=="summary.json" for call in writes.call_args_list))
            verify.assert_called(); self.assertEqual(opt.state["parameter"]["step"],0)

    def test_no_training_script_imports(self):
        tree=ast.parse(Path(e.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn("train_unet",imports)
        self.assertNotIn("train_experiment_b",imports)


    def test_exact_e_configuration_difference(self):
        baseline = json.loads((e.E_BASELINE / "configuration.json").read_text())
        config = e.load_config()
        differences = {k for k in config.keys() | baseline.keys() if config.get(k) != baseline.get(k)}
        self.assertEqual(differences, {"experiment", "loss", "loss_parameters", "experiment_e_sha256", "augmentation", "experiment_f_sha256", "recovery", "experiment_h_sha256"})

    def test_unchanged_baseline_functions(self):
        previous = ast.parse((e.E_BASELINE.parents[1] / "train_experiment_e.py").read_text())
        current = ast.parse(Path(e.__file__).read_text())
        names = ("make_loaders", "bounded_batches", "StepBudget", "check_completion", "metrics",
                 "checkpoint_state", "verify_checkpoint", "BestCheckpoint", "create_run", "output_path",
                 "make_dataset", "inspect_dataset", "verify_records", "coverage_group")
        for name in names:
            a = next(n for n in previous.body if getattr(n,"name",None)==name)
            b = next(n for n in current.body if getattr(n,"name",None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)

    def test_loss_formula(self):
        z=e.torch.zeros(1,1,2,2,requires_grad=True)
        y=e.torch.tensor([[[[1.,0.],[1.,0.]]]])
        c=e.loss_components(z,y)
        self.assertAlmostEqual(c["bce"].item(), e.math.log(2),places=6)
        self.assertAlmostEqual(c["dice_loss"].item(),.4,places=6)
        self.assertAlmostEqual(c["total_loss"].item(),e.math.log(2)+.4,places=6)

    def test_loss_gradients(self):
        z=e.torch.zeros(2,1,2,2,requires_grad=True)
        y=e.torch.tensor([[[[1.,0.],[0.,1.]]],[[[0.,1.],[1.,0.]]]])
        e.loss_components(z,y)["total_loss"].backward()
        self.assertTrue(e.torch.isfinite(z.grad).all())
        self.assertTrue((z.grad[y==1]<0).all())
        self.assertTrue((z.grad[y==0]>0).all())

    def test_dice_gradient_finite_difference(self):
        z=e.torch.zeros(1,1,2,2,requires_grad=True)
        y=e.torch.tensor([[[[1.,0.],[0.,1.]]]])
        e.loss_components(z,y)["dice_loss"].backward()
        plus=z.detach().clone();minus=z.detach().clone()
        plus[0,0,0,0]+=.01;minus[0,0,0,0]-=.01
        numerical=(e.loss_components(plus,y)["dice_loss"]-e.loss_components(minus,y)["dice_loss"])/.02
        self.assertAlmostEqual(z.grad[0,0,0,0].item(),numerical.item(),places=5)

    def test_empty_reference_loss(self):
        y=e.torch.zeros(1,1,2,2);z=e.torch.zeros_like(y,requires_grad=True)
        c=e.loss_components(z,y)
        self.assertAlmostEqual(c["dice_loss"].item(),2/3,places=6)
        c["total_loss"].backward()
        self.assertTrue((z.grad>0).all())
        self.assertAlmostEqual(e.loss_components(e.torch.full_like(y,-1000),y)["dice_loss"].item(),0.)
        self.assertIsNone(e.metrics(0,0,0,4)["recall"])
        self.assertIsNone(e.metrics(0,0,0,4)["precision"])

    def test_loss_extreme_logits(self):
        for target in (0.,1.):
            for value in (-1000.,1000.):
                z=e.torch.full((1,1,2,2),value,requires_grad=True)
                c=e.loss_components(z,e.torch.full_like(z,target))
                self.assertTrue(e.torch.isfinite(c["total_loss"]))
                c["total_loss"].backward()
                self.assertTrue(e.torch.isfinite(z.grad).all())

    def test_per_image_not_pooled_dice(self):
        z=e.torch.zeros(2,1,2,2)
        y=e.torch.stack([e.torch.zeros(1,2,2),e.torch.ones(1,2,2)])
        c=e.loss_components(z,y)
        expected=((1-1/3)+(1-5/7))/2
        self.assertAlmostEqual(c["dice_loss"].item(),expected,places=6)
        self.assertNotAlmostEqual(c["dice_loss"].item(),1-5/9,places=4)
        self.assertAlmostEqual(c["total_loss"].item(),sum(e.loss_components(z[i:i+1],y[i:i+1])["total_loss"].item() for i in range(2))/2,places=6)

    def test_invalid_loss_inputs(self):
        z=e.torch.zeros(1,1,2,2)
        with self.assertRaises(ValueError): e.loss_components(z,e.torch.full_like(z,.5))
        with self.assertRaises(ValueError): e.loss_components(z,e.torch.zeros(2,1,2,2))
        with self.assertRaises(ValueError): e.loss_components(e.torch.full_like(z,float("nan")),z)

    def test_component_evaluation_aggregation(self):
        targets=e.torch.ones(73,1,2,2);images=e.torch.zeros(73,3,2,2)
        records=[dict(image_key=str(i),coco_image_id=i,reference_foreground_coverage=1.,coverage_group="ge_30_percent") for i in range(73)]
        class Model:
            def eval(self): pass
            def __call__(self,x): return e.torch.zeros(len(x),1,2,2)
        result=e.evaluate(Model(),[(images,targets)],records)
        for r in result["per_image"]:
            self.assertAlmostEqual(r["dice_loss"],2/7,places=6)
            self.assertAlmostEqual(r["total_loss"],r["bce"]+r["dice_loss"],places=6)
        self.assertAlmostEqual(result["aggregate"]["macro_mean"]["dice_loss"],2/7,places=6)
        self.assertEqual(result["coverage_groups"]["ge_30_percent"],result["aggregate"])

    def test_missing_final_evaluation_rejected(self):
        opt=FakeOptimizer();counter=e.StepBudget(opt);counter.successful=5180
        opt.state["parameter"]["step"]=5180
        with self.assertRaises(ValueError):
            e.check_completion(counter,opt,list(range(0,5180,140)),10220,140,37)
        counter.close()


    def test_exact_f_difference(self):
        baseline=json.loads((e.ROOT/'experiments/experiment_f_dice_loss/config.json').read_text())
        current=e.load_config()
        self.assertEqual({k for k in current.keys() | baseline.keys() if current.get(k)!=baseline.get(k)},
                         {'experiment','augmentation','experiment_f_sha256','recovery','experiment_h_sha256'})

    def test_joint_flip_alignment_and_no_source_mutation(self):
        x=e.torch.arange(24.).reshape(2,3,2,2)
        y=x[:,:1].clone(); original=x.clone()
        for p in (0.,1.):
            a,b,flags=e.horizontal_flip_batch(x,y,e.torch.Generator().manual_seed(42),p)
            self.assertTrue(e.torch.equal(a, x if p==0 else x.flip(-1)))
            self.assertTrue(e.torch.equal(b, y if p==0 else y.flip(-1)))
            self.assertEqual(flags,[bool(p)]*2)
        self.assertTrue(e.torch.equal(x,original))

    def test_augmentation_rng_isolated_and_repeatable(self):
        state=e.torch.get_rng_state().clone()
        a=e.torch.Generator().manual_seed(42);b=e.torch.Generator().manual_seed(42)
        x=e.torch.zeros(100,3,2,2);y=x[:,:1]
        f=e.horizontal_flip_batch(x,y,a)[2]
        self.assertEqual(f,e.horizontal_flip_batch(x,y,b)[2])
        self.assertTrue(any(f) and not all(f))
        self.assertTrue(e.torch.equal(state,e.torch.get_rng_state()))

    def test_f_evaluation_and_selection_unchanged(self):
        previous=ast.parse((e.ROOT/'experiments/experiment_f_dice_loss/train_experiment_f.py').read_text())
        current=ast.parse(Path(e.__file__).read_text())
        for name in ('make_loaders','loss_components','BestCheckpoint','StepBudget','check_completion','prepare'):
            a=next(n for n in previous.body if getattr(n,'name',None)==name)
            b=next(n for n in current.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)


    def test_sampler_exact_all_140_passes_and_resume(self):
        sampler=e.BatchStream(42)
        baseline=e.DataLoader(list(range(73)),batch_size=2,shuffle=True,
            generator=e.torch.Generator().manual_seed(42),num_workers=0,drop_last=False)
        count=0
        for epoch in range(140):
            for batch in baseline:
                count+=1
                self.assertEqual(sampler.next_indices(),batch.tolist())
                if count in (1,37,140,5180):
                    buffer=io.BytesIO();e.torch.save(sampler.state(),buffer);buffer.seek(0)
                    restored=e.BatchStream(999);restored.restore(e.torch.load(buffer,weights_only=True));sampler=restored
        self.assertEqual(count,5180)
        self.assertEqual(sampler.pass_number,140)
        self.assertEqual(sampler.position,37)

    def test_recovery_rng_and_next_outputs_exact_no_optimizer_steps(self):
        e.random.seed(42);e.np.random.seed(42);e.torch.manual_seed(42)
        model=e.torch.nn.Conv2d(3,1,1);optimizer=e.torch.optim.Adam(model.parameters(),lr=.001)
        # Populate synthetic Adam state without step(), backward(), or training.
        for p in model.parameters():
            optimizer.state[p]={'step':e.torch.tensor(140.),'exp_avg':e.torch.zeros_like(p),'exp_avg_sq':e.torch.zeros_like(p)}
        sampler=e.BatchStream(42);aug=e.torch.Generator().manual_seed(42);ev=e.torch.Generator().manual_seed(42)
        for _ in range(140):
            n=len(sampler.next_indices());e.torch.rand(n,generator=aug)
        state=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state(),rng=e.capture_rng(aug,ev)))
        buffer=io.BytesIO();e.torch.save(state,buffer);buffer.seek(0);state=e.torch.load(buffer,weights_only=True)
        def next_values(model,sampler,aug):
            ids=sampler.next_indices();x=e.torch.arange(len(ids)*12.).reshape(len(ids),3,2,2);y=x[:,:1]
            x,y,flags=e.horizontal_flip_batch(x,y,aug)
            with e.torch.inference_mode():out=model(x).clone()
            return ids,flags,out,e.random.random(),float(e.np.random.rand()),e.torch.rand(3)
        expected=next_values(model,sampler,aug)
        resumed=e.torch.nn.Conv2d(3,1,1);opt=e.torch.optim.Adam(resumed.parameters(),lr=.001)
        resumed.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        stream=e.BatchStream(9);stream.restore(state['sampler']);a=e.torch.Generator();v=e.torch.Generator()
        e.restore_rng(state['rng'],a,v);actual=next_values(resumed,stream,a)
        for lhs,rhs in zip(expected,actual):
            if isinstance(lhs,e.torch.Tensor):self.assertTrue(e.torch.equal(lhs,rhs))
            else:self.assertEqual(lhs,rhs)
        e.verify_checkpoint(opt.state_dict(),state['optimizer'])
        self.assertTrue(e.torch.equal(v.get_state(),ev.get_state()))

    def test_probability_moments(self):
        p=e.torch.tensor([0.,.25,.5,.75,1.])
        m=e.ProbabilityMoments();m.add(p[:2]);m.add(p[2:]);actual=m.result()
        self.assertEqual(actual['min'],0.);self.assertEqual(actual['max'],1.)
        self.assertAlmostEqual(actual['mean'],.5)
        self.assertAlmostEqual(actual['std'],float(p.double().std(unbiased=False)))

    def test_gradient_norms_no_training(self):
        model=e.torch.nn.Linear(2,1)
        model.weight.grad=e.torch.tensor([[3.,4.]]);model.bias.grad=e.torch.tensor([12.])
        stats=e.gradient_statistics(model)
        self.assertEqual(stats['global_l2'],13.)
        self.assertEqual(stats['parameter_l2_min'],5.)
        self.assertEqual(stats['parameter_l2_max'],12.)
        self.assertEqual(stats['parameter_l2_mean'],8.5)
        model.bias.grad.fill_(float('nan'))
        with self.assertRaises(ValueError):e.gradient_statistics(model)

    def test_recovery_output_refuses_overwrite(self):
        with patch.object(Path,'exists',return_value=True),patch.object(e.torch,'save') as save:
            with self.assertRaises(ValueError):e.save_recovery(e.EXPERIMENT/'runs'/'fake',{'update':140})
            save.assert_not_called()
        with self.assertRaises(ValueError):e.output_path(e.ROOT/'experiments/experiment_h_horizontal_flip/runs/fake','x')

    def test_sampler_corruption_rejected(self):
        s=e.BatchStream(42);s.next_indices();state=s.state();state['order'][0]=[0,0]
        with self.assertRaises(ValueError):e.BatchStream(42).restore(state)

    def test_resume_requires_train(self):
        with patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit):e.main(['--resume','fake.pth'])

    def test_unmodified_core_and_only_diagnostic_config_changes(self):
        baseline=json.loads((e.ROOT/'experiments/experiment_h_horizontal_flip/config.json').read_text());current=e.load_config()
        self.assertEqual({k for k in current.keys()|baseline.keys() if current.get(k)!=baseline.get(k)},
                         {'experiment','recovery','experiment_h_sha256'})


if __name__ == "__main__":
    e.torch.set_num_threads(2)
    unittest.main(verbosity=2)
