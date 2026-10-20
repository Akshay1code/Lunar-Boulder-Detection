"""Non-training checks: synthetic tensors, mocked optimizer and output writes."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_four_image as e
import numpy as np
import torch


class FakeOptimizer:
    def __init__(self):
        self.calls = 0
        self.fail = False
    def register_step_pre_hook(self, fn):
        self.pre = fn
        return Mock()
    def register_step_post_hook(self, fn):
        self.post = fn
        return Mock()
    def step(self):
        self.pre(self, (), {})
        if self.fail:
            raise RuntimeError("simulated failed step")
        self.calls += 1
        self.post(self, (), {})


class Safeguards(unittest.TestCase):
    def test_only_budget_differs_from_saved_run(self):
        baseline = e.json.loads((e.BASELINE_RUN / "configuration.json").read_text())
        config = e.load_config()
        e.verify_budget_only(config, baseline)
        for key, value in (("seed",43),("learning_rate",.0003),("threshold",.3),("evaluate_every",40)):
            changed=copy.deepcopy(config);changed[key]=value
            with self.assertRaises(ValueError):e.verify_budget_only(changed,baseline)
        changed=copy.deepcopy(config);changed["selected"].reverse()
        with self.assertRaises(ValueError):e.verify_budget_only(changed,baseline)

    def test_implementation_identical_except_budget_and_preflight(self):
        import ast
        old=ast.parse((e.ROOT/"experiments/four_image_overfit/train_four_image.py").read_text())
        new=ast.parse(Path(e.__file__).read_text())
        old_nodes={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new_nodes={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        class RestoreBudget(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is int and node.value in (800,801,200):
                    node.value={800:400,801:401,200:100}[node.value]
                elif isinstance(node.value,str):
                    node.value=node.value.replace("801","401").replace("200 updates","100 updates")
                return node
        for name,node in old_nodes.items():
            if name=="load_config":continue
            self.assertEqual(ast.dump(node),ast.dump(RestoreBudget().visit(new_nodes[name])),name)
        self.assertEqual(set(new_nodes)-set(old_nodes),{"verify_budget_only","verify_previous_experiment"})

    def test_original_prefix_schedule_and_41_evaluations(self):
        sequence=[]
        e.execute_updates(lambda step,index:sequence.append((step,index)))
        self.assertEqual(sequence[:400],[(i,(i-1)%4) for i in range(1,401)])
        evaluations=[0]+[step for step,index in sequence if step%20==0]
        self.assertEqual(evaluations,list(range(0,801,20)))
        self.assertEqual(len(evaluations),41)

    def test_selection(self):
        records = e.load_config()["selected"]
        e.validate_selection(records)
        for bad in (records[:3], records+[records[0]], list(reversed(records))):
            with self.assertRaises(ValueError):
                e.validate_selection(bad)
        for key, value in (("original_split", "test"), ("reference_nonempty", False), ("image_sha256", records[1]["image_sha256"])):
            bad = copy.deepcopy(records)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                e.validate_selection(bad)

    def test_fixed_config(self):
        config = e.load_config()
        config["max_updates"] = 801
        with patch.object(e.json, "loads", return_value=config):
            with self.assertRaises(ValueError):
                e.load_config()

    def test_forbidden_splits_never_instantiate(self):
        with patch.object(e, "LunarLandslideDataset") as loader:
            for split in ("valid", "test"):
                with self.assertRaises(ValueError):
                    e.make_dataset(split)
            loader.assert_not_called()

    def test_annotation_association(self):
        record = dict(coco_image_id=1, filename="a.jpg", annotation_ids=[3,4], landslide_annotation_ids=[3])
        info = dict(id=1, file_name="a.jpg")
        anns = [dict(id=3, image_id=1, category_id=7), dict(id=4, image_id=1, category_id=8)]
        e.verify_association(record, info, anns, {7})
        for bad in (dict(info, id=2), dict(info, file_name="wrong.jpg")):
            with self.assertRaises(ValueError):
                e.verify_association(record, bad, anns, {7})
        with self.assertRaises(ValueError):
            e.verify_association(record, info, anns, {8})
        anns[0]["image_id"] = 2
        with self.assertRaises(ValueError):
            e.verify_association(record, info, anns, {7})

    def test_polygon_rounding_fill_and_category_filter(self):
        info = dict(height=5, width=5)
        anns = [dict(category_id=7, segmentation=[[.6,.6,2.4,.6,2.4,2.4,.6,2.4]]),
                dict(category_id=8, segmentation=[[0,0,4,0,4,4,0,4]])]
        mask = np.zeros((5,5), dtype=np.uint8)
        mask[1:3,1:3] = 255
        e.verify_mask_rules(mask, info, anns, {7})
        bad = mask.copy(); bad[0,0] = 255
        with self.assertRaises(ValueError):
            e.verify_mask_rules(bad, info, anns, {7})
        with self.assertRaises(ValueError):
            e.verify_mask_rules(np.zeros_like(mask), info, anns, {7})

    def test_metrics(self):
        target = torch.tensor([0.,1.,1.,0.])
        empty = torch.zeros(4, dtype=torch.bool)
        m = e.metrics(empty, target)
        self.assertEqual((m["dice"],m["recall"],m["predicted_foreground_coverage"]), (0,0,0))
        self.assertIsNone(m["precision"])
        m = e.metrics(torch.ones(4, dtype=torch.bool), target)
        self.assertAlmostEqual(m["dice"], 2/3)
        self.assertEqual((m["recall"], m["precision"], m["reference_foreground_coverage"]), (1,.5,.5))
        m = e.metrics(target.bool(), target)
        self.assertEqual((m["dice"],m["recall"],m["precision"]), (1,1,1))
        self.assertIsNone(e.metrics(empty, torch.zeros(4))["recall"])

    def test_aggregate_undefined_precision(self):
        row = dict(bce=.2, **e.metrics(torch.zeros(2,dtype=torch.bool),torch.tensor([0.,1.])))
        result = e.aggregate([row]*4)
        self.assertIsNone(result["macro_mean"]["precision"])
        self.assertEqual(result["defined_image_counts"]["precision"], 0)
        positive = dict(bce=.2, **e.metrics(torch.tensor([False,True]),torch.tensor([0.,1.])))
        result = e.aggregate([row]*3+[positive])
        self.assertEqual(result["macro_mean"]["precision"], 1)
        self.assertEqual(result["defined_image_counts"]["precision"], 1)

    def test_inference_mode_and_half_threshold(self):
        class Model:
            def eval(self): pass
            def __call__(self, x):
                if not torch.is_inference_mode_enabled(): raise AssertionError("Expected inference mode")
                return x
        tensors = [(torch.tensor([[[[-1.,0.]]]]),torch.tensor([[[[0.,1.]]]]))]*4
        records = [dict(image_key=str(i),coco_image_id=i) for i in range(4)]
        result = e.evaluate(Model(), tensors, records)
        self.assertEqual(result["aggregate"]["macro_mean"]["dice"], 1)
        self.assertEqual(result["threshold"], .5)

    def test_800_successes_and_block_801(self):
        optimizer = FakeOptimizer(); counter = e.StepBudget(optimizer)
        counts = [0]*4; steps = []
        def update(step, index):
            optimizer.step(); counts[index] += 1; steps.append(step)
        e.execute_updates(update)
        self.assertEqual(steps, list(range(1,801)))
        self.assertEqual(counts,[200]*4)
        self.assertEqual((optimizer.calls,counter.successful),(800,800))
        with self.assertRaises(ValueError): optimizer.step()
        self.assertEqual(optimizer.calls,800)
        counter.close()

    def test_failed_steps_not_counted(self):
        optimizer = FakeOptimizer(); counter = e.StepBudget(optimizer); optimizer.fail = True
        with self.assertRaises(RuntimeError): optimizer.step()
        self.assertEqual(counter.successful,0)
        counter.close()

    def test_completion_invariants(self):
        p=object(); optimizer=Mock(param_groups=[{"params":[p]}],state={p:{"step":800}})
        counter=Mock(successful=800)
        e.check_completion(counter,optimizer,list(range(0,801,20)),[200]*4)
        for successes,history,counts in ((799,list(range(0,801,20)),[200]*4),(800,[0,800],[200]*4),(800,list(range(0,801,20)),[199,201,200,200])):
            counter.successful=successes
            with self.assertRaises(ValueError): e.check_completion(counter,optimizer,history,counts)
        counter.successful=800;optimizer.state[p]["step"]=799
        with self.assertRaises(ValueError): e.check_completion(counter,optimizer,list(range(0,801,20)),[200]*4)

    def test_interruption_stops_iteration(self):
        seen=[]
        def update(step,index):
            if step==8: raise KeyboardInterrupt()
            seen.append(step)
        with self.assertRaises(KeyboardInterrupt): e.execute_updates(update)
        self.assertEqual(seen,list(range(1,8)))

    def test_interrupted_run_no_completion(self):
        optimizer=FakeOptimizer()
        with patch.object(e,"UNet"), patch.object(e.torch.optim,"Adam",return_value=optimizer), patch.object(e,"create_run",return_value=e.EXPERIMENT/"runs/mock"), patch.object(e,"write_json") as write, patch.object(e,"save_checkpoint") as save, patch.object(Path,"open",return_value=io.StringIO()), patch.object(e.importlib.metadata,"version",return_value="mock"), patch.object(e,"evaluate",side_effect=KeyboardInterrupt()), patch.object(e,"verify_unchanged") as verify, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt): e.train({},[],[],{})
            self.assertEqual(optimizer.calls,0)
            self.assertNotIn("summary.json",[c.args[1] for c in write.call_args_list])
            save.assert_not_called();verify.assert_called_once_with({})

    def test_output_escape_and_overwrite_protection(self):
        run=e.EXPERIMENT/"runs/mock"
        for name in ("../../../../checkpoints/unet_best.pth",str(e.ROOT/"checkpoints/unet_best.pth")):
            with self.assertRaises(ValueError): e.output_path(run,name)
        with patch.object(Path,"open",side_effect=FileExistsError()) as opened, patch.object(e.torch,"save") as save:
            with self.assertRaises(FileExistsError): e.save_checkpoint(run,{})
            self.assertEqual(opened.call_args.args[0],"xb");save.assert_not_called()
        with patch.object(Path,"mkdir",side_effect=[None,FileExistsError()]) as mkdir:
            with self.assertRaises(FileExistsError): e.create_run()
            self.assertEqual(mkdir.call_args.kwargs,{"exist_ok":False})

    def test_changed_or_missing_hash_fails(self):
        for value in ("different",OSError("missing")):
            with patch.object(e,"digest",**({"side_effect":value} if isinstance(value,Exception) else {"return_value":value})):
                with self.assertRaises(ValueError): e.verify_unchanged({"original":"hash"})

    def test_default_does_nothing(self):
        with patch.object(e,"prepare") as prepare,patch.object(e,"train") as train,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.main([]),0);prepare.assert_not_called();train.assert_not_called()

    def test_validate_only_no_training_or_output(self):
        with patch.object(e,"prepare",return_value=([],[],{})),patch.object(e,"verify_unchanged"),patch.object(e,"train",side_effect=AssertionError("training forbidden")),patch.object(e,"UNet",side_effect=AssertionError("model forbidden")),patch.object(e,"create_run",side_effect=AssertionError("write forbidden")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.main(["--validate-only"]),0)

    def test_failed_preflight_never_trains(self):
        with patch.object(e,"prepare",side_effect=ValueError("bad mask")),patch.object(e,"train") as train,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(e.main(["--train"]),1);train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
