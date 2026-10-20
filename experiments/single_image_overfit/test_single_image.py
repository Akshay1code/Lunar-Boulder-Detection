"""Non-training tests; no real optimizer steps, filesystem writes, or split reads."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_single_image as experiment
import torch


class FakeOptimizer:
    """Emulate step hook dispatch, without a model or parameter updates."""
    def __init__(self):
        self.pre = self.post = None
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
            raise RuntimeError("simulated optimizer failure")
        self.calls += 1
        self.post(self, (), {})


class Safeguards(unittest.TestCase):
    def test_exact_budget_and_201_blocked(self):
        optimizer = FakeOptimizer()
        counter = experiment.StepBudget(optimizer)
        steps = []
        def update(step):
            optimizer.step()
            steps.append(step)
        experiment.execute_updates(update)
        self.assertEqual(steps, list(range(1, 201)))
        self.assertEqual((optimizer.calls, counter.successful), (200, 200))
        with self.assertRaises(ValueError):
            optimizer.step()
        self.assertEqual((optimizer.calls, counter.successful), (200, 200))
        counter.close()

    def test_failed_step_not_counted(self):
        optimizer = FakeOptimizer()
        counter = experiment.StepBudget(optimizer)
        optimizer.fail = True
        with self.assertRaises(RuntimeError):
            optimizer.step()
        self.assertEqual(counter.successful, 0)
        counter.close()

    def test_interruption_does_not_continue_budget(self):
        seen = []
        def update(step):
            if step == 8:
                raise KeyboardInterrupt()
            seen.append(step)
        with self.assertRaises(KeyboardInterrupt):
            experiment.execute_updates(update)
        self.assertEqual(seen, list(range(1, 8)))

    def test_completion_requires_both_counters_and_history(self):
        parameter = object()
        optimizer = Mock(param_groups=[{"params": [parameter]}], state={parameter: {"step": 200}})
        counter = Mock(successful=200)
        experiment.check_completion(counter, optimizer, list(range(201)))
        for count in (0, 199, 201):
            counter.successful = count
            with self.assertRaises(ValueError):
                experiment.check_completion(counter, optimizer, list(range(201)))
        counter.successful = 200
        with self.assertRaises(ValueError):
            experiment.check_completion(counter, optimizer, list(range(200)))
        optimizer.state[parameter]["step"] = 199
        with self.assertRaises(ValueError):
            experiment.check_completion(counter, optimizer, list(range(201)))

    def test_selection_requires_exactly_one_fixed_image(self):
        name = experiment.IMAGE_KEY.split("/", 1)[1]
        pair = (Path(name), Path("mask.png"))
        self.assertEqual(experiment.selected_index([pair]), 0)
        for pairs in ([], [pair, pair]):
            with self.assertRaises(ValueError):
                experiment.selected_index(pairs)

    def test_nontraining_splits_rejected_before_loader(self):
        with patch.object(experiment, "LunarLandslideDataset") as loader:
            for split in ("valid", "test", "validation"):
                with self.assertRaises(ValueError):
                    experiment.make_dataset(split)
            loader.assert_not_called()

    def test_fixed_config(self):
        self.assertEqual(experiment.load_config(), experiment.FIXED)
        changed = dict(experiment.FIXED, max_updates=201)
        with patch.object(experiment.json, "loads", return_value=changed):
            with self.assertRaises(ValueError):
                experiment.load_config()

    def test_metric_edges(self):
        target = torch.tensor([0., 1., 1., 0.])
        empty = torch.zeros(4, dtype=torch.bool)
        m = experiment.metrics(empty, target)
        self.assertEqual((m["dice"], m["recall"], m["foreground_coverage"]), (0, 0, 0))
        m = experiment.metrics(target.bool(), target)
        self.assertEqual((m["dice"], m["recall"]), (1, 1))
        m = experiment.metrics(torch.ones(4, dtype=torch.bool), target)
        self.assertAlmostEqual(m["dice"], 2/3)
        self.assertEqual(m["recall"], 1)
        m = experiment.metrics(empty, torch.zeros(4))
        self.assertIsNone(m["dice"])
        self.assertIsNone(m["recall"])

    def test_evaluation_is_inference_only_at_half(self):
        class Model:
            def eval(self):
                pass
            def __call__(self, x):
                if not torch.is_inference_mode_enabled():
                    raise AssertionError("Inference mode required")
                return x
        result = experiment.evaluate(Model(), torch.tensor([[[[-1., 0.]]]]), torch.tensor([[[[0., 1.]]]]))
        self.assertEqual(result["dice"], 1)
        self.assertEqual(result["foreground_coverage"], .5)
        self.assertEqual(result["threshold"], .5)
        self.assertIn("Training-image fit", result["label"])

    def test_path_escape_and_existing_file_protection(self):
        run = experiment.EXPERIMENT / "runs/mock"
        self.assertEqual(experiment.output_path(run, "last.pth"), (run / "last.pth").resolve())
        for name in ("../../../../checkpoints/unet_best.pth", str(experiment.ROOT / "checkpoints/unet_best.pth")):
            with self.assertRaises(ValueError):
                experiment.output_path(run, name)
        with patch.object(Path, "open", side_effect=FileExistsError()) as opened:
            with self.assertRaises(FileExistsError):
                experiment.write_json(run, "summary.json", {})
            self.assertEqual(opened.call_args.args[0], "x")

    def test_run_collision_not_reused(self):
        with patch.object(Path, "mkdir", side_effect=[None, FileExistsError()]) as mkdir:
            with self.assertRaises(FileExistsError):
                experiment.create_run()
            self.assertEqual(mkdir.call_args.kwargs, {"exist_ok": False})

    def test_hash_mismatch_and_missing_file_reported(self):
        with patch.object(experiment, "digest", return_value="different"):
            with self.assertRaisesRegex(ValueError, "Hash discrepancy"):
                experiment.verify_unchanged({"selected.png": "original"})
        with patch.object(experiment, "digest", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(ValueError, "missing"):
                experiment.verify_unchanged({"selected.png": "original"})

    def test_default_no_preflight_training_or_writes(self):
        with patch.object(experiment, "prepare") as prepare, patch.object(experiment, "train") as train, patch.object(experiment, "create_run") as create, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]), 0)
            prepare.assert_not_called()
            train.assert_not_called()
            create.assert_not_called()

    def test_validate_only_no_model_optimizer_or_writes(self):
        with patch.object(experiment, "prepare", return_value=(None, None, {}, {})), patch.object(experiment, "verify_unchanged") as verify, patch.object(experiment, "train", side_effect=AssertionError("training forbidden")), patch.object(experiment, "UNet", side_effect=AssertionError("model forbidden")), patch.object(experiment.torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")), patch.object(experiment, "create_run", side_effect=AssertionError("writes forbidden")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main(["--validate-only"]), 0)
            verify.assert_called_once_with({})

    def test_failed_preflight_never_trains(self):
        with patch.object(experiment, "prepare", side_effect=ValueError("bad hash")), patch.object(experiment, "train") as train, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main(["--train"]), 1)
            train.assert_not_called()

    def test_interrupted_run_never_writes_completion(self):
        # Enter orchestration with entirely mocked model/optimizer/I/O. Interrupt
        # baseline evaluation before any update; no real training is performed.
        optimizer = FakeOptimizer()
        with patch.object(experiment, "UNet"), patch.object(experiment.torch.optim, "Adam", return_value=optimizer), patch.object(experiment, "create_run", return_value=experiment.EXPERIMENT / "runs/mock"), patch.object(experiment, "write_json") as write, patch.object(Path, "open", return_value=io.StringIO()), patch.object(experiment.importlib.metadata, "version", return_value="mock"), patch.object(experiment, "evaluate", side_effect=KeyboardInterrupt()), patch.object(experiment, "verify_unchanged") as verify, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                experiment.train(experiment.FIXED, None, None, {}, {})
            self.assertEqual(optimizer.calls, 0)
            self.assertNotIn("summary.json", [call.args[1] for call in write.call_args_list])
            verify.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
