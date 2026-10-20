"""Non-training checks: synthetic tensors and mocked writes only."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_experiment_b as experiment
import torch


class SafeguardTests(unittest.TestCase):
    def test_selection_and_count_discrepancies(self):
        selected = experiment.checked_selection([True] * 73 + [False] * 206, 279, 73, "train")
        self.assertEqual(selected, list(range(73)))
        for flags in ([True] * 72 + [False] * 207, [True] * 74 + [False] * 205, [True] * 73):
            with self.assertRaises(ValueError):
                experiment.checked_selection(flags, 279, 73, "train")
        self.assertEqual(len(experiment.checked_selection([True] * 19 + [False] * 61, 80, 19, "valid")), 19)

    def test_disallowed_split_never_instantiates_loader(self):
        with patch.object(experiment, "LunarLandslideDataset") as loader:
            with self.assertRaises(ValueError):
                experiment.make_dataset("test", 256)
            loader.assert_not_called()

    def test_metric_edges(self):
        m = experiment.metrics(0, 0, 0, 100)
        self.assertIsNone(m["dice"])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["precision"])
        self.assertEqual(m["predicted_foreground_coverage"], 0)
        m = experiment.metrics(0, 10, 0, 100)
        self.assertEqual(m["dice"], 0)
        self.assertIsNone(m["recall"])
        self.assertEqual(m["precision"], 0)
        m = experiment.metrics(0, 0, 10, 100)
        self.assertEqual((m["dice"], m["recall"]), (0, 0))
        self.assertIsNone(m["precision"])
        m = experiment.metrics(3, 1, 2, 10)
        self.assertAlmostEqual(m["dice"], 6 / 9)
        self.assertAlmostEqual(m["recall"], 3 / 5)
        self.assertAlmostEqual(m["precision"], 3 / 4)
        self.assertEqual(m["predicted_foreground_coverage"], .4)

    def test_aggregation_tracks_undefined_values(self):
        rows = []
        for tp, fp, fn in [(0, 0, 0), (0, 1, 0)]:
            rows.append(dict(tp=tp, fp=fp, fn=fn, pixels=4, bce=.2,
                             **experiment.metrics(tp, fp, fn, 4)))
        group = experiment.aggregate(rows)
        self.assertIsNone(group["macro"]["recall"])
        self.assertEqual(group["macro_valid_counts"]["dice"], 1)
        self.assertEqual(group["macro_valid_counts"]["recall"], 0)
        self.assertEqual(group["pooled"]["predicted_foreground_coverage"], 1 / 8)

    def test_evaluation_is_inference_only_and_fixed_threshold(self):
        class IdentityLogits:
            def eval(self):
                pass

            def __call__(self, x):
                if not torch.is_inference_mode_enabled():
                    raise AssertionError("Evaluation must disable gradients")
                return x

        logits = torch.tensor([[[[-1., 0.]]], [[[-1., -1.]]]])
        targets = torch.tensor([[[[0., 1.]]], [[[0., 0.]]]])
        records = [{"image_key": "valid/a", "reference_nonempty": True},
                   {"image_key": "valid/b", "reference_nonempty": False}]
        summary, rows = experiment.evaluate(IdentityLogits(), [(logits, targets)], records, .5)
        self.assertEqual(rows[0]["dice"], 1)
        self.assertEqual(summary["nonempty"]["images"], 1)
        self.assertIsNone(rows[1]["dice"])
        self.assertIsNone(summary["empty_unverified"]["pooled"]["recall"])

    def test_exact_update_budget_without_training(self):
        class FakeLoader:
            fetched = 0

            def __len__(self):
                return 37

            def __iter__(self):
                for batch in range(37):
                    self.fetched += 1
                    yield batch

        loader = FakeLoader()
        steps = list(experiment.bounded_batches(loader, 700))
        self.assertEqual(len(steps), 700)
        self.assertEqual(loader.fetched, 700)
        self.assertEqual(steps[-1][:3], (700, 19, 34))
        self.assertEqual([s for s, *_ in steps if s % 140 == 0], [140, 280, 420, 560, 700])
        with self.assertRaises(ValueError):
            list(experiment.bounded_batches([], 700))

    def test_checkpoint_path_protection(self):
        run = experiment.EXPERIMENT_ROOT / "runs/unit_test"
        allowed = experiment.output_path(run, "checkpoints/best.pth")
        self.assertTrue(allowed.is_relative_to(run.resolve()))
        for run_dir, relative in [(run, "../../../../checkpoints/unet_best.pth"),
                                  (run, str(experiment.ORIGINAL_CHECKPOINT.resolve())),
                                  (experiment.PROJECT_ROOT, "checkpoints/unet_best.pth")]:
            with self.assertRaises(ValueError):
                experiment.output_path(run_dir, relative)
        with patch.object(Path, "open", side_effect=FileExistsError("existing checkpoint")), \
             patch.object(Path, "mkdir"), patch.object(experiment.torch, "save") as save:
            with self.assertRaises(FileExistsError):
                experiment.save_checkpoint(run, "best.pth", {})
            save.assert_not_called()

    def test_run_collision_is_not_reused(self):
        with patch.object(Path, "mkdir", side_effect=[None, FileExistsError("collision")]) as mkdir:
            with self.assertRaises(FileExistsError):
                experiment.create_run()
            self.assertEqual(mkdir.call_args_list[-1].kwargs, {"exist_ok": False})

    def test_default_does_not_prepare_or_train(self):
        with patch.object(experiment, "prepare") as prepare, \
             patch.object(experiment, "train") as train, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]), 0)
            prepare.assert_not_called()
            train.assert_not_called()

    def test_validate_only_has_no_training_or_writes(self):
        fake = ({}, {"train": [None] * 279, "valid": [None] * 80},
                {"train": list(range(73)), "valid": list(range(19))},
                {str(experiment.ORIGINAL_CHECKPOINT): "mock_read_only_hash"})
        with patch.object(experiment, "prepare", return_value=fake), \
             patch.object(experiment, "verify_unchanged") as verify, \
             patch.object(experiment, "train", side_effect=AssertionError("Training forbidden")), \
             patch.object(experiment, "UNet", side_effect=AssertionError("Model construction forbidden")), \
             patch.object(experiment.torch, "save", side_effect=AssertionError("Checkpoint write forbidden")), \
             patch.object(experiment, "create_run", side_effect=AssertionError("Run creation forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main(["--validate-only"]), 0)
            verify.assert_called_once_with(fake[3])

    def test_failed_preflight_cannot_reach_training(self):
        with patch.object(experiment, "prepare", side_effect=ValueError("Expected 73")), \
             patch.object(experiment, "train") as train, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main(["--train"]), 1)
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
