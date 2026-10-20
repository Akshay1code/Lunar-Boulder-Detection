"""Synthetic safety tests: no optimizer updates and no dataset access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import train_experiment_i as I


class SafetyTests(unittest.TestCase):
    def test_joint_alignment_invertibility_counts(self):
        y=torch.zeros(8,1,8,8);y[:,:,1:3,4:7]=1
        x=y.repeat(1,3,1,1)
        with patch.object(torch,'randint',return_value=torch.tensor([0,1,2,3,0,1,2,3])):
            a,b,turns=I.joint_rotate(x,y,torch.Generator().manual_seed(42))
        self.assertEqual(set(turns),{0,1,2,3})
        self.assertTrue(torch.equal(a[:,0:1],b))
        self.assertTrue(torch.equal(y.sum((1,2,3)),b.sum((1,2,3))))
        for j,k in enumerate(turns):self.assertTrue(torch.equal(torch.rot90(b[j],-k,(-2,-1)),y[j]))

    def test_rng_reproducibility_and_isolation(self):
        x=torch.arange(64.).reshape(1,1,8,8);y=(x>20).float()
        state=torch.get_rng_state().clone()
        a=I.joint_rotate(x,y,torch.Generator().manual_seed(42))
        b=I.joint_rotate(x,y,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a[0],b[0]));self.assertEqual(a[2],b[2])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_metrics_and_strict_selection(self):
        self.assertIsNone(I.F.metrics(0,0,0,10)['dice'])
        self.assertIsNone(I.F.metrics(0,0,1,10)['precision'])
        self.assertEqual(I.F.metrics(1,0,0,10)['dice'],1)
        self.assertTrue(I.improves(0,None));self.assertFalse(I.improves(.5,.5))
        self.assertFalse(I.improves(.4,.5));self.assertTrue(I.improves(.6,.5))

    def test_schedule_and_successful_step_guard_without_steps(self):
        self.assertEqual(len(I.SCHEDULE),38);self.assertEqual(I.SCHEDULE[-1],5180)
        batches=list(I.F.bounded_batches([1,2,3],5))
        self.assertEqual([b[0] for b in batches],[1,2,3,4,5])
        # Invoke counter hooks directly; never call optimizer.step().
        counter=object.__new__(I.F.StepBudget);counter.successful=5179
        counter.pre(None,None,None);counter.post(None,None,None)
        self.assertEqual(counter.successful,5180)
        with self.assertRaises(ValueError):counter.pre(None,None,None)

    def test_output_unique_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(I,'HERE',Path(tmp)):
            a=I.create_run();b=I.create_run();self.assertNotEqual(a,b)
            I.write_json(a,'a.json',{'value':1})
            with self.assertRaises(ValueError):I.write_json(a,'a.json',{})
            with self.assertRaises(ValueError):I.output_path(a,'../../escape.pth')
            with self.assertRaises(ValueError):I.output_path(Path(tmp),'escape.pth')

    def test_checkpoint_weights_are_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(I,'HERE',Path(tmp)):
            run=I.create_run();model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
            initial=model.weight.detach().clone()
            path=I.save_checkpoint(run,'best_0.pth',model,optimizer,{},dict(update=0),torch.Generator())
            with torch.no_grad():model.weight.add_(1)
            loaded=torch.load(path,weights_only=True)
            self.assertTrue(torch.equal(loaded['model_state_dict']['weight'],initial))
            self.assertEqual(loaded['update'],0)

    def test_evaluation_unaugmented_no_grad(self):
        class Stub(torch.nn.Module):
            def forward(self,x):
                assert not torch.is_grad_enabled()
                return x[:,0:1]
        x=torch.ones(1,3,4,4);y=torch.ones(1,1,4,4)
        with patch.object(I,'joint_rotate',side_effect=AssertionError('Evaluation augmented')):
            result=I.evaluate(Stub(),[(x,y)],[dict(image_key='valid/synthetic',reference_foreground_coverage=1.)])
        self.assertEqual(result['nonempty']['mean']['dice'],1.)

    def test_default_no_prepare(self):
        with patch.object(I,'prepare',side_effect=AssertionError('Default prepared data')):I.main([])

    def test_interruption_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(I,'HERE',Path(tmp)), \
             patch.object(I.F,'make_loaders',return_value=([],[],torch.Generator())), \
             patch.object(I.F,'UNet',return_value=torch.nn.Linear(2,1)), \
             patch.object(I,'DataLoader',return_value=[]), \
             patch.object(I,'evaluate',side_effect=KeyboardInterrupt), \
             patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Training forbidden')):
            with self.assertRaises(KeyboardInterrupt):I.train({},None,[],[],None,[],{})
            run=next((Path(tmp)/'runs').iterdir())
            self.assertFalse((run/'summary.json').exists())
            self.assertEqual(json.loads((run/'incomplete.json').read_text())['status'],'incomplete')


if __name__=='__main__':unittest.main(verbosity=2)
