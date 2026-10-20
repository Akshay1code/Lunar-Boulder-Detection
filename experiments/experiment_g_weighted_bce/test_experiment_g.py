"""Non-training tests; no optimizer updates, no dataset access or disk writes."""
import io
import sys
sys.dont_write_bytecode=True
import unittest
from unittest.mock import patch
import train_experiment_g as G
import torch


class Checks(unittest.TestCase):
    def test_weight_and_loss(self):
        z=torch.zeros(2,1,2,2,requires_grad=True)
        y=torch.tensor([0.,1.,0.,1.]*2).reshape_as(z)
        c=G.loss_components(z,y);f=G.F.loss_components(z,y)
        self.assertEqual(G.POS_WEIGHT,5.521724)
        self.assertTrue(torch.equal(c['dice_loss'],f['dice_loss']))
        self.assertAlmostEqual(float(c['weighted_bce'].detach()),float(torch.log(torch.tensor(2.)))*(1+G.POS_WEIGHT)/2,places=5)
        self.assertTrue(torch.equal(c['total_loss'],c['weighted_bce']+c['dice_loss']))
        c['total_loss'].backward()  # Synthetic loss gradient only; no model training.
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_empty_targets(self):
        z=torch.zeros(1,1,2,2);y=torch.zeros_like(z)
        c=G.loss_components(z,y)
        self.assertEqual(float(c['weighted_bce']),float(c['bce']))
        self.assertAlmostEqual(float(c['dice_loss']),2/3,places=6)

    def test_budget_and_schedule(self):
        batches=list(G.F.bounded_batches([0]*37,5180))
        self.assertEqual(len(batches),5180)
        self.assertEqual(batches[-1][:3],(5180,140,37))
        self.assertEqual(len(G.SCHEDULE),38)
        self.assertEqual(G.SCHEDULE[-1],5180)
        counter=object.__new__(G.F.StepBudget);counter.successful=5180
        with self.assertRaises(ValueError):counter.pre(None,None,None)
        counter.successful=199;counter.post(None,None,None)
        self.assertEqual(counter.successful,200)

    def test_incomplete_rejected(self):
        counter=type('Counter',(),{'successful':5179})()
        with self.assertRaises(ValueError):G.F.check_completion(counter,None,[],0,0,0)

    def test_checkpoint_snapshot(self):
        model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters())
        s=G.state(model,optimizer,{},dict(update=140),'test')
        original=s['model_state_dict']['weight'].clone()
        with torch.no_grad():model.weight.add_(1)
        self.assertTrue(torch.equal(s['model_state_dict']['weight'],original))
        buffer=io.BytesIO();torch.save(s,buffer);buffer.seek(0)
        G.F.verify_checkpoint(torch.load(buffer,weights_only=True),s)

    def test_ties_and_lower_scores_no_write(self):
        best=G.Best(G.EXPERIMENT/'runs'/'unused','validation');best.score=.5
        for score in (.5,.4):
            self.assertFalse(best.consider(None,None,None,{'validation':{'nonempty':{'macro_mean':{'dice':score}}}}))

    def test_paths(self):
        with self.assertRaises(ValueError):G.destination(G.ROOT/'checkpoints','best.pth')
        with self.assertRaises(ValueError):G.destination(G.EXPERIMENT/'runs'/'example','../escape.pth')

    def test_evaluation_no_grad_and_rng(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                self.grad_enabled=torch.is_grad_enabled()
                return torch.zeros(x.shape[0],1,2,2)
        model=Model();model.train()
        x=torch.zeros(1,3,2,2);y=torch.zeros(1,1,2,2)
        r=[dict(image_key='valid/synthetic',coco_image_id=0,reference_foreground_coverage=0.)]
        result=G.evaluate(model,[(x,y)],r)
        self.assertFalse(model.grad_enabled);self.assertTrue(model.training)
        self.assertEqual(result['empty_unverified']['images_with_foreground'],1)
        self.assertIsNone(result['nonempty']['macro_mean']['dice'])


if __name__=='__main__':unittest.main()
