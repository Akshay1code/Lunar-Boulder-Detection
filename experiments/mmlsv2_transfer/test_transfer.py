"""Synthetic safety checks; never loads data or trains the U-Net."""
import sys
sys.dont_write_bytecode=True
import unittest
import torch
import numpy as np
from pathlib import Path
from run_transfer import Counter, I, F, path, HERE, preprocess

class SafetyTests(unittest.TestCase):
    def test_rgb_and_nearest_mask(self):
        x=np.zeros((128,128,7),np.float32);x[:,:,:3]=[.2,.4,.8]
        y=np.zeros((128,128),np.uint8);y[2:7,4:9]=1
        a,b=preprocess(x,y)
        self.assertTrue(torch.allclose(a[:,0,0],torch.tensor([.2,.4,.8])))
        self.assertEqual(int(b.sum()),100)
        self.assertTrue(np.array_equal(b[0].numpy()[::2,::2],y))
    def test_rotation_reproducible_aligned(self):
        y=torch.zeros(2,1,8,8);y[:,:,1:3,4:7]=1;x=y.repeat(1,3,1,1)
        a,b,k=I.joint_rotate(x,y,torch.Generator().manual_seed(42))
        c,d,l=I.joint_rotate(x,y,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        self.assertTrue(torch.equal(a[:,0:1],b));self.assertEqual(k,l)
    def test_step_cap(self):
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.Adam([p]);counter=Counter(opt,2)
        try:
            for _ in range(2):p.grad=torch.tensor(0.);opt.step()
            self.assertEqual(counter.n,2)
            with self.assertRaises(RuntimeError):opt.step()
            self.assertEqual(counter.n,2)
        finally:counter.close()
    def test_loss_gradients_and_metrics(self):
        z=torch.zeros((2,1,8,8),requires_grad=True);y=torch.zeros_like(z);y[0,:,1:4,1:4]=1
        c=F.loss_components(z,y);c['total_loss'].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertAlmostEqual(float(c['bce']),np.log(2),places=6)
        self.assertIsNone(F.metrics(0,0,0,64)['dice'])
        self.assertFalse(I.improves(.5,.5));self.assertTrue(I.improves(.6,.5))
    def test_paths(self):
        with self.assertRaises(ValueError):path(HERE,'config.json')
        with self.assertRaises(ValueError):path(HERE/'runs'/'synthetic','../../config.json')
    def test_schedule_and_exposure(self):
        self.assertEqual(2330//233,10);self.assertEqual(5180//37,140)
        self.assertEqual(len(range(0,5181,140)),38)

if __name__=='__main__':unittest.main()
