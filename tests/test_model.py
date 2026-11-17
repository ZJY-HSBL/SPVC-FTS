import unittest
import numpy as np
from spvc_fts import fit, select_bandwidth, msm
from spvc_fts.model import kernel

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7); n=32
        c=np.zeros(n); l=np.zeros(n); u=np.zeros(n)
        c[0],l[0],u[0]=.2,.5,.7
        for i in range(1,n):
            c[i]=.65*c[i-1]+.08*np.sin(i/6)+rng.normal(0,.015)
            l[i]=.2+.55*l[i-1]+.03*c[i]+rng.normal(0,.005)
            u[i]=.22+.6*u[i-1]+.02*c[i]+rng.normal(0,.005)
        self.data=np.column_stack([c,l,u])
    def test_fit_both_stages(self):
        res=fit(self.data,lags=(1,1,1),h_np=.3,h_vc=.3)
        self.assertEqual(res['stage2'].shape,(31,3))
        self.assertTrue(np.isfinite(res['stage2']).all())
        self.assertTrue(np.all(res['stage2'][:,1:]>0))
        self.assertGreater(msm(res['actual'],res['stage2']),0.8)
    def test_cross_validation(self):
        for stage in (1,2):
            result=select_bandwidth(self.data,[.2,.4],stage=stage,lags=(1,1,1),h_np=.3)
            self.assertIn(result['bandwidth'],[.2,.4])
            self.assertTrue(np.isfinite(result['scores']).all())
    def test_msm_identity_and_disjoint(self):
        a=np.array([[0,1,1.]])
        self.assertAlmostEqual(msm(a,a),1.)
        self.assertAlmostEqual(msm(a,[[5,1,1]]),0.)
        self.assertAlmostEqual(msm(a,[[1,1,1]]),1/7)
    def test_kernels(self):
        for name in ('epanechnikov','biweight','triweight'):
            self.assertEqual(kernel([2],name)[0],0)
        with self.assertRaises(ValueError): kernel([0],'bad')

if __name__ == '__main__': unittest.main()
