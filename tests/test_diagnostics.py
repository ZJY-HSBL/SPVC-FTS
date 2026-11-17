import unittest
import numpy as np
from spvc_fts.diagnostics import median_runs, centre_ar, cook_centre

class DiagnosticsTests(unittest.TestCase):
    def test_runs(self):
        r=median_runs([1,3,2,4,1,5])
        self.assertEqual(r['above'],3);self.assertEqual(r['below'],3);self.assertEqual(r['runs'],6)
        self.assertLess(r['p_value'],.1)
    def test_ar_and_cook(self):
        rng=np.random.default_rng(9); x=rng.normal(size=60); data=np.column_stack((x,np.ones(60),np.ones(60)))
        baseline=centre_ar(data,3); d=cook_centre(data,3)
        self.assertEqual(baseline.shape,(57,3));self.assertTrue(np.all(baseline[:,1:]==0))
        self.assertTrue(np.all(np.isfinite(d['distance'])));self.assertTrue((d['distance']>=0).all())
    def test_rank(self):
        with self.assertRaises(ValueError): centre_ar(np.ones((20,3)),3)

if __name__=='__main__': unittest.main()
