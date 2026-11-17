import unittest
import numpy as np
from experiments.run_paper_cases import prep

class PaperCases(unittest.TestCase):
    def test_inputs_and_temperature_domain(self):
        cases=prep()
        self.assertEqual(set(cases),{'shenzhen','gold','shanghai'})
        for dates, data, meta in cases.values():
            self.assertEqual(len(dates),len(data))
            self.assertTrue(np.isfinite(data).all())
            self.assertTrue((data[:,1:]>0).all())
        self.assertEqual(cases['shanghai'][2]['original_normal_infeasible_count'],12)
        self.assertEqual(len(cases['shenzhen'][0]),198)

if __name__=='__main__': unittest.main()
