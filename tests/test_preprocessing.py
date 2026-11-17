import unittest
import numpy as np
from scipy.special import ndtr
from spvc_fts import financial_returns, temperature_fuzzy, errors

class PaperFormulas(unittest.TestCase):
    def test_financial_intervals(self):
        x = financial_returns([8, 9], [10, 12])[0]
        self.assertAlmostEqual(x[0]-x[1], np.log(9/10))
        self.assertAlmostEqual(x[0]+x[2], np.log(12/8))
    def test_temperature_probability(self):
        x = np.linspace(-1, 1, 100)
        result = temperature_fuzzy(x, probability=.03)
        mu, std = x.mean(), x.std()
        p = ndtr((x-mu)/std)
        np.testing.assert_allclose(p-ndtr((x-result[:,1]-mu)/std), .03)
        np.testing.assert_allclose(ndtr((x+result[:,2]-mu)/std)-p, .03)
    def test_tail_infeasibility(self):
        with self.assertRaises(ValueError): temperature_fuzzy([0]*99+[100])
    def test_metrics(self):
        result = errors([[1,2,3]], [[0,0,0]])
        self.assertEqual(result, {'mse':14., 'mae':6.})

if __name__ == '__main__': unittest.main()
