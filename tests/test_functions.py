import unittest
from featurePruner.helper_functions import checkFullCorrelation
from featurePruner.secondary_utils import n_largest_correlation
import pandas as pd

class HelperTests(unittest.TestCase):
    
    def make_data(self):
        return pd.DataFrame({"A":[1,2,3,4,5], "B":[4,3,5,1,2], "C":[6,7,8,9,10]})
    
    def test_probe_for_redundant_vars(self):
        df = self.make_data()
        res = checkFullCorrelation(df)
        res = set([item for tup in res for item in tup])
        self.assertEqual(res, {"A", "C"})
        
    def test_largest_corr(self):
        df = self.make_data()
        res = n_largest_correlation(df.corr(), 3)
        self.assertEqual(res.iloc[1,2], -0.6)

if __name__ == "__main__":
    unittest.main()