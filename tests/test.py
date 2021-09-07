import unittest
import numpy as np
import pandas as pd
from src.data import data as data
from src.preprocessing import preprocessing as preprocess
from src.models import model as model
from src.visualization import visualize as vis

class TestData(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.df = data.load_data("../data/kddcup.data.gz")
        cls.df = preprocess.features_encoder(cls.df)
        cls.train_df, cls.val_df, cls.test_df = preprocess.split_data(cls.df)

    
    def test_df_split(self):
        
        train_labelCounts = self.train_df["label"].cat.remove_unused_categories().value_counts().to_numpy()
        val_labelCounts = self.val_df["label"].cat.remove_unused_categories().value_counts().to_numpy()
        test_labelCounts = self.test_df["label"].cat.remove_unused_categories().value_counts().to_numpy()
        
        train_ratio = train_labelCounts/(train_labelCounts+val_labelCounts+test_labelCounts)
        val_ratio = val_labelCounts/(train_labelCounts+val_labelCounts+test_labelCounts)
        test_ratio = test_labelCounts/(train_labelCounts+val_labelCounts+test_labelCounts)
        
        train_ratio_ref = np.full(np.shape(train_ratio), 0.6)
        val_ratio_ref = np.full(np.shape(val_ratio), 0.2)
        test_ratio_ref = np.full(np.shape(test_ratio), 0.2)
        
        np.testing.assert_allclose(train_ratio, train_ratio_ref,err_msg = "split test failed on train_set", atol = 0.05)
        np.testing.assert_allclose(val_ratio, val_ratio_ref, err_msg = "split test failed on val_set",atol = 0.05)
        np.testing.assert_allclose(test_ratio, test_ratio_ref, err_msg = "split test failed on test_set",atol = 0.05)
        
        
        

class TestModels(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
