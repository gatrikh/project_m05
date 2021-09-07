import unittest
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
import math

from src.data import data as data
from src.preprocessing import preprocessing as preprocess
from src.models import model as model
from src.visualization import visualize as vis

class TestData(unittest.TestCase):

    def test_load_data(self):
        df = data.load_data("data/kddcup.data.gz")

        self.assertEqual(df.shape[0], 4898431, "Should be 4898431")
        self.assertEqual(df.shape[1], 42, "Should be 42")

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.df = data.load_data("data/kddcup.data.gz")
        cls.df = preprocess.features_encoder(cls.df)
        cls.train_df, cls.val_df, cls.test_df = preprocess.split_data(cls.df)

    def test_features_encoder(self):

        for col in self.df.columns: 
            self.assertEqual(self.df[col].dtype == "object", False, "Should be False")
    
    def test_normalize(self): 

        df_norm = preprocess.normalize(self.df)

        array_1 = df_norm["dst_bytes"].to_numpy()
        array_2 = (df_norm["dst_bytes"].to_numpy() - np.mean(df_norm["dst_bytes"].to_numpy())/np.std(df_norm["dst_bytes"].to_numpy()))

        np.testing.assert_allclose(array_1, array_2, err_msg="Normalization test failed", atol=0.0005)

    def test_df_split(self):

        nbr_label = pd.unique(self.df["label"])
        nbr_label = nbr_label.sort_values()

        list_len_train = []
        list_len_val = []
        list_len_test = []

        for label in nbr_label:

            len_subset = self.df[self.df["label"] == label].shape[0]
        
            if len_subset >= 4:
                
                len_train = math.floor(0.6 * len_subset)
                len_val = math.floor((len_subset - len_train)*0.5)
                len_test = len_subset - len_train - len_val

                list_len_train.append(len_train)
                list_len_val.append(len_val)
                list_len_test.append(len_test)

            elif len_subset == 3: 

                list_len_train.append(1)
                list_len_val.append(1)
                list_len_test.append(1)

            elif len_subset == 2:
                list_len_train.append(1)
                list_len_val.append(1)
                
            elif len_subset == 1: 
                list_len_train.append(1)

        arr_len_train = np.array(list_len_train)
        arr_len_val = np.array(list_len_val)
        arr_len_test = np.array(list_len_test)

        len_train_df = self.train_df["label"].value_counts().sort_index().to_numpy()
        len_val_df = self.val_df["label"].value_counts().sort_index().to_numpy()
        len_test_df = self.test_df["label"].value_counts().sort_index().to_numpy()

        np.testing.assert_array_equal(arr_len_train, len_train_df, "Train arrays length not equal")
        np.testing.assert_array_equal(arr_len_val, len_val_df, "Validation arrays length not equal")
        np.testing.assert_array_equal(arr_len_test, len_test_df, "Test arrays length not equal")
        
class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = data.load_data("data/kddcup.data.gz")
        cls.df = preprocess.features_encoder(cls.df)
        train, _ = train_test_split(cls.df, test_size=0.95, random_state=0)
        cls.train, cls.val = train_test_split(train, test_size=0.3, random_state=0)
        cls.parameters = {
            "n_estimators": [100, 150, 200], 
            "criterion": ["gini", "entropy"], 
            "max_depth": [None], 
            "min_samples_split": [2], 
            "max_features": ["auto", None]
        }

    def test_get_x_y_from_df(self):
        n_features = self.train.shape[1]
        x_train, y_train = model.get_x_y_from_df(self.train)
        self.assertEqual(x_train.shape[1], n_features - 1, f"Should be {n_features - 1}")
        self.assertEqual(len(y_train.shape), 1, f"Should be {1}")

    def test_generate_models_parameters(self):
        models = model.generate_models_parameters(self.parameters)
        self.assertEqual(len(models), 12, f"Should be {12}")

    def test_train_validate_models(self):
        models = model.generate_models_parameters(self.parameters)
        best_model = model.train_validate_models(self.train, self.val, models)
        keys = ["n_estimators", "criterion", "max_depth", "min_samples_split", "max_features"]
        for key in keys: 
            self.assertEqual(key in best_model, True, "Should be True")

    def test_train_model(self):
        clf = model.train_model(self.train, self.val, self.parameters)
        x_train, y_train = model.get_x_y_from_df(self.train)
        acc, _, _ = vis.accuracy(y_train, clf.predict(x_train))
        self.assertEqual(acc > 0.999, True, "Should be True")

if __name__ == '__main__':

    # suite = unittest.TestSuite()
    # suite.addTest(TestPreprocessing("test_df_split"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    unittest.main()
