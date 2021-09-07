import unittest

from src.data import data as data
from src.preprocessing import preprocessing as preprocess
from src.models import model as model
from src.visualization import visualize as vis

from sklearn.model_selection import train_test_split

class TestData(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 3)), 6, "Should be 6")


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
    unittest.main()
