import pytest
import numpy as np
import pandas as pd
import math
from src.data import data as data
from src.preprocessing import preprocessing as preprocess
from src.models import model as model
from src.visualization import visualize as vis

class TestPreprocessing:

    @pytest.fixture(scope="class")
    def config(self):
        df = data.load_data("data/kddcup_test.data.gz")
        df_encoded = preprocess.features_encoder(df)
        df_norm = preprocess.normalize(df_encoded)
        train_df, val_df, test_df = preprocess.split_data(df_norm)

        parameters = {
            "n_estimators": [100], 
            "criterion": ["gini"], 
            "max_depth": [None], 
            "min_samples_split": [2], 
            "max_features": ["auto"]
        }
        
        return train_df, val_df, test_df, parameters
    
    def test_dsp(self, config):
        train_df, _, _, _ = config
        vis.dsp(train_df)

    def test_memory(self, config): 
        train_df, _, _, _ = config

        size = vis.memory(train_df)
        assert isinstance(size, float)

    def test_accuracy(self, config): 
        train_df, val_df, test_df, parameters = config 

        clf = model.train_model(train_df, val_df, parameters)
        x_test, y_test = model.get_x_y_from_df(test_df)
        acc, correct, incorrect = vis.accuracy(y_test, clf.predict(x_test))
        assert isinstance(acc, float)
        assert isinstance(correct, int)
        assert isinstance(incorrect, int)
        assert correct > incorrect
        assert acc < 1
        assert 0.9 < acc and acc < 1.0

