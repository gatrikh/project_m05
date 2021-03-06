import pytest
import numpy as np
import pandas as pd
import math
import os
from nid import data
from nid import preprocess
from nid import model
from nid import visualize as vis

class TestPreprocessing:

    @pytest.fixture(scope="class")
    def config(self):
        df = data.load_data("kddcup_test.data.gz")
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

        model_paramters = {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'max_features': None}
        clf = model.train_model(train_df, val_df, model_paramters)
        x_test, y_test = model.get_x_y_from_df(test_df)
        acc, correct, incorrect = vis.accuracy(y_test, clf.predict(x_test))
        assert isinstance(acc, float)
        assert isinstance(correct, int)
        assert isinstance(incorrect, int)
        assert correct > incorrect
        assert acc < 1
        assert 0.9 < acc and acc < 1.0
        
    def test_get_confusion_matrix(self, config):
        train_df, val_df, test_df, _ = config 
    
        model_paramters = {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'max_features': None}
        clf = model.train_model(train_df, val_df, model_paramters)
        
        x_test, y_test = model.get_x_y_from_df(test_df)

        vis.get_confusion_matrix(clf, x_test, y_test, True, "unit_test", "")
        assert os.path.exists('normalized_unit_test.png')
        os.remove('normalized_unit_test.png')
        assert os.path.exists('unnormalized_unit_test.png')
        os.remove('unnormalized_unit_test.png')
        

