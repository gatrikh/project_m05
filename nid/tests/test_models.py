import pytest
import os
from nid import data
from nid import preprocess
from nid import model
from nid import visualize as vis

class TestModels:

    @pytest.fixture(scope="class")
    def config(self):
        df = data.load_data("kddcup_test.data.gz")
        df_encoded = preprocess.features_encoder(df)
        df_norm = preprocess.normalize(df_encoded)
        train_df, val_df, test_df = preprocess.split_data(df_norm)

        parameters = {
            "n_estimators": [100, 150], 
            "criterion": ["gini", "entropy"], 
            "max_depth": [None], 
            "min_samples_split": [2], 
            "max_features": ["auto"]
        }
        
        return train_df, val_df, test_df, parameters

    def test_get_x_y_from_df(self, config):
        train_df, _, _, _ = config
        n_features = train_df.shape[1]
        x_train, y_train = model.get_x_y_from_df(train_df)
        assert x_train.shape[1] == n_features - 1
        assert len(y_train.shape) == 1

    def test_generate_models_parameters(self, config):

        _, _, _, parameters  = config
        models = model.generate_models_parameters(parameters)
        assert len(models) == 4

    def test_train_validate_models(self, config):

        train_df, val_df, test_df, parameters = config
        models = model.generate_models_parameters(parameters)
        best_model = model.train_validate_models(train_df, val_df, models)
        keys = ["n_estimators", "criterion", "max_depth", "min_samples_split", "max_features"]
        for key in keys: 
            assert key in best_model

    def test_train_model(self, config):

        train_df, val_df, test_df, parameters = config
        model_parameters = {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'max_features': None}
        clf = model.train_model(train_df, val_df, model_parameters)
        x_test, y_test = model.get_x_y_from_df(test_df)
        acc, _, _ = vis.accuracy(y_test, clf.predict(x_test))
        assert acc > 0.95
    
    def test_write_read_model(self, config): 
        
        train_df, val_df, test_df, parameters = config
        model_parameters = {'n_estimators': 50, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'max_features': "auto"}
        clf = model.train_model(train_df, val_df, model_parameters)
        model.write_model(clf, "model_rfc_test", "")
        clf_1 = model.read_model("model_rfc_test", "")
        assert type(clf) == type(clf_1)
        assert os.path.isfile("model_rfc_test.pkl")
        os.remove("model_rfc_test.pkl")
        assert not os.path.isfile("model_rfc_test.pkl")

