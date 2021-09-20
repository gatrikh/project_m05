import pytest
import numpy as np
import pandas as pd
import math
from src.data import data as data
from src.preprocessing import preprocessing as preprocess

class TestPreprocessing:

    @pytest.fixture(scope="class")
    def config(self):
        df = data.load_data("data/kddcup_test.data.gz")
        return df
    
    def test_features_encoder(self, config):

        df_encoded = preprocess.features_encoder(config)

        for col in df_encoded.columns: 
            assert df_encoded[col].dtype != "object"

    def test_normalize(self, config): 
        
        df_encoded = preprocess.features_encoder(config)
        df_norm = preprocess.normalize(df_encoded)

        array_1 = df_norm["dst_bytes"].to_numpy()
        array_2 = (df_norm["dst_bytes"].to_numpy() - np.mean(df_norm["dst_bytes"].to_numpy())/np.std(df_norm["dst_bytes"].to_numpy()))

        np.testing.assert_allclose(array_1, array_2, atol=0.0005)

    def test_df_split(self, config):
        
        df_encoded = preprocess.features_encoder(config)
        df_norm = preprocess.normalize(df_encoded)
        train_df, val_df, test_df = preprocess.split_data(df_norm)

        nbr_label = pd.unique(df_norm["label"])
        nbr_label = nbr_label.sort_values()

        list_len_train = []
        list_len_val = []
        list_len_test = []

        for label in nbr_label:

            len_subset = df_norm[df_norm["label"] == label].shape[0]
        
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
                list_len_test.append(1)
                
            elif len_subset == 1: 
                list_len_train.append(1)

        arr_len_train = np.array(list_len_train)
        arr_len_val = np.array(list_len_val)
        arr_len_test = np.array(list_len_test)

        len_train_df = train_df["label"].astype("int").value_counts().sort_index().to_numpy()
        len_val_df = val_df["label"].astype("int").value_counts().sort_index().to_numpy()
        len_test_df = test_df["label"].astype("int").value_counts().sort_index().to_numpy()

        np.testing.assert_array_equal(arr_len_train, len_train_df, "Train arrays length not equal")
        np.testing.assert_array_equal(arr_len_val, len_val_df, "Validation arrays length not equal")
        np.testing.assert_array_equal(arr_len_test, len_test_df, "Test arrays length not equal")