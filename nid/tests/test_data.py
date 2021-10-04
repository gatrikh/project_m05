import pytest
from nid import data
import math

class TestData: 
    def test_load_data(self):
        df = data.load_data("kddcup_test.data.gz")

        assert df.shape[0] == math.ceil(0.005 * 4898431)
        assert df.shape[1] == 42

