from unittest.case import TestCase

import pandas as pd
from btgym.datafeed import BTgymBaseData


class TestBTgymBaseData(TestCase):
    def test_set_params(self):
        btgym_data = BTgymBaseData(dataframe=pd.DataFrame())
        self.assertRaises(AttributeError, getattr, btgym_data, "test_param")
        btgym_data.set_params({"test_param": "test_value"})
        self.assertTrue(btgym_data.test_param is "test_value")

    def test_set_global_timestamp(self):
        btgym_data = BTgymBaseData(dataframe=pd.DataFrame())
        btgym_data.set_global_timestamp(1)
        self.assertEqual(btgym_data.global_timestamp, 1)

    def test_reset(self):
        self.fail()

    def test_to_btfeed(self):
        btgym_data = BTgymBaseData(pd.DataFrame())
        res = btgym_data.to_btfeed()
        print(type(res))

    def test_sample(self):
        self.fail()
