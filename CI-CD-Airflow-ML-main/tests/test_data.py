import os
import pandas as pd
from src.data_preprocessing import preprocess_data


def test_data_preprocessing(tmp_path):
    temp_data_dir = os.path.join(tmp_path, "processed")
    result = preprocess_data(data_dir=temp_data_dir)

    assert os.path.exists(result["train_path"])
    assert os.path.exists(result["test_path"])

    train_df = pd.read_csv(result["train_path"])
    test_df = pd.read_csv(result["test_path"])

    assert not train_df.empty
    assert not test_df.empty
    assert "target" in train_df.columns
    assert "target" in test_df.columns
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0
