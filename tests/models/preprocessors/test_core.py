from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vizard.models.preprocessors.core import (
    PandasTrainTestSplit,
    TrainTestEvalSplit,
    ColumnSelector,
)


# Sample DataFrame
@pytest.fixture
def sample_df():
    data = {
        "feature1": list(np.arange(start=1, stop=11)),
        "feature2": list(np.arange(start=11, stop=21)),
        "target": [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    }
    return pd.DataFrame(data)


@patch("vizard.models.preprocessors.core.TrainTestEvalSplit.set_configs")
def test_train_test_eval_split(mock_set_configs, sample_df):
    # Mocked configuration for testing
    mock_config = {
        "test_ratio": 0.20,
        "eval_ratio": 0.10,
        "shuffle": True,
        "stratify": None,
        "random_state": 42,
    }
    mock_set_configs.return_value = mock_config

    splitter = TrainTestEvalSplit()
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(sample_df, "target")

    assert len(x_train) == 7  # 70% of data
    assert len(x_test) == 2  # 20% of data
    assert len(x_eval) == 1  # 10% of data (taken from train set)
    assert x_train.shape[1] == 2  # Features
    assert y_train.shape[0] == 7
    assert y_test.shape[0] == 2
    assert y_eval.shape[0] == 1

    # Assert correct splitting with stratification
    sample_df["stratify_col"] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    splitter = TrainTestEvalSplit(stratify="stratify_col")
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(sample_df, "target")
    assert pd.Series(y_train).value_counts().to_dict() == {0: 4, 1: 3}
    assert pd.Series(y_test).value_counts().to_dict() == {0: 1, 1: 1}
    assert pd.Series(y_eval).value_counts().to_dict() == {0: 1}

    # Assert values are NumPy arrays
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)

    # Test with explicit random_state and stratify
    splitter = TrainTestEvalSplit(random_state=100, stratify=sample_df["stratify_col"])
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(sample_df, "target")
    assert pd.Series(y_train).value_counts().to_dict() == {0: 4, 1: 3}

    # Test with eval_ratio = 0
    mock_config["eval_ratio"] = 0.0
    splitter = TrainTestEvalSplit()
    x_train, x_test, y_train, y_test = splitter(sample_df, "target")
    assert len(x_train) == 8
    assert len(x_test) == 2


@patch("vizard.models.preprocessors.core.PandasTrainTestSplit.set_configs")
def test_pandas_train_test_split(mock_set_configs, sample_df):
    mock_config = {
        "train_ratio": 0.80,
        "shuffle": True,
        "stratify": None,
        "random_state": 42,
    }
    mock_set_configs.return_value = mock_config

    splitter = PandasTrainTestSplit()
    train_df, test_df = splitter(sample_df, "target")

    assert train_df.shape[0] == 8  # 80% of data
    assert test_df.shape[0] == 2  # 20% of data

    # test equality if seed is the same even if shuffle true
    splitter = PandasTrainTestSplit()
    train_df1, test_df1 = splitter(sample_df, "target")
    train_df2, test_df2 = splitter(sample_df, "target")  # Repeat with same random_state

    assert train_df1.equals(train_df2)
    assert test_df1.equals(test_df2)

    # test equality if seed is the same even if shuffle true
    splitter = PandasTrainTestSplit()
    train_df1, test_df1 = splitter(sample_df, "target")
    mock_config["random_state"] = 85
    splitter2 = PandasTrainTestSplit()
    train_df2, test_df2 = splitter2(sample_df, "target")

    assert not train_df1.equals(train_df2)
    assert not test_df1.equals(test_df2)


@pytest.fixture
def sample_df2():
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.1],
            "b": ["x", "y", "z"],
            "c_Country": [1.5, 2.5, 3.5],
            "d_Country": [True, False, True],
            "e_Country": [False, False, True],
        }
    )
    return data


def test_column_selector(sample_df2):
    selector = ColumnSelector(
        columns_type="string", dtype_include=bool, pattern_include=".*Country"
    )
    selected_columns = selector(sample_df2)
    assert selected_columns == ["d_Country", "e_Country"]

    selector = ColumnSelector(
        columns_type="string",
        dtype_include=float,
        pattern_include=".*Country",
        dtype_exclude=bool,
    )
    selected_columns = selector(sample_df2)
    assert selected_columns == ["c_Country"]

    selector = ColumnSelector(
        columns_type="numeric", dtype_include=float, pattern_exclude="a.*"
    )
    selected_columns = selector(sample_df2)
    assert selected_columns == [2]  # Index of column 'd_Country'

    selector = ColumnSelector(
        columns_type="numeric", dtype_include=float, pattern_include="c.*"
    )
    selected_columns = selector(sample_df2)
    assert selected_columns == [2]  # Index of column 'd_Country'

    selector = ColumnSelector(columns_type="string", dtype_include=str)
    with pytest.raises(TypeError):
        selector(sample_df2.values)  # Pass a NumPy array instead of DataFrame
