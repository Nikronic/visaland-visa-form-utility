from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vizard.models.preprocessors.core import (
    TrainTestEvalSplit,
)


# Mocked configuration for testing
mock_config = {
    "test_ratio": 0.20,
    "eval_ratio": 0.10,
    "shuffle": True,
    "stratify": None,
    "random_state": 42,
}

# Sample DataFrame
data = {
    "feature1": list(np.arange(start=1, stop=11)),
    "feature2": list(np.arange(start=11, stop=21)),
    "target": [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
}
df = pd.DataFrame(data)


@patch("vizard.models.preprocessors.core.TrainTestEvalSplit.set_configs")
def test_train_test_eval_split(mock_set_configs):
    mock_set_configs.return_value = mock_config

    splitter = TrainTestEvalSplit()
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(df, "target")

    assert len(x_train) == 7  # 70% of data
    assert len(x_test) == 2  # 20% of data
    assert len(x_eval) == 1  # 10% of data (taken from train set)
    assert x_train.shape[1] == 2  # Features
    assert y_train.shape[0] == 7
    assert y_test.shape[0] == 2
    assert y_eval.shape[0] == 1

    # Assert correct splitting with stratification
    df["stratify_col"] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    splitter = TrainTestEvalSplit(stratify="stratify_col")
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(df, "target")
    assert pd.Series(y_train).value_counts().to_dict() == {0: 4, 1: 3}
    assert pd.Series(y_test).value_counts().to_dict() == {0: 1, 1: 1}
    assert pd.Series(y_eval).value_counts().to_dict() == {0: 1}

    # Assert values are NumPy arrays
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)

    # Test with explicit random_state and stratify
    splitter = TrainTestEvalSplit(random_state=100, stratify=df["stratify_col"])
    x_train, x_test, x_eval, y_train, y_test, y_eval = splitter(df, "target")
    assert pd.Series(y_train).value_counts().to_dict() == {0: 4, 1: 3}

    # Test with eval_ratio = 0
    mock_config["eval_ratio"] = 0.0
    splitter = TrainTestEvalSplit()
    x_train, x_test, y_train, y_test = splitter(df, "target")
    assert len(x_train) == 8
    assert len(x_test) == 2
