"""Contains preprocessing methods for preparing data solely for estimators in :mod:`vizard.models.estimators <vizard.models.estimators>`

This preprocessors expect "already cleaned" data acquired by :mod:`vizard.data <vizard.data>` 
for sole usage of machine learning models for desired frameworks (let's say
changing dtypes or one hot encoding for torch or sklearn that is only
useful for these frameworks)


Following modules are available:
    - :mod:`vizard.models.preprocessors.core`: contains implementations that could be shared between
        all other preprocessors modules defined here
    - :mod:`vizard.models.preprocessors.pytorch`: contains implementations to be used solely
        for PyTorch only for preprocessing purposes,
        e.g. https://pytorch.org/docs/stable/data.html
    - :mod:`vizard.models.preprocessors.sklearn`: contains implementations to be used solely
        for Scikit-Learn only for preprocessing purposes,
        e.g. https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

"""
import pathlib

# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / "data"

# we have to import path to `/data` here to avoid circular import in `core.py`
CANADA_COLUMN_TRANSFORMER_CONFIG_X = (
    DATA_DIR / "canada_column_transformer_config_x.json"
)
"""Configs for transforming *features* data for Canada

For information about how to use it and what fields are expected, 
see :class:`vizard.models.preprocessors.core.ColumnTransformerConfig`.
"""

CANADA_COLUMN_TRANSFORMER_CONFIG_Y = (
    DATA_DIR / "canada_column_transformer_config_y.json"
)
"""Configs for transforming *target* data for Canada

For information about how to use it and what fields are expected, 
see :class:`vizard.models.preprocessors.core.ColumnTransformerConfig`.
"""

CANADA_TRAIN_TEST_EVAL_SPLIT = DATA_DIR / "canada_train_test_eval_split.json"
"""Configs for splitting dataframe into numpy ndarray of train, test, eval for Canada

For information about how to use it and what fields are expected,
see :class:`vizard.models.preprocessors.core.TrainTestEvalSplit`.
"""

CANADA_PANDAS_TRAIN_TEST_SPLIT = DATA_DIR / "canada_pandas_train_test_split.json"
"""Configs for splitting dataframe train and test for Canada

For information about how to use it and what fields are expected,
see :class:`vizard.models.preprocessors.core.PandasTrainTestSplit`.
"""


# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# a dictionary of transforms and their names
TRANSFORMS = {
    "OneHotEncoder": OneHotEncoder,
    "LabelEncoder": LabelEncoder,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "LabelBinarizer": LabelBinarizer,
    "MultiLabelBinarizer": MultiLabelBinarizer,
}
"""A dictionary of transforms and their names used to verify configs in :class:`vizard.models.preprocessors.core.ColumnTransformerConfig`

This is used to verify that the configs are correct and that the transformers are available.

Note: 
    All transforms from third-party library or our own must be included in that list to be usable
    by :class:`vizard.models.preprocessors.core.ColumnTransformerConfig`.
"""


import logging

from .core import (
    ColumnSelector,
    ColumnTransformerConfig,
    PandasTrainTestSplit,
    TrainTestEvalSplit,
    get_transformed_feature_names,
    move_dependent_variable_to_end,
)
from .helpers import preview_column_transformer

# set logger
logger = logging.getLogger(__name__)

__all__ = [
    # configs constants
    "CANADA_COLUMN_TRANSFORMER_CONFIG_X",
    "CANADA_COLUMN_TRANSFORMER_CONFIG_Y",
    "CANADA_TRAIN_TEST_EVAL_SPLIT",
    "CANADA_PANDAS_TRAIN_TEST_SPLIT",
    # other constants
    "TRANSFORMS",
    # sklearn
    "ColumnTransformer",
    "OneHotEncoder",
    "LabelEncoder",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    # ours
    "PandasTrainTestSplit",
    "TrainTestEvalSplit",
    "ColumnTransformerConfig",
    "ColumnSelector",
    "move_dependent_variable_to_end",
    "preview_column_transformer",
    "get_transformed_feature_names",
]
