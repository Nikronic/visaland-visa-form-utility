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
DATA_DIR = parent_dir / 'data'

# we have to import path to `/data` here to avoid circular import in `core.py`
CANADA_COLUMN_TRANSFORMER_CONFIG_X = DATA_DIR / 'canada_column_transformer_config_x.json'
"""Configs for transforming *train* data for Canada

For information about how to use it and what fields are expected, 
see :class:`vizard.models.preprocessors.core.ColumnTransformerConfig`.
"""

CANADA_TRAIN_TEST_EVAL_SPLIT = DATA_DIR / 'canada_train_test_eval_split.json'
"""Configs for splitting dataframe into numpy ndarray of train, test, eval for Canada

For information about how to use it and what fields are expected,
see :class:`vizard.models.preprocessors.core.TrainTestEvalSplit`.
"""


# sklearn wrappers
from .core import ColumnTransformer
from .core import OneHotEncoder
from .core import LabelEncoder
from .core import StandardScaler
from .core import MinMaxScaler
from .core import RobustScaler
from .core import MaxAbsScaler

# ours: core
from .core import move_dependent_variable_to_end
from .core import ColumnTransformerConfig
from .core import TrainTestEvalSplit
from .core import ColumnSelector
# ours: helpers
from .helpers import preview_column_transformer

# helpers
import logging



# set logger
logger = logging.getLogger(__name__)
