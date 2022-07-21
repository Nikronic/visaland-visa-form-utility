"""Contains preprocessing methods for preparing data solely for estimators in :mod:`vizard.models.estimators <vizard.models.estimators>`

This preprocessors expect "already cleaned" data acquired by :mod:`vizard.data <vizard.data>` 
for sole usage of machine learning models for desired frameworks (let's say
changing dtypes or one hot encoding for torch or sklearn that is only
useful for these frameworks)
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
from .core import column_selector
# ours: helpers
from .helpers import preview_column_transformer

# helpers
import logging


# set logger
logger = logging.getLogger(__name__)
