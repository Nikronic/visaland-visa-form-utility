import logging
import pathlib

from snorkel.analysis import Scorer
from snorkel.augmentation import (
    ApplyAllPolicy,
    ApplyEachPolicy,
    ApplyOnePolicy,
    MeanFieldPolicy,
    PandasTFApplier,
    RandomPolicy,
)
from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.slicing import PandasSFApplier, slice_dataframe

from . import augmentation, labeling, modeling, slicing

# set logger
logger = logging.getLogger(__name__)


# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / "data"


LABEL_MODEL_CONFIGS = DATA_DIR / "label_model_configs.json"
"""Path to the json file containing the configs for the LabelModel

In the highest level of configs, there are keys that contain the name of method
or property of a `Callable` object. These needs to be passed to the :meth:`parse`
method of the :class:`vizard.configs.core.JsonConfigHandler` class.

    >>> from vizard.configs import JsonConfigHandler
    >>> from vizard.labeling.model import LabelModel
    >>> configs = JsonConfigHandler.parse(LABEL_MODEL_CONFIGS, 'LabelModel')
    >>> configs['method_fit']['n_epochs']
    100
    >>> configs['method_init']['cardinality']
    2

"""
