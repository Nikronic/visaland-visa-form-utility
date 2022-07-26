__all__ = [
    'report_label_model'
]

# core
import numpy as np
# snorkel
from snorkel.labeling.model import LabelModel
# ours
from vizard.utils.helpers import loggingdecorator
# helpers
from typing import Dict
import logging


# configure logging
logger = logging.getLogger(__name__)


def report_label_model(label_model: LabelModel, label_matrix: np.ndarray,
                       gold_labels: np.ndarray, metrics: list,
                       set: str, **kwargs) -> Dict[str, float]:
    """Reports given ``metrics`` for the ``snorkel.LabelModel``

    Args:
        label_model (LabelModel): snorkel.LabelModel_ model
        label_matrix (:class:`numpy.ndarray`): label matrix produced by
            applying ``snorkel.PandasLFApplier.apply`` on a dataframe
        gold_labels (:class:`numpy.ndarray`): ground truth labels for given ``label_matrix``
        metrics (list): a list of metrics from sklearn.metrics_
        set (str): ``'train'`` or ``'test'`` set (affects print and logging)
        **kwargs (dict): additional keyword arguments to pass to ``label_model.score``:
            1. ``tie_break_policy``. Defaults to ``'abstain'``. 
            See snorkel.labeling.model.LabelModel.score_ for more info.

    .. _snorkel.LabelModel: https://snorkel.readthedocs.io/en/latest/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html
    .. _sklearn.metrics: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    .. _snorkel.labeling.model.LabelModel.score: https://snorkel.readthedocs.io/en/latest/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html#snorkel.labeling.model.label_model.LabelModel.score

    Returns:
        Dict[str, float]: a dictionary of metrics with keys as given in ``metrics``
    """
    tie_break_policy = kwargs.get('tie_break_policy', 'abstain')

    label_model_metrics = label_model.score(L=label_matrix, Y=gold_labels,
                                            tie_break_policy=tie_break_policy,
                                            metrics=metrics)
    logger.info(f'Label Model {set}ing stats: ')
    for m in metrics:
        logger.info('Label Model {}ing {}: {:.1f}%'.format(
            set, m, label_model_metrics[m] * 100))
    
    return label_model_metrics
