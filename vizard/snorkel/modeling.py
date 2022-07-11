__all__ = ['report_label_model']

# core
import numpy as np
# snorkel
from snorkel.labeling.model import LabelModel
# ours
from vizard.utils.helpers import loggingdecorator
# helpers
from typing import List
import logging


# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@loggingdecorator(logger.name+'.func', level=logging.INFO, output=False, input=False)
def report_label_model(label_model: LabelModel, label_matrix: np.ndarray,
                       gold_labels: np.ndarray, metrics: list,
                       set: str, **kwargs) -> None:
    """Reports given `metrics` for the `snorkel.LabelModel`

    Args:
        label_model (LabelModel): `snorkel.LabelModel` [#]_ model (`torch.Module` base class)
        label_matrix (np.ndarray): label matrix produced by applying `PandasLFApplier.apply`
            on a dataframe
        gold_labels (np.ndarray): ground truth labels for given `label_matrix`
        metrics (list): a list of metrics from `sklearn.metrics` [#]_
        set (str): ``'train'`` or ``'test'`` set (affects print and logging)
        **kwargs (dict): additional keyword arguments to pass to `label_model.score`:
            1. `tie_break_policy`. Defaults to ``'abstain'``.
                See `snorkel.labeling.model.LabelModel.score` for more info.

    .. [#] https://snorkel.readthedocs.io/en/latest/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html
    .. [#] https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

    """
    tie_break_policy = kwargs['tie_break_policy'] if 'tie_break_policy' in kwargs.keys() else 'abstain'

    label_model_metrics = label_model.score(L=label_matrix, Y=gold_labels,
                                            tie_break_policy=tie_break_policy,
                                            metrics=metrics)
    logger.info('Label Model {}ing stats: '.format(set))
    for m in metrics:
        logger.info('Label Model {}ing {}: {:.1f}%'.format(
            set, m, label_model_metrics[m] * 100))
