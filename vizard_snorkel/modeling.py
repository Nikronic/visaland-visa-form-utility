__all__ = ['report_label_model']


from snorkel.labeling.model import LabelModel
import numpy as np

from vizard_utils.helpers import loggingdecorator

# utils
from typing import List
import logging


# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@loggingdecorator(logger.name+'.func', level=logging.INFO, output=False, input=False)
def report_label_model(label_model: LabelModel, label_matrix, gold_labels: np.ndarray,
                       metrics: List, set: str, **kwargs) -> None:
    """
    Reports given `metrics` for the `snorkel.LabelModel`

    args:
        label_model: `snorkel.LabelModel` model (`torch.Module` base class)
        label_matrix: label matrix produced by applying `PandasLFApplier.apply` on a dataframe
        gold_labels: ground truth for given `label_matrix`
        metrics: a list of metrics from `sklearn.metrics`
        set: `'train'` or `'test'` set (affects print and logging)
    """
    tie_break_policy = kwargs['tie_break_policy'] if 'tie_break_policy' in kwargs.keys() else 'abstain'

    label_model_metrics = label_model.score(L=label_matrix, Y=gold_labels,
                                            tie_break_policy=tie_break_policy,
                                            metrics=metrics)
    logger.info('Label Model {}ing stats: '.format(set))
    for m in metrics:
        logger.info('Label Model {}ing {}: {:.1f}%'.format(
            set, m, label_model_metrics[m] * 100))
