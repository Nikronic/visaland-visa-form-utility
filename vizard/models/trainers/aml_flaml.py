# core
import numpy as np
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
# helpers
from typing import Any, Dict, List, Union, Callable
import logging


# configure logging
logger = logging.getLogger(__name__)


def get_loss_score(
    y_predict: np.ndarray,
    y_true: np.ndarray,
    metrics: Union[List[str], str, Callable]
) -> Dict[str, Any]:
    """Gives loss score given predicted and true labels and metrics

    Args:
        y_predict (np.ndarray): Predicted labels, same shape as ``y_true``
        y_true (np.ndarray): Ground truth labels, same shape as ``y_predict``
        metrics (Union[List[str], str, callable]): ``metrics`` can be either
            a metric name (``str``) or a list of metric names that is supported
            by ``flaml.ml.sklearn_metric_loss_score``.

            ``metrics`` can also be a custom metric which in that case must be class
            that implements ``__call__`` method with following signature:::

                def __call__(
                    X_test, y_test, estimator, labels,
                    X_train, y_train, weight_test=None, weight_train=None,
                    config=None, groups_test=None, groups_train=None,
                ):
                    return metric_to_minimize, metrics_to_log

    Returns:
        Dict[str, Any]:
        Dictionary of ``{'metric_name': metric_value}`` for all given
        ``metrics``.

    See Also:
        * :func:`report_loss_score`
    """
    # name of metrics for logging purposes
    metrics_names: List[str] = []
    # actual metrics param for APIs that need name of metrics e.g. sklearn
    metrics_values = []

    # if `metrics` is one of flaml's metrics
    if isinstance(metrics, str):
        metrics_names = [metrics]
        metrics_values = sklearn_metric_loss_score(
            metric_name=metrics,
            y_predict=y_predict,
            y_true=y_true
        )
    # if `metrics` is a list of metrics of flaml's metrics
    elif isinstance(metrics, list):
        metrics_names = metrics
        metrics_values = [
            sklearn_metric_loss_score(
                metric_name=metric,
                y_predict=y_predict,
                y_true=y_true) for metric in metrics
        ]
    # if `metrics` is a class
    # TODO: fix complaining that mypy does not understand Callable (herald :D)
    elif isinstance(metrics, Callable):  # type: ignore
        metrics = metrics()
        metrics_names = [f'custom_{metrics.__class__.__name__}']
        metrics_values = [
            metrics(y_predict=y_predict, y_true=y_true)  # type: ignore
        ]

    # return dictionary of metrics values and their names
    metrics_name_value: dict = {}
    for metric_name, metric_value in zip(metrics_names, metrics_values):
        metrics_name_value[metric_name] = metric_value
    return metrics_name_value


def report_loss_score(metrics: Dict[str, Any]) -> str:
    """Prints a dictionary of ``{'metric_name': metric_value}``

    Such a dictionary can be produced via :func:`get_loss_score`.

    Args:
        metrics (Dict[str, Any]): Dictionary of ``{'metric_name': metric_value}``

    Returns: 
        str: 
        A string containing the loss score and their corresponding names
        in a new line. e.g.::

            'accuracy: 0.97'
            'f1: 0.94'

    """
    msg: str = ''
    for metric_name, metric_value in metrics.items():
        if is_score_or_loss(metric_name):
            msg += f'{metric_name} score: {1 - metric_value:.2f}\n'
        else:
            msg += f'{metric_name} loss: {metric_value:.2f}\n'
    return msg


def is_score_or_loss(metric: str) -> bool:
    """Check if metric is a score or loss

    If metric is a loss (the lower the better), then the value itself
    will be reported. If metric is a score (the higher the better), then
    the ``1 - value`` will be reported.

    The reason is that ``flaml`` uses ``1 - value`` to minimize the error
    when users' chosen ``metric`` is a **score** rather than a **loss**.

    Args:
        metric (str): metric name that is supported by ``flaml``. For 
            more info see :func:`flaml.ml.sklearn_metric_loss_score`.

    See Also:
        * :func:`report_loss_score <vizard.models.trainer.aml_flaml.report_loss_score>`

    Returns:
        bool: If is a score then return ``True``, otherwise return ``False``
    """
    # determine if it is 'score' (maximization) or 'loss' (minimization)
    result = False
    if metric in [
        'r2',
        'accuracy',
        'roc_auc',
        'roc_auc_ovr',
        'roc_auc_ovo',
        'f1',
        'ap',
        'micro_f1',
        'macro_f1',
    ]:
        result = True
    return result


def report_feature_importances(
    estimator: Any,
    feature_names: List[str]
) -> str:
    """Prints feature importances of an fitted ``flaml.AutoML`` instance

    Args:
        estimator (Any): :class:`flaml.AutoML` underlying estimator that has ``feature_importances_``
            attribute, i.e. pass ``estimator`` in ``flaml_obj.model.estimator.feature_importances_``. 
        feature_names (List[str]): List of feature names. One can pass
            columns of original dataframe as ``feature_names``

    Returns:
        str: A string containing the feature importances in a new line. e.g.::

            'feature_1: 0.1'
            'feature_2: 0.0'
            'feature_2: 0.4'
            ...

    """
    msg: str = ''
    feature_importance = estimator.feature_importances_
    for feature_name, feature_importance in zip(feature_names,
                                                feature_importance):
        msg += f'{feature_name}: {feature_importance:.2f}\n'
    return msg
