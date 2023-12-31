import numpy as np
import pytest

from vizard.models.trainers.aml_flaml import get_loss_score, sklearn_metric_loss_score


def test_get_loss_score():
    # test single metric string
    y_predict = np.array([0, 1, 1])
    y_true = np.array([0, 0, 1])

    metrics_names = "accuracy"
    metrics_values = sklearn_metric_loss_score(
        metric_name=metrics_names, y_predict=y_predict, y_true=y_true
    )
    result = get_loss_score(y_predict, y_true, [metrics_names])
    assert result == {metrics_names: metrics_values}

    # test multiple metrics list
    metrics_names = ["accuracy", "f1"]
    metrics_values = [
        sklearn_metric_loss_score(y_predict=y_predict, y_true=y_true, metric_name=mn)
        for mn in metrics_names
    ]
    result = get_loss_score(y_predict, y_true, metrics_names)
    assert result == {k: v for k, v in zip(metrics_names, metrics_values)}

    # test custom metric class
    class MyMetric:
        def __call__(self, y_predict, y_true):
            return 0.75

    metrics_names = MyMetric
    result = get_loss_score(y_predict, y_true, metrics_names)
    assert result == {"custom_MyMetric": 0.75}
