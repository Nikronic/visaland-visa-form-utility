import numpy as np
from vizard.seduce import functional


def test_sigmoid():
    input_numpy = np.linspace(1, 100, 5) / 100
    input_float = 0.4
    scaler1 = 8

    output_float = functional.sigmoid(input_float, scaler1)
    correct_float = 0.9608342772032357
    assert output_float == correct_float, "float sigmoid failed"
    output_numpy: np.ndarray = functional.sigmoid(input_numpy, scaler1)
    correct_numpy = np.array(
        [0.51998934, 0.88695417, 0.98270684, 0.99757622, 0.99966465]
    )
    assert np.isclose(output_numpy, correct_numpy).all(), np.isclose(
        output_numpy, correct_numpy
    )


def test_adjusted_sigmoid():
    input_numpy = np.linspace(1, 100, 5) / 100
    input_float = 0.4
    adjusted_min = 0.3
    adjusted_max = 0.7
    scaler1 = 8
    output_float = functional.adjusted_sigmoid(
        input_float, adjusted_min, adjusted_max, scaler1
    )
    correct_float = 0.3855638786081178
    assert np.isclose(
        output_float, correct_float
    ).all(), "float adjusted sigmoid failed"
    output_numpy: np.ndarray = functional.adjusted_sigmoid(
        input_numpy, adjusted_min, adjusted_max, scaler1
    )
    correct_numpy = np.array([0.01, 0.2575, 0.50602296, 0.7525, 1.0])
    assert np.isclose(
        output_numpy, correct_numpy
    ).all(), "np array adjusted sigmoid failed"


def test_bi_level_adjusted_sigmoid():
    input_numpy = np.linspace(1, 100, 5) / 100
    input_float = 0.4
    adjusted_min = 0.3
    adjusted_max = 0.7
    closer_adjusted_min = 0.4
    closer_adjusted_max = 0.6
    scaler1 = 8
    scaler2 = 2
    output_float = functional.bi_level_adjusted_sigmoid(
        input_float,
        adjusted_min,
        adjusted_max,
        closer_adjusted_min,
        closer_adjusted_max,
        scaler1,
        scaler2,
    )
    correct_float = 0.39900662908474405
    assert np.isclose(
        output_float, correct_float
    ).all(), "float bi level adjusted sigmoid failed"
    output_numpy: np.ndarray = functional.bi_level_adjusted_sigmoid(
        input_numpy,
        adjusted_min,
        adjusted_max,
        closer_adjusted_min,
        closer_adjusted_max,
        scaler1,
        scaler2,
    )
    correct_numpy = np.array([0.01, 0.2575, 0.50526316, 0.7525, 1])
    assert np.isclose(
        output_numpy, correct_numpy
    ).all(), "np array bi level adjusted sigmoid failed"
