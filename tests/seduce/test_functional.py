import numpy as np
from vizard.seduce import functional

input_numpy = np.linspace(1, 100, 5) / 100
input_float = 0.4
adjusted_min = 0.3
adjusted_max = 0.7
closer_adjusted_min = 0.4 
closer_adjusted_max = 0.6
scaler1 = 8
scaler2 = 2


def test_sigmoid():
    output_float = functional.sigmoid(input_float, scaler1)
    correct_float = 0.9608342772032357
    assert output_float == correct_float, "float sigmoid failed"
    output_numpy: np.ndarray = functional.sigmoid(input_numpy, scaler1)
    correct_numpy = np.array(
        [0.50499983, 0.62597786, 0.73302015, 0.81831902, 0.88079708]
    )
    assert output_numpy.all() == correct_numpy.all(), "np array sigmoid failed"

def test_adjusted_sigmoid():
    output_float = functional.adjusted_sigmoid(input_float,adjusted_min,adjusted_max, scaler1)
    correct_float = 0.3855638786081178
    assert output_float == correct_float, "float adjusted sigmoid failed"
    output_numpy: np.ndarray = functional.adjusted_sigmoid(input_float,adjusted_min,adjusted_max, scaler1)
    correct_numpy = np.array([0.01,0.2575,0.50506645, 0.7525,1.0])
    assert output_numpy.all() == correct_numpy.all(), "np array adjusted sigmoid failed"

def test_bi_level_adjusted_sigmoid():
    output_float = functional.bi_level_adjusted_sigmoid(input_float, adjusted_min, adjusted_max, closer_adjusted_min, closer_adjusted_max, scaler1, scaler2)
    correct_float = 0.39900662908474405
    assert output_float == correct_float, "float bi level adjusted sigmoid failed"
    output_numpy: np.ndarray = functional.bi_level_adjusted_sigmoid(input_numpy, adjusted_min, adjusted_max, closer_adjusted_min, closer_adjusted_max, scaler1, scaler2)
    correct_numpy = np.array([0.01, 0.2575, 0.50526316, 0.7525, 1])
    assert output_numpy.all() == correct_numpy.all(), "np array bi level adjusted sigmoid failed"
