from math import isclose
from vizard.models.estimators.manual.functional import extend_mean

from pytest import mark


@mark.parametrize(
    argnames=["given_percent", "given_new_value", "given_shift", "expected"],
    argvalues=[
        (0.0, 0.3, -0.5, 0.3),  # left boundary
        (1.0, 0.3, -0.5, 0.8),  # right boundary
        (0.1, 0.8, -0.4, 0.86),  # middle points
        (0.8, 0.1, -0.4, 0.58),  # middle points
        (0.58, 0.1, -0.4, 0.448),  # middle points
        (0.8, 0.1, None, 0.82),  # middle points
    ],
)
def test_extend_mean(
    given_percent: float, given_new_value: float, given_shift, expected: float
):
    given = extend_mean(
        percent=given_percent, new_value=given_new_value, shift=given_shift
    )
    assert isclose(expected, given)
