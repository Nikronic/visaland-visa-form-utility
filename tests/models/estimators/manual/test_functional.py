from math import isclose

from pytest import mark

from vizard.models.estimators.manual.functional import extend_mean, truncated_scaler


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


@mark.parametrize(
    argnames=[
        "given_x",
        "given_x_min_range",
        "given_x_max_range",
        "given_target_min_range",
        "given_target_max_range",
        "expected_output",
    ],
    argvalues=[
        (250, 100, 600, -0.05, 0.07, -0.013999999999999999),
        (500, 100, 600, -0.05, 0.07, 0.04600000000000001),
    ],
)
def test_truncated_scaler(
    given_x,
    given_x_min_range,
    given_x_max_range,
    given_target_min_range,
    given_target_max_range,
    expected_output,
):
    output = truncated_scaler(
        x=given_x,
        x_min_range=given_x_min_range,
        x_max_range=given_x_max_range,
        target_min_range=given_target_min_range,
        target_max_range=given_target_max_range,
    )

    assert isclose(expected_output, output)
