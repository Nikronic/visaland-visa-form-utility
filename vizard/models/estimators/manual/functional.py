from typing import Optional


def extend_mean(
    percent: float, new_value: float, shift: Optional[float] = None
) -> float:
    """Takes a percent and adds a new value while keep the percent standardized

    This method is to take a standardized value ``percent`` (in range ``[0, 1]``), and takes
    another standard value ``new_value``. Then adds this value to the ``percent`` while
    keeping ``percent`` standardized (i.e., in range of ``[0, 1]``). The ``shift`` enables
    pushing the value toward 0 if negative and 1 if positive, proportionate to ``percent``
    value.

    Args:
        percent (float): A number in range of ``[0, 1]``
        new_value (float): the value to be added to ``percent`` in range of ``[0, 1]``
        shift (Optional[float], optional): A standardized value to push the final
            result around proportionate to the ``percent`` value. Defaults to None.
            When None, the value is set to the value of ``new_value``.

    Returns:
        float: Newly standardized value of ``percent`` in range of ``[0, 1]``
    """

    # the old behavior: new_percent = percent - (new_value * percent) + new_value
    if shift is None:
        shift = -new_value

    new_percent: float = percent + (shift * percent) + new_value
    return new_percent


def truncated_scaler(
    x: float,
    x_min_range: float,
    x_max_range: float,
    target_min_range: float,
    target_max_range: float,
) -> float:
    """Scales a value to a desired range given range of the input

    Args:
        x (float): Input value to be scaled
        x_min_range (float): denote the minimum of the range of input
        x_max_range (float): denote the maximum of the range of input
        target_min_range (float): denote the minimum of the range of desired target scaling
        target_max_range (float): denote the maximum of the range of desired target scaling

    Returns:
        float: The scaled value in range of ``[target_min_range, target_max_range]``

    Reference:
        - https://stats.stackexchange.com/a/281164/216826
    """

    interpolated: float = 0.0
    interpolated = (x - x_min_range) / (x_max_range - x_min_range)
    interpolated = (
        interpolated * (target_max_range - target_min_range) + target_min_range
    )
    return interpolated
