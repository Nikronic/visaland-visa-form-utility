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
