def extend_mean(percent: float, new_value: float) -> float:
    """Takes a percent and adds a new value while keep the percent standardized

    This method is to take a standardized value ``percent`` (in range ``[0, 1]``), and takes
    another standard value ``new_value``. Then adds this value to the ``percent`` while
    keeping ``percent`` standardized (i.e., in range of ``[0, 1]``)

    Args:
        percent (float): A number in range of ``[0, 1]``
        new_value (float): the value to be added to ``percent`` in range of ``[0, 1]``

    Returns:
        float: Newly standardized value of ``percent`` in range of ``[0, 1]``
    """

    new_percent: float = percent + new_value * (1 - percent)
    return new_percent
