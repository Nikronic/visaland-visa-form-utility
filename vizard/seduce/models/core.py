from typing import Optional

import numpy as np
from vizard.seduce import functional


class SigmoidSeducer:
    def __init__(self) -> None:
        pass

    def __type_check(x):
        if (
            not isinstance(x, int)
            or not isinstance(x, float)
            or not isinstance(x, np.ndarray)
        ):
            raise ValueError(f"Type x {type(x)} is not right.")

    def sigmoid(self, x, scaler):
        """sigmoid function with a custom scaler
        for more information check :func:`vizard.seduce.functional.sigmoid`

        """
        return functional.sigmoid(x, scaler)

    def adjusted_sigmoid(
        self,
        x: int | float | np.ndarray,
        adjusted_min: float,
        adjusted_max: float,
        scaler: Optional[int | float] = 1,
    ):
        """apply sigmoid to some parts of our function
        for more information check :func:`vizard.seduce.functional.adjusted_sigmoid`
        """
        # Check input
        self.__type_check(x)

        output = functional.adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler)

        return output

    def bi_level_adjusted_sigmoid(
        self,
        x: int | float | np.ndarray,
        adjusted_min: float,
        adjusted_max: float,
        closer_adjusted_min: Optional[float] = None,
        closer_adjusted_max: Optional[float] = None,
        scaler1: Optional[int | float] = None,
        scaler2: Optional[int | float] = 1,
    ) -> np.ndarray:
        """apply sigmoid to limited parts of function focus more on closer part to middle of our adjustment
        for more information check :func:`vizard.seduce.functional.bi_level_adjusted_sigmoid`
        """
        # check if user provided any value for optional values
        # if it was provided we use them
        # if not we use some we calculate them by provided values
        if scaler1 is None and scaler2 is None:
            scaler1 = scaler2 = 1
        elif scaler2 is None:
            scaler2 = scaler1 / 8
        elif scaler1 is None:
            scaler1 = scaler2 * 8
        scaler2 = 1

        if closer_adjusted_min is None or closer_adjusted_max is None:
            adjusted_mean = (adjusted_max + adjusted_min) / 2
            distance = adjusted_max - adjusted_min
            one_third = distance / 3
            closer_adjusted_max = adjusted_mean + one_third / 2
            closer_adjusted_max = adjusted_mean - one_third / 2

        result = functional.bi_level_adjusted_sigmoid(
            x,
            adjusted_min,
            adjusted_max,
            closer_adjusted_min,
            closer_adjusted_max,
            scaler1,
            scaler2,
        )

        # type check
        self.__type_check(x)

        return result
