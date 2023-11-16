from typing import Optional

import numpy as np
from vizard.seduce import functional


class SigmoidSeducer:
    """use a customized sigmoid to manipulate numbers
        Numbers around 50% aren't intreating enough and give the user a uncertainty which is like what was the purpose of doing this form I am were i was before doing this shit and it tells me nothing
    Solution
        One way to do this is to use math a function like Sigmoid and some randomization to round numbers around 50 to some other numbers.
    """

    def __init__(self) -> None:
        pass

    def __type_check(self, x):
        if (
            not isinstance(x, int)
            and not isinstance(x, float)
            and not isinstance(x, np.ndarray)
        ):
            raise ValueError(f"Type x {type(x)} is not right.")

    def sigmoid(self, x, scaler):
        """sigmoid function with a custom scaler
        for more information check :func:`vizard.seduce.functional.sigmoid`


        """
        self.__type_check(x)

        return functional.sigmoid(x, scaler)

    def adjusted_sigmoid(
        self,
        x: int | float | np.ndarray,
        adjusted_min: float,
        adjusted_max: float,
        scaler: int | float = 1,
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
        scaler2: Optional[int | float] = None,
    ) -> np.ndarray:
        """apply sigmoid to limited parts of function focus more on closer part to middle of our adjustment
        for more information check :func:`vizard.seduce.functional.bi_level_adjusted_sigmoid`
        """
        self.__type_check(x)

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
            closer_adjusted_min = adjusted_mean - one_third / 2

        if closer_adjusted_min >= closer_adjusted_max:
            closer_adjusted_max = closer_adjusted_min + 1
        if adjusted_min >= adjusted_max:
            adjusted_max = adjusted_min + 1

        result = functional.bi_level_adjusted_sigmoid(
            x,
            adjusted_min,
            adjusted_max,
            closer_adjusted_min,
            closer_adjusted_max,
            scaler1,
            scaler2,
        )
        return result
