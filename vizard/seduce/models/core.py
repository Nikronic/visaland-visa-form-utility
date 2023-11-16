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
        for more information check :func:vizard.seduce.functional.sigmoid

        """
        return functional.sigmoid(x, scaler)

    def adjusted_sigmoid(self, x, adjusted_min, adjusted_max, scaler):
        """apply sigmoid to some parts of our function
        for more information check :func:vizard.seduce.functional.adjusted_sigmoid
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
        closer_adjusted_min: float,
        closer_adjusted_max: float,
        scaler1: int | float,
        scaler2: int | float,
    ) -> np.ndarray:
        """apply sigmoid to limited parts of function focus more on closer part to middle of our adjustment
        for more information check :func:vizard.seduce.functional.bi_level_adjusted_sigmoid
        """
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
