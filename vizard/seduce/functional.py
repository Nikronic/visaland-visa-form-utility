import numpy as np

class SigmoidSeducer:

    def __init__(self) -> None:
        pass

    def __type_check(x):
        if not isinstance(x, int) or not isinstance(x, float) or not isinstance(x, np.ndarray):
            raise ValueError(f'Type x {type(x)} is shitty.')

    def __sigmoid(self, x, scaler):

        self.__type_check(x)

        return 1 / (1 + np.exp(-scaler * x))

    def __adjusted_sigmoid(self, x, adjusted_min, adjusted_max, scaler):
        # Check input
        self.__type_check(x)
        mask = np.logical_and(adjusted_min < x, x < adjusted_max)

        # Shift and scale the input to fit the sigmoid function
        adjusted_mean = (adjusted_min + adjusted_max) / 2
        sigmoid_input = x - adjusted_mean

        a = (self.__sigmoid(adjusted_max - adjusted_mean, scaler) - \
            self.__sigmoid(adjusted_min - adjusted_mean, scaler)) / (adjusted_max - adjusted_min)
        b = adjusted_min - self.sigmoid(adjusted_min - adjusted_mean, scaler) / a

        output = np.where(mask, (self.sigmoid(sigmoid_input, scaler) / a) + b, x)

        return output

    def bi_level_adjusted_sigmoid(
            self,
            x: int | float | np.ndarray,
            adjusted_min: float,
            adjusted_max: float,
            closer_adjusted_min: float,
            closer_adjusted_max: float,
            scaler1: int | float,
            scaler2: int | float) -> np.ndarray:
        
        # type check
        self.__type_check(x)
        

        mask = np.logical_and(adjusted_min < x, x < adjusted_max)

        # Use NumPy's vectorized operations for better performance
        result = np.where(np.logical_and(closer_adjusted_min < x, x < closer_adjusted_max),
                        self.__adjusted_sigmoid(x, closer_adjusted_min, closer_adjusted_max, scaler1),
                        self.__adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler2))

        return np.where(mask, result, x)

