import numpy as np

def sigmoid(x, scaler):
    """sigmoid function with a custom scaler 
    Args:
        x (int | float | np.ndarray): Input of the sigmoid function
        scaler (int): multiply inputs with scaler number, use 1 for normal sigmoid 

    Returns:
        int | float | np.ndarray:
            output of the sigmoid function with our customized scaler --> f(x) = 1 / (1 + e^(-scaler*x))

    """
    return 1 / (1 + np.exp(-scaler * x))

def adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler):
    """apply sigmoid to some parts of our function
    Args:
        x (int | float | np.ndarray): Input of the sigmoid function
        adjusted_min (float): start of our adjustment inputs before that wont get any changes
        adjusted_max (float): end of our adjustment, inputs after that wont get any changes
        scaler (int): multiply inputs with scaler number, use 1 for normal sigmoid 

    Returns:
        int | float | np.ndarray:
            for inputs of x, adjusted_min < x < adjusted_max gives the
            output of the sigmoid function with our customized scaler --> f(x) = 1 / (1 + e^(-scaler*x))
            for other inputs return them without any change

    """
    # Check input
    mask = np.logical_and(adjusted_min < x, x < adjusted_max)

    # Shift and scale the input to fit the sigmoid function
    adjusted_mean = (adjusted_min + adjusted_max) / 2
    sigmoid_input = x - adjusted_mean

    a = (sigmoid(adjusted_max - adjusted_mean, scaler) - \
        sigmoid(adjusted_min - adjusted_mean, scaler)) / (adjusted_max - adjusted_min)
    b = adjusted_min - sigmoid(adjusted_min - adjusted_mean, scaler) / a

    output = np.where(mask, (sigmoid(sigmoid_input, scaler) / a) + b, x)

    return output

def bi_level_adjusted_sigmoid(
        x: int | float | np.ndarray,
        adjusted_min: float,
        adjusted_max: float,
        closer_adjusted_min: float,
        closer_adjusted_max: float,
        scaler1: int | float,
        scaler2: int | float) -> np.ndarray:
    """apply sigmoid to limited parts of function focus more on closer part to middle of our adjustment

    This is an extension of :func:`vizard.seduce.functional.adjusted_sigmoid`

    Args:
        x (int | float | np.ndarray): Input of the sigmoid function
        adjusted_min (float): start of our adjustment inputs before that wont get any changes
        closer_adjusted_min (float): min for closer part to middle of our adjustment this part will be modified more
        # adjusted_max (float): end of our adjustment, inputs after that wont get any changes
        closer_adjusted_max (float): max for closer part to middle of our adjustment this part will be modified more
        scaler1 (int): multiply inputs with scaler number, use 1 for normal sigmoid, this scaler will be used in closer to middle part
        scaler2 (int): multiply inputs with scaler number, use 1 for normal sigmoid, this scaler will be used in farther to middle part

    Returns:
        int | float | np.ndarray:
            for inputs of x, closer_adjusted_min < x < closer_adjusted_max gives the
            output of the sigmoid function with our customized scaler --> f(x) = 1 / (1 + e^(-scaler1*x))
            for other inputs that are x, adjusted_min < x < adjusted_max gives the
            output of the sigmoid function with our customized scaler --> f(x) = 1 / (1 + e^(-scaler2*x))
            for any other inputs return them without any change

    """
    mask = np.logical_and(adjusted_min < x, x < adjusted_max)

    # Use NumPy's vectorized operations for better performance
    result = np.where(np.logical_and(closer_adjusted_min < x, x < closer_adjusted_max),
                    adjusted_sigmoid(x, closer_adjusted_min, closer_adjusted_max, scaler1),
                    adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler2))

    return np.where(mask, result, x)