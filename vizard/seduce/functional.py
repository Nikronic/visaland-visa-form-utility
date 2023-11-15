import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, scaler):
    return 1 / (1 + np.exp(-scaler * x))

def adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler):
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
        x,
        adjusted_min,
        adjusted_max,
        closer_adjusted_min,
        closer_adjusted_max,
        scaler1,
        scaler2):
    mask = np.logical_and(adjusted_min < x, x < adjusted_max)

    # Use NumPy's vectorized operations for better performance
    result = np.where(np.logical_and(closer_adjusted_min < x, x < closer_adjusted_max),
                      adjusted_sigmoid(x, closer_adjusted_min, closer_adjusted_max, scaler1),
                      adjusted_sigmoid(x, adjusted_min, adjusted_max, scaler2))

    return np.where(mask, result, x)

adjusted_max = 0.65
adjusted_min = 0.35
scaler1 = 128
scaler2 = scaler1 / 8

distance = adjusted_max - adjusted_min
one_third = distance / 3
adjusted_mean = (adjusted_min + adjusted_max) / 2
closer_adjusted_min = adjusted_mean - one_third / 2
closer_adjusted_max = adjusted_mean + one_third / 2

#example
#print(bi_level_adjusted_sigmoid(0.47, adjusted_min, adjusted_max, closer_adjusted_min, closer_adjusted_max, scaler1, scaler2))

chances = np.linspace(0, 1, 101)

sigmoid_values = bi_level_adjusted_sigmoid(chances, adjusted_min, adjusted_max, closer_adjusted_min, closer_adjusted_max, scaler1, scaler2)

# Plot 

# plt.plot(chances, sigmoid_values, label="adjusted")
# plt.plot(chances, chances, label="without change")
# plt.title('Adjusted Sigmoid Function')
# plt.xlabel('Chance')
# plt.ylabel('Adjusted Sigmoid Value')
# plt.axvline(x=adjusted_min, color='red', linestyle='--')
# plt.axvline(x=adjusted_max, color='red', linestyle='--')
# plt.legend()
# plt.grid(True)
# plt.show()
