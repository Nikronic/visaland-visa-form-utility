import math
import matplotlib.pyplot as plt

def sigmoid(x,scaler=16):
    return 1 / (1 + math.exp(-scaler*x))
                
def adjusted_sigmoid(x,adjusted_min,adjusted_max):
    # Check input
    if (adjusted_min<x<adjusted_max):

    # Shift and scale the input to fit the sigmoid function
        adjusted_mean = (adjusted_min+adjusted_max)/2
        # Sigmoid function
        sigmoid_input = x - adjusted_mean
        a = (sigmoid(adjusted_max-adjusted_mean,scaler) - sigmoid(adjusted_min-adjusted_mean,scaler))/(adjusted_max-adjusted_min)
        b = adjusted_min - sigmoid(adjusted_min-adjusted_mean,scaler)/a

        output = (sigmoid(sigmoid_input,scaler)/a)+b


        return output
    else:
        return x


adjusted_max = 0.65
adjusted_min = 0.35

scaler  = 32


print(adjusted_sigmoid(0.45,adjusted_min,adjusted_max))

chances = [i/100 for i in range(101)]

sigmoid_values = [adjusted_sigmoid(chance,adjusted_min,adjusted_max) for chance in chances]
sigmoid_values2 = [chance for chance in chances]

# Plot the adjusted sigmoid function
plt.plot(chances, sigmoid_values,label="adjusted")
plt.plot(chances, sigmoid_values2,label="without change")
plt.title('adjusted Sigmoid Function')
plt.xlabel('Chance')
plt.ylabel('adjusted Sigmoid Value')
plt.axvline(x=adjusted_min, color='red', linestyle='--', label='Vertical at x=0.35')
plt.axvline(x=adjusted_max, color='green', linestyle='--', label='Vertical at x=0.65')
plt.legend()
plt.grid(True)
plt.show()


