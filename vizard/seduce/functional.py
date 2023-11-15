import math
import matplotlib.pyplot as plt

def sigmoid(x,scaler):
    return 1 / (1 + math.exp(-scaler*x))
                
def adjusted_sigmoid(x,adjusted_min,adjusted_max,scaler):
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

def bi_level_adjusted_sigmoid(x,adjusted_min,adjusted_max):
    if adjusted_min<=x<=adjusted_max:
        if closer_adjusted_min<x<closer_adjusted_max:
            return adjusted_sigmoid(x,closer_adjusted_min,closer_adjusted_max,scaler1)
        else:
            return adjusted_sigmoid(x,adjusted_min,adjusted_max,scaler2)
    else:
        return x

adjusted_max = 0.65
adjusted_min = 0.35
scaler1  = 128

scaler2  = scaler1/8

distance = adjusted_max - adjusted_min

# divide into two groups
one_third = distance/3
adjusted_mean = (adjusted_min+adjusted_max)/2

# closer part to 50
closer_adjusted_min = adjusted_mean - one_third/2
closer_adjusted_max = adjusted_mean + one_third/2




print(second_adjusted_sigmoid(0.43,adjusted_min,adjusted_max))

chances = [i/100 for i in range(101)]

sigmoid_values = [second_adjusted_sigmoid(chance,adjusted_min,adjusted_max) for chance in chances]
sigmoid_values2 = [chance for chance in chances]

# Plot the adjusted sigmoid function
plt.plot(chances, sigmoid_values,label="adjusted")
plt.plot(chances, sigmoid_values2,label="without change")
plt.title('adjusted Sigmoid Function')
plt.xlabel('Chance')
plt.ylabel('adjusted Sigmoid Value')
plt.axvline(x=adjusted_min, color='red', linestyle='--', label='Vertical at x=0.35')
plt.axvline(x=adjusted_max, color='red', linestyle='--', label='Vertical at x=0.65')

plt.legend()
plt.grid(True)
plt.show()
