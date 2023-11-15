import math
import matplotlib.pyplot as plt
import numpy as np

def adjusted_sigmoid(x):
    # Check input
    if (0.37<=x<=0.67):
    # Shift and scale the input to fit the sigmoid function

        # Sigmoid function
        x = x - 0.52 

        #a = 2.77 = (adjusted_sigmoid(0.67) - adjusted_sigmoid(0.37)) / (0.67-0.37)
        #b = 0.34 = 0.37 - adjusted_sigmoid(0.37)/a
        sigmoid = (((1 / (1 + math.exp(-16*x))/2.7788486900405176666666666666667))+0.340069368371147392)
        # sigmoid = 1 / (1 + math.exp(-16*x))

        return sigmoid 
    else:
        return x

def adjusted_sigmoid2(x):
    return x



print(adjusted_sigmoid(0.37))

chances = [i/100 for i in range(101)]

sigmoid_values = [adjusted_sigmoid(chance) for chance in chances]
sigmoid_values2 = [adjusted_sigmoid2(chance) for chance in chances]

# Plot the adjusted sigmoid function
plt.plot(chances, sigmoid_values,label="adjusted")
plt.plot(chances, sigmoid_values2,label="without change")
plt.title('adjusted Sigmoid Function')
plt.xlabel('Chance')
plt.ylabel('adjusted Sigmoid Value')
plt.axvline(x=0.37, color='red', linestyle='--', label='Vertical at x=0.37')
plt.axvline(x=0.67, color='green', linestyle='--', label='Vertical at x=0.67')
plt.legend()
plt.grid(True)
plt.show()


