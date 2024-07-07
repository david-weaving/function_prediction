import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [-4, -3.5, -2.8, -1.6, -0.9, 0.0, 1.2]  # x values from the user
y = [0.5, 1.1, 2.3, 4.7, 9.2, 15.0, 27.5] # y values from the user

# x = [-3.0, -2.1, -1.5, -0.7, 0.0, 0.8, 1.5, 2.3, 3.0, 4.0]
# y = [5.0, 8.5, 2.3, 1.8, 1.0, 2.5, 3.2, 10.0, 20.1, 55.0]


# x = np.linspace(0, 10, 100)
# y = np.sin(x) + np.random.normal(0, 0.1, 100)  # Simulated noisy sine wave

x=[-1,-2,-3,-4,-5,-6]
y=[2,1.5,0.8,0.4,0.2,0.1]

n = 4 # degree of polynomial

#func.poly_avgerage(x,y,n)
#func.sine_average(x,y)
func.exp_average(x,y)
