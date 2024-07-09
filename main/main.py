import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
y = [35.0, 8.0, 1.0, -2.0, 3.0, 46.0]


predicted_function = func.predict_function(x,y)

print(f'Your function: {predicted_function}')
n = 4 # degree of polynomial


if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)

