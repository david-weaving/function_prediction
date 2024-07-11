import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [1,2,3,4,5,6]
y = [1,4,9,16,25,36]

x = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
y = [2.7, 4.2, 5.6, 7.2, 8.1, 8.9]

predicted_function = func.predict_function(x,y)

print(f'Your function: {predicted_function}')
n = 2 # degree of polynomial

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)


