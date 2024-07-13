import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [1,2,3,4,5,6]
y = [1,4,9,16,25,36]

x = [-3.00, -1.50, 0.00, 1.50, 3.00, 4.50]
y = [12.50, 4.75, 1.00, 2.75, 4.50, 5.75]


predicted_function = func.predict_function(x,y)



print(f'Your function: {predicted_function}')

n = 3 # degree of polynomial

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)
if predicted_function == "ln":
    func.ln_average(x,y)


