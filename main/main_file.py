import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys


x = [1,2,3,4,5,6]
y = [16,2,-2,0,3,7]


x = [-13.70, -8.45, -11.93, -4.70, -6.86, -4.32]
y = [9.38, -3.54, -5.67, -3.54, -3.28, 5.21]


predicted_function = func.predict_function(x,y)

n= func.predict_degree(x,y)

print(f'Your function: {predicted_function}')

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)
if predicted_function == "ln":
    func.ln_average(x,y)
