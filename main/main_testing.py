import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

# used mainly for testing outside of website.

x = [1,2,3,4,5,6]
y = [16,2,-2,0,3,7]


x=[-6.01, -5.46, -4.90, -4.52, -3.79, -3.72]

y=[3.90, -2.06, 4.04, -2.10, 3.97, -2.00]


predicted_function = func.predict_function(x,y)
func.sine_average(x,y)
n= func.predict_degree(x,y)
exit()

print(f'Your function: {predicted_function}')

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)
if predicted_function == "ln":
    func.ln_average(x,y)
