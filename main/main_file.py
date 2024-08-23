import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys


x = [1,2,3,4,5,6]
y = [16,2,-2,0,3,7]


x = [-4.67,-2.87,-0.0222,2.60,4.44,6.36]
y = [-2.13,-5.56,-6.93,-5.56,-1.93,2.90]

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
