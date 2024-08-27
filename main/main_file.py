import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys


x = [1,2,3,4,5,6]
y = [16,2,-2,0,3,7]


x=[-3.97, -1.19, 1.48, 3.99, 6.54, 9.06]

y=[-8.93, -6.63, -4.80, -2.60, -0.60, 1.47]


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
