import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [9,27,19,16,100,90]
y = [54,72,67,129,651,9]


predicted_function = func.predict_function(x,y)

print(f'Your function: {predicted_function}')
n = 5 # degree of polynomial


if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)