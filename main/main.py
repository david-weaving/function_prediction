import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [77, 78, 79, 80, 81, 82]
y = [78, 79, 80, 81, 82, 83]
predicted_function = func.predict_function(x,y)

print(f'Your function: {predicted_function}')
n = 2 # degree of polynomial

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)
if predicted_function == "linear":
    func.poly_avgerage(x,y,1)
