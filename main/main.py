import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [1,2,3,4,5,6]
y = [1,4,9,16,25,36]

x = [0.84, 3.2, 3.56, 5.23, 9.19, 9.2]
y = [33.97, 38.62, 39.12, 41.06, 44.24, 44.24]

predicted_function = func.predict_function(x,y)
func.ln_average(x,y)



print(f'Your function: {predicted_function}')
exit()
n = 2 # degree of polynomial

if predicted_function == "polynomial":
    func.poly_avgerage(x,y,n)
if predicted_function == "sine":
    func.sine_average(x,y)
if predicted_function == "exponential":
    func.exp_average(x,y)


