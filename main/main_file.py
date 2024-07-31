import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys


x = [1,2,3,4,5,6]
y = [1,4,9,16,25,36]

# Points for sine:
x = [-1.38, 0.01, 3.21, 3.3, 3.73, 6.8]
y = [0.02, 1.85, 1.91, 1.97, 1.82, 1.89]
# Points for Poly:
x = [-1.38, 0.01, 3.21, 3.3, 3.73, 6.8]
y = [-3.93, 1.02, 137.56, 148.08, 205.88, 1096.62]
# Points for Exp:
# x = [-1.38, 0.01, 3.21, 3.3, 3.73, 6.8]
# y = [2.94, 17.08, 9075.81, 10865.32, 25673.77, 11913078.03]



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
