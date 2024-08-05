import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys


x = [1,2,3,4,5,6]
y = [1,4,9,16,25,36]


x = [1,2,3,4,5,6]
y = [4,78,100,34,90,14]

x=[1,2,3,4,5,6]
y=[1,4,110,16,25,2000]

func.poly_avgerage(x,y,4)
func.sine_average(x,y)
# sys.exit()
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
