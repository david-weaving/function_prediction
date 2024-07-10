import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import exit
import func
from math import sqrt
from scipy.optimize import curve_fit

x = [10,11,12,13]
y = [7.90,8.10,8.29,8.48]

# for A*sqrt(x + C) + D

# A is pretty accurate, so is C

def sqrt_func(x,A,C,D):
   return A*np.sqrt(x + C) + D

A = (y[0]*np.sqrt(y[2]*y[2]-y[1]*y[1]))/(np.sqrt(x[0]*(y[2]*y[2]-y[1]*y[1])+y[1]*y[1]*x[2]-x[1]*y[2]*y[2]))

C = (y[1]*y[1]*x[2] - x[1]*y[2]*y[2])/(y[2]*y[2]-y[1]*y[1])

D = y[3]/(A*np.sqrt(x[3]+C))

initial_guess = [A,C,D]

params, _ = curve_fit(sqrt_func, x,y, p0=initial_guess,maxfev=5000)

A,C,D = params

print(f"A = {A}")
print(f"C = {C}")
print(f"D = {D}")

