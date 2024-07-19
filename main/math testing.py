import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import exit
import func
from math import sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


x = [-2,-1,0,1,2,3]

x = np.random.uniform(-1,4, 6)
x = np.sort(x)  # Sort x values
x = np.round(x, 2)
x = x.tolist()

# make a function to generate points for all functions


roughness = np.random.uniform(1,5,6)
roughness = np.round(roughness, 2)

# Generate 6 random x values in the range [-4, 5)
# x = np.random.uniform(0.4, 50, 6)
# x = np.sort(x)  # Sort x values
# x = np.round(x, 2)
# Ensure positive arguments for log function
#x_adjusted = x + np.maximum(-np.min(x) + 0.1, 0)

# Generate random coefficients
random_coeffs = np.random.uniform(0.1, 7, 4)
coeffs = np.round(random_coeffs, 0)

# Calculate y values
y = [coeffs[0] * np.log(coeffs[1] * x_adj + coeffs[2]) + coeffs[3] for x_adj in x]
y = np.round(y, 2)  # Round y values to two decimal places



# Combine x and y into points
points = list(zip(x, y))

# Print points and individual arrays for verification
print("Points for ln:")
#print(points)
print(f"x = {x}")
print(f"y = {y.tolist()}")



# Generate random coefficients
random_coeffs = np.random.uniform(1, 2, 4)
coeffs = np.round(random_coeffs, 0)

y=[]
# Calculate y values
y = [coeffs[0]*np.sin(coeffs[1]*x_adj + coeffs[2]) + coeffs[3] for x_adj in x]



# y = y + roughness
y = np.round(y, 2)  # Round y values to two decimal places

# Combine x and y into points
points = list(zip(x, y))

print("Points for sine:")
#print(points)
print(f"x = {x}")
print(f"y = {y.tolist()}")



# Generate random coefficients
random_coeffs = np.random.uniform(-2, 4, 4)
coeffs = np.round(random_coeffs, 0)

y=[]
# Calculate y values
y = [coeffs[0]*x_adj*x_adj + x_adj*coeffs[1] + coeffs[2] for x_adj in x]
roughness = []



# y = y + roughness
y = np.round(y, 2)  # Round y values to two decimal places

# Combine x and y into points
points = list(zip(x, y))

print("Points for Poly:")
#print(points)
print(f"x = {x}")
print(f"y = {y.tolist()}")



# Generate random coefficients
random_coeffs = np.random.uniform(2, 4, 4)
coeffs = np.round(random_coeffs, 0)

y=[]
# Calculate y values
y = [coeffs[0]*np.exp(coeffs[1]*x_adj + coeffs[2]) + coeffs[3] for x_adj in x]



# y = y + roughness
y = np.round(y, 2)  # Round y values to two decimal places

# Combine x and y into points
points = list(zip(x, y))

print("Points for Exp:")
#print(points)
print(f"x = {x}")
print(f"y = {y.tolist()}")
