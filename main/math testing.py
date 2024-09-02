import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import exit
import func
from math import sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# all my incoherent math testing :)

x = [-2,-1,0,1,2,3]

x = np.random.uniform(-2,7, 6)
x = np.sort(x)
x = np.round(x, 2)
x = x.tolist()


roughness = np.random.uniform(1,5,6)
roughness = np.round(roughness, 2)


# x = np.random.uniform(0.4, 50, 6)
# x = np.sort(x)  # Sort x values
# x = np.round(x, 2)
# x_adjusted = x + np.maximum(-np.min(x) + 0.1, 0)


random_coeffs = np.random.uniform(0.1, 7, 4)
coeffs = np.round(random_coeffs, 0)


y = [coeffs[0] * np.log(coeffs[1] * x_adj + coeffs[2]) + coeffs[3] for x_adj in x]
y = np.round(y, 2)


points = list(zip(x, y))


print("Points for ln:")
print(f"x = {x}")
print(f"y = {y.tolist()}")




random_coeffs = np.random.uniform(1, 2, 4)
coeffs = np.round(random_coeffs, 0)

y=[]
y = [coeffs[0]*np.sin(coeffs[1]*x_adj + coeffs[2]) + coeffs[3] for x_adj in x]


y = np.round(y, 2)

points = list(zip(x, y))

print("Points for sine:")
print(f"x = {x}")
print(f"y = {y.tolist()}")


random_coeffs = np.random.uniform(-2, 4, 6)
coeffs = np.round(random_coeffs, 0)

y=[]
y = [coeffs[0]*x_adj*x_adj*x_adj + coeffs[1]*x_adj*x_adj + coeffs[2]*x_adj + coeffs[3] for x_adj in x]
roughness = []




y = np.round(y, 2)

points = list(zip(x, y))

print("Points for Poly:")
print(f"x = {x}")
print(f"y = {y.tolist()}")


random_coeffs = np.random.uniform(1, 2, 4)
coeffs = np.round(random_coeffs, 0)

y=[]
y = [coeffs[0]*np.exp(coeffs[1]*x_adj + coeffs[2]) + coeffs[3] for x_adj in x]



y = np.round(y, 2)

points = list(zip(x, y))

print("Points for Exp:")
print(f"x = {x}")
print(f"y = {y.tolist()}")
