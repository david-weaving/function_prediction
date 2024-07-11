import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import exit
import func
from math import sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



import numpy as np
from scipy.optimize import curve_fit

# Define the logarithmic function
def ln_func(x, A, B, C, D):
    return A * np.log(B * x + C) + D

# Example data points (replace with your actual x and y arrays)
x = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
y = [3.07, 5.16, 6.38, 7.24, 7.9, 8.46]

# Initial guess for A, B, C, D
A_guess = np.max(y) - np.min(y)  # Approximate amplitude
B_guess = 1  # Scale factor
C_guess = 0.1  # Shift to avoid log(0)
D_guess = np.min(y)  # Vertical offset

initial_guess = [A_guess, B_guess, C_guess, D_guess]

# Fit the model to all points
params, _ = curve_fit(ln_func, x, y, p0=initial_guess, maxfev=5000)

# Extract the optimized parameters
A_opt, B_opt, C_opt, D_opt = params

print(f"A = {A_opt}")
print(f"B = {B_opt}")
print(f"C = {C_opt}")
print(f"D = {D_opt}")



x_min = np.min(x)
x_max = np.max(x)
y_max = np.max(y)
y_min = np.min(y)

x_common = np.linspace(x_min, x_max, 400)

# continue the graph
x_forward = np.linspace(x_max+0.1, 50, 400)
x_backward = np.linspace((0.1-C_opt)/B_opt, x_min-0.1, 400) # C_opt * -1 is the end bound (where we dont go below zero in our square root)
x_common = np.append(x_common, x_forward)
x_common = np.insert(x_common, 0, x_backward)

y_values = ln_func(x_common,A_opt,B_opt,C_opt,D_opt)

# Plotting
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1

x_plot_min = x_min - x_margin
x_plot_max = x_max + x_margin
y_plot_min = y_min - y_margin
y_plot_max = y_max + y_margin


plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points', zorder=2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('User Points')
plt.grid(True)
plt.xlim(x_plot_min, x_plot_max)
plt.ylim(y_plot_min, y_plot_max)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points', zorder=2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Natural Log Fitting')
plt.grid(True)
plt.xlim(x_plot_min, x_plot_max)
plt.ylim(y_plot_min, y_plot_max)
plt.plot(x_common, y_values, color='red', label='Average Graph', zorder=1)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_common, y_values, color='red', label='Average Graph', zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Natural Log Fitting (Full Graph)')
plt.grid(True)
plt.legend()
plt.show()