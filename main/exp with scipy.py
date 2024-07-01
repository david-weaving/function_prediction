import numpy as np
import itertools
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt


x_graph_limit_low = -10
x_graph_limit_high = 2.2
y_graph_limit_low = -6.5
y_graph_limit_high = 1.6

# Define the exponential model fitting function
def fit_exp(x_points, y_points):
    def exp_model(x, A, b, C):
        return A * np.exp(b * x) + C
    
    # Initial guess for parameters A, b, and C
    initial_guess = [2.0, 0.1, 1.0]
    
    # Fit the model to the points
    params, _ = curve_fit(exp_model, x_points, y_points, p0=initial_guess)
    return params

# Define the exponential value evaluation function
def expval(A, b, C, x_common):
    return A * np.exp(b * x_common) + C

# Sample data points
x = np.array([-3,-2,-1,0])
y = np.array([7000, 200, 21,3])


x_min = np.min(x)
x_max = np.max(x)
y_max = np.max(y)
y_min = np.min(y)

# Generate all combinations of three points
combinations = list(itertools.combinations(range(len(x)), 3))

all_y_values = []
x_common = np.linspace(x_min, x_max, 400)
x_forward = np.linspace(x_max+0.1, 50, 400)
x_backward = np.linspace(-50, x_min-0.1, 400)
x_common = np.append(x_common, x_forward)
x_common = np.insert(x_common, 0, x_backward)

# Iterate over all combinations, fit the model, and evaluate it
for combo in combinations:
    x_points = [x[i] for i in combo]
    y_points = [y[i] for i in combo]
    
    try:
        A, b, C = fit_exp(x_points, y_points)
        y_values = expval(A, b, C, x_common)
        all_y_values.append(y_values)
    except RuntimeError as e:
        print(f"Error fitting combination {combo}: {e}")

# Compute the average y-values
all_y_values = np.array(all_y_values)
y_average = np.mean(all_y_values, axis=0)

# Output the average y-values
#print("Average y-values:", y_average)


plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points',zorder=2)
plt.plot(x_common, y_average, color='red', label='Average Graph', zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Averaged Graph')
plt.grid(True)
plt.xlim(abs(x_min)*x_graph_limit_low, abs(x_max)*x_graph_limit_high) # for limits on the plots
plt.ylim(abs(y_min)*y_graph_limit_low, abs(y_max)*y_graph_limit_high)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_common, y_average, color='red', label='Average Graph',zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Full Graph')
plt.grid(True)
plt.legend()
plt.show()