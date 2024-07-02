import numpy as np
import itertools
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential model fitting function
def fit_exp(x_points, y_points):
    def exp_model(x, A, b, c, E, C):
        return A * b ** (c * x + E) + C
    
    # Initial guess for parameters A, b, c, E, and C
    A_guess = (y_points[-1] - y_points[0]) / (np.exp(x_points[-1]) - np.exp(x_points[0]))
    b_guess = np.exp(np.log(y_points[-1] / y_points[0]) / (x_points[-1] - x_points[0]))
    c_guess = 1.0
    E_guess = 0.0
    C_guess = y_points[0] - A_guess * b_guess ** (c_guess * x_points[0] + E_guess)
    
    initial_guess = [A_guess, b_guess, c_guess, E_guess, C_guess]
    
    # Fit the model to the points
    params, _ = curve_fit(exp_model, x_points, y_points, p0=initial_guess)
    return params

# Define the exponential value evaluation function
def expval(A, b, c, E, C, x_common):
    return A * b ** (c * x_common + E) + C

# Sample data points
x = np.array([-4.2, -3.5, -2.8, -1.6, -0.9, 0.0, 1.2, 2.3, 3.4, 4.7])
y = np.array([0.5, 1.1, 2.3, 4.7, 9.2, 15.0, 27.5, 50.3, 91.8, 168.4])

x_min = np.min(x)
x_max = np.max(x)
y_max = np.max(y)
y_min = np.min(y)

# Generate all combinations of six points
combinations = list(itertools.combinations(range(len(x)), 6))

all_y_values = []
x_common = np.linspace(x_min, x_max, 400)

# Iterate over all combinations, fit the model, and evaluate it
for combo in combinations:
    x_points = [x[i] for i in combo]
    y_points = [y[i] for i in combo]
    
    try:
        A, b, c, E, C = fit_exp(x_points, y_points)
        y_values = expval(A, b, c, E, C, x_common)
        all_y_values.append(y_values)
    except RuntimeError as e:
        print(f"Error fitting combination {combo}: {e}")

# Compute the average y-values
all_y_values = np.array(all_y_values)
y_average = np.mean(all_y_values, axis=0)

# Plotting
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1

x_plot_min = x_min - x_margin
x_plot_max = x_max + x_margin
y_plot_min = y_min - y_margin
y_plot_max = y_max + y_margin

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points', zorder=2)
plt.plot(x_common, y_average, color='red', label='Average Graph', zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Averaged Graph')
plt.grid(True)
plt.xlim(x_plot_min, x_plot_max)
plt.ylim(y_plot_min, y_plot_max)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_common, y_average, color='red', label='Average Graph', zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Full Graph')
plt.grid(True)
plt.legend()
plt.show()
