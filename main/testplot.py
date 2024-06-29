import itertools
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- THE CONTROL PANEL -----------------------  
x = [3,18,2,25,35,41]  # x values from the user
y = [1,5,8,15,25,40]   # y values from the user

#x = [-20,-10,0,10,22,30]
#y = [20,9,7,14,25,0]


x_graph_limit_low = -10
x_graph_limit_high = 1.2
y_graph_limit_low = -5.5
y_graph_limit_high = 1.6

n = 2 # degree of polynomial

def append_row(matrix, new_row):  # function to append rows into matrix
    return np.vstack([matrix, new_row])

def fit_polynomial(x_points, y_points, degree):
    A = np.vander(x_points, degree + 1)
    b = np.array(y_points)
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    return coeffs
# Generate all combinations of 3 points out of 6
combinations = list(itertools.combinations(range(len(x)), 3))

# Common x values to evaluate the polynomial on
x_min = np.min(x)
x_max = np.max(x)
y_max = np.max(y)
y_min = np.min(y)
x_common = np.linspace(x_min, x_max, 400)

# List to store all the polynomial values
all_y_values = []

for combo in combinations:
    x_points = [x[i] for i in combo]
    y_points = [y[i] for i in combo]
    coeffs = fit_polynomial(x_points, y_points, n)
    y_values = np.polyval(coeffs, x_common)
    all_y_values.append(y_values)

# Average the y values from all polynomials
y_average = np.mean(all_y_values, axis=0)

# Plotting
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
