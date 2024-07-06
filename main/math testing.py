
# this mess is purely a playground for math related things and testing specific functions


import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from math import pi
import cmath
from scipy.optimize import curve_fit
from sys import exit


def sine_average(x, y):
    # fitting sine function
    def fit_sine(x_points, y_points):

        # initial guess for the form Asin(Bx + D) + C
        A_guess = (np.max(y_points) - np.min(y_points)) / 2 # amplitude
        B_guess = 2 * np.pi / (np.max(x_points) - np.min(x_points))  # frequency
        D_guess = 0 # shift
        C_guess = np.mean(y_points) # off set
        
        initial_guess = [A_guess, B_guess, D_guess, C_guess]
        
        # Fit the model to the points
        params, _ = curve_fit(sineval, x_points, y_points, p0=initial_guess,maxfev=10000) # once again curve_fit helps us fit the curve given an f(x) input, in this case; f(x)=Asin(Bx+D)+C
        return params

    # sine function for evaluation
    def sineval(x, A, B, D, C):
        return A * np.sin(B * x + D) + C

    # for printing the function
    def print_sine(A,B,C,D):
        if D > 0 and C > 0:
            print(f"Your SINE function: {A}sin({B}x + {D}) + {C}")
        elif D < 0 and C > 0:
            print(f"Your SINE function: {A}sin({B}x - {abs(D)}) + {C}")
        elif D > 0 and C < 0:
            print(f"Your SINE function: {A}sin({B}x + {D}) - {abs(C)}")
        else:
            print(f"Your SINE function: {A}sin({B}x - {abs(D)}) - {abs(C)}")

    x_min = np.min(x)
    x_max = np.max(x)
    y_max = np.max(y)
    y_min = np.min(y)

    x_common = np.linspace(x_min, x_max, 400)

    # continue the graph
    x_forward = np.linspace(x_max+0.1, 50, 400)
    x_backward = np.linspace(-50, x_min-0.1, 400)
    x_common = np.append(x_common, x_forward)
    x_common = np.insert(x_common, 0, x_backward)


    try:
        A, B, D, C = fit_sine(x, y)
        y_values = sineval(x_common, A, B, D, C)
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        exit()

    print_sine(A,B,C,D)

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
    plt.title('Sine Fitting')
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
    plt.title('Sine Fitting (Full Graph)')
    plt.grid(True)
    plt.legend()
    plt.show()

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)  # Simulated noisy sine wave

sine_average(x, y)

