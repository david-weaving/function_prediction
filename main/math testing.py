
# this mess is purely a playground for math related things and testing specific functions


import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from math import pi
import cmath
from scipy.optimize import curve_fit
from sys import exit


x=[-2,-1,0,1,2,4]
y=[7.4,2.72,1,0.37,0.14,0.02]


x = [-5.0, -3.0, -1.0]
y = [148, 20.4, 2.9]

x = [-2,-1.3,0,2]
y = [4,2.6,1.5,1]

x = [-2,-1.01,0]
y=[4,0,-2]



# new code seems to work for both decay and growth with a better fitting for guesses
def exp_average(x, y):
    # Define the exponential model fitting function
    def exp_model(x, A, b, C):
        return A * np.exp(b * x) + C

    # Initial guess for parameters
    def initial_guess(x_points, y_points):
        A_guess = (np.max(y_points) - np.min(y_points)) / (np.exp(np.max(x_points)) - np.exp(np.min(x_points)))
        
        try:
            b_guess = np.log(y_points[-1] / y_points[0]) / (x_points[-1] - x_points[0])
            if np.isnan(b_guess):
                b_guess = 0.1  # Default guess for b if logarithm calculation fails
        except ValueError:
            b_guess = 0.1  # Default guess for b if logarithm calculation fails
        
        C_guess = np.min(y_points)  # Initialize C_guess based on the minimum y-value
        
        return [A_guess, b_guess, C_guess]

    # Fit the model to the data
    try:
        params, _ = curve_fit(exp_model, x, y, p0=initial_guess(x, y), maxfev=2000)
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return

    A_fit, b_fit, C_fit = params
    if C_fit > 0 or C_fit == 0:
        print(f"Exp Function: y = {A_fit} + e^{b_fit}x + {C_fit}")
    else:
        print(f"Exp Function: y = {A_fit} + e^{b_fit}x - {abs(C_fit)}")

    # Generate model values for plotting
    x_common = np.linspace(np.min(x), np.max(x), 400)
    y_fit = exp_model(x_common, A_fit, b_fit, C_fit)

    # Plotting


    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Data Points')
    plt.plot(x_common, y_fit, color='red', label='Fitted Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Exponential Curve Fitting')
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage:
# x = [0.1, 0.2, 0.3, 0.45, 0.7]
# y = [0.37, 0.14, 0.05, 0.01, 0.0009]
exp_average(x, y)



