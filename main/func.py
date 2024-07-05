import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit

def rearrange_arrays(array1, array2):
    # Step 1: Combine and sort arrays
    combined = sorted(zip(array1, array2), key=lambda x: x[0])
    
    # Unzip sorted pairs back into separate arrays
    sorted_array1, sorted_array2 = zip(*combined)
    
    # Convert to lists for easier manipulation
    sorted_array1 = list(sorted_array1)
    sorted_array2 = list(sorted_array2)
    
    # Step 2: Extract highest, lowest, and third highest values
    highest = sorted_array1[-1]
    lowest = sorted_array1[0]
    third_highest = sorted_array1[-3]
    
    # Step 3: Remove highest, lowest, and third highest from sorted lists
    sorted_array1.remove(highest)
    sorted_array1.remove(lowest)
    sorted_array1.remove(third_highest)
    
    index_highest = sorted_array2.pop(-1)
    index_lowest = sorted_array2.pop(0)
    index_third_highest = sorted_array2.pop(-2)  # third highest is now the second from the end after popping highest
    
    # Step 4: Rearrange arrays
    new_array1 = [highest, lowest, third_highest] + sorted_array1
    new_array2 = [index_highest, index_lowest, index_third_highest] + sorted_array2
    
    return new_array1, new_array2


def append_row(matrix, new_row):  # function to append rows into matrix
    return np.vstack([matrix, new_row])


def poly_avgerage(x,y,degree):

    def fit_polynomial(x_points, y_points, degree):
        A = np.vander(x_points, degree + 1) # this creates a matrix of len(x_points) x degree+1. IT CAPTURES EVERY POINT.
        b = np.array(y_points)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]  # clever way of solving the matrix, not as computationally heavy as inverse solving
        return coeffs
 
    # for plotting and populating x with more numbers
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


    coeffs = fit_polynomial(x, y, degree) # returns coefficients
    y_values = np.polyval(coeffs, x_common)  # fits the points and coefficients to the polynomial

    # plotting

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_plot_min = x_min - x_margin
    x_plot_max = x_max + x_margin
    y_plot_min = y_min - y_margin
    y_plot_max = y_max + y_margin

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points',zorder=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('User Points')
    plt.grid(True)
    plt.xlim(x_plot_min, x_plot_max) # for limits on the plots
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.show()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points', zorder=2)
    plt.plot(x_common,   y_values, color='red', label='Average Graph', zorder=1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polynomial Fitting')
    plt.grid(True)
    plt.xlim(x_plot_min, x_plot_max)
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x_common,  y_values, color='red', label='Average Graph', zorder=1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polynomial Fitting (Full Graph)')
    plt.grid(True)
    plt.legend()
    plt.show()


def exp_average(x, y):
    # define the exp model fitting function
    def exp_model(x, A, b, C):
        return A * np.exp(b * x) + C

    # initial guess for parameters
    def initial_guess(x_points, y_points):
        A_guess = (np.max(y_points) - np.min(y_points)) / (np.exp(np.max(x_points)) - np.exp(np.min(x_points)))
        
        try:
            b_guess = np.log(y_points[-1] / y_points[0]) / (x_points[-1] - x_points[0])
            if np.isnan(b_guess):
                b_guess = 0.1  # if a negative falls into the log we default guess
        except ValueError:
            b_guess = 0.1  # if there is some value error in the log we default guess
        
        C_guess = np.min(y_points)  # C is guessed based on the smallest y value
        
        return [A_guess, b_guess, C_guess]

    # fit the model to the data
    try:
        params, _ = curve_fit(exp_model, x, y, p0=initial_guess(x, y), maxfev=2000) # returns the coeffs
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return

    A_fit, b_fit, C_fit = params
    if C_fit > 0 or C_fit == 0:
        print(f"Exp Function: y = {A_fit} + e^{b_fit}x + {C_fit}")
    else:
        print(f"Exp Function: y = {A_fit} + e^{b_fit}x - {abs(C_fit)}")

    # create our function
    x_common = np.linspace(np.min(x), np.max(x), 400)
    y_fit = exp_model(x_common, A_fit, b_fit, C_fit) 

    # plotting
    x_min = np.min(x)
    x_max = np.max(x)
    y_max = np.max(y)
    y_min = np.min(y)

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_plot_min = x_min - x_margin
    x_plot_max = x_max + x_margin
    y_plot_min = y_min - y_margin
    y_plot_max = y_max + y_margin

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points',zorder=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('User Points')
    plt.grid(True)
    plt.xlim(x_plot_min, x_plot_max) # for limits on the plots
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Data Points')
    plt.plot(x_common, y_fit, color='red', label='Fitted Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Exponential Curve Fitting')
    plt.grid(True)
    plt.xlim(x_plot_min, x_plot_max) # for limits on the plots
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x_common, y_fit, color='red', label='Fitted Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Exponential Curve Fitting (Full)')
    plt.grid(True)
    plt.legend()
    plt.show()