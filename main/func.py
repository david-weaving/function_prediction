import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
import tensorflow as tf

def sort_array(x,y):
    # Combine the arrays using zip
    combined = list(zip(x, y))

    # Sort the combined list by the first element (elements of x)
    sorted_combined = sorted(combined, key=lambda pair: pair[0])

    # Unzip the combined list back into two separate lists
    x_sorted, y_sorted = zip(*sorted_combined)

    # Convert the tuples back to lists
    x_sorted = list(x_sorted)
    y_sorted = list(y_sorted)

    return x_sorted, y_sorted

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
    
    def print_poly(coeffs): # used for printing polynomial
        p = np.size(coeffs) - 1
        print("Your polynomial function: ", end="")
        for i in coeffs:
            if p == np.size(coeffs)-1:
                print(f"{i}x^{p}", end="")
            elif i > 0 and p > 0 and p != 1:
                print(f" + {i}x^{p}", end="")
            elif i < 0 and p > 0 and p != 1:
                print(f" - {abs(i)}x^{p}", end="")
            elif i > 0 and p == 1:
                print(f" + {i}x", end="")
            elif i < 0 and p == 1:
                print(f" - {abs(i)}x", end="")
            elif i > 0 and p == 0:
                print(f" + {i}", end="")
            elif i < 0 and p == 0:
                print(f" - {abs(i)}", end="")
            p=p-1
    
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
    print_poly(coeffs)

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
        except (ValueError, IndexError):
            b_guess = 0.1  # if there is some value error in the log we default guess
            
        
        C_guess = np.min(y_points)  # C is guessed based on the smallest y value
        
        return [A_guess, b_guess, C_guess]

    def print_exp(params):
        if params[2] > 0 or params[2] == 0:
            print(f"Exp Function: y = {params[0]}e^{params[1]}x + {params[2]}")
        else:
            print(f"Exp Function: y = {params[0]}e^{params[1]}x - {abs(params[2])}")

    # fit the model to the data
    try:
        params, _ = curve_fit(exp_model, x, y, p0=initial_guess(x, y), maxfev=30000000) # returns the coeffs
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return

    print_exp(params)

    A_fit, b_fit, C_fit = params
    
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
        params, _ = curve_fit(sineval, x_points, y_points, p0=initial_guess,maxfev=50000) # once again curve_fit helps us fit the curve given an f(x) input, in this case; f(x)=Asin(Bx+D)+C
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

    x_common = np.linspace(x_min, x_max, 900)

    # continue the graph
    x_forward = np.linspace(x_max+0.1, 150, 900)
    x_backward = np.linspace(-150, x_min-0.1, 900)
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

def sqrt_average(x,y):  # most likely not using this, the points are too sensitive
    
    # sqrt function
    def sqrt_func(x, A, C, D):
        return A * np.sqrt(x + C) + D

    # inital guess for the form y = A*sqrt(x + C) + D
    def fit_square_root(x,y):
        A_guess = (y[-1] * np.sqrt(y[-2] * y[-2] - y[-3] * y[-3])) / (np.sqrt(x[-1] * (y[-2] * y[-2] - y[-3] * y[-3]) + y[-3] * y[-3] * x[-2] - x[-3] * y[-2] * y[-2]))
        C_guess = (y[-3] * y[-3] * x[-2] - x[-3] * y[-2] * y[-2]) / (y[-2] * y[-2] - y[-3] * y[-3])
        D_guess = y[-1] / (A_guess * np.sqrt(x[-1] + C_guess))
        initial_guess = [A_guess, C_guess, D_guess]

        # Fit the model to all points
        params, _ = curve_fit(sqrt_func, x, y, p0=initial_guess, maxfev=50000)
        return params

    # return A,C,D
    A_opt, C_opt, D_opt = fit_square_root(x,y)

    print(f"Square Root Function: {A_opt} * sqrt(x + {C_opt}) + {D_opt}")

    x_min = np.min(x)
    x_max = np.max(x)
    y_max = np.max(y)
    y_min = np.min(y)

    x_common = np.linspace(x_min, x_max, 400)

    # continue the graph
    x_forward = np.linspace(x_max+0.1, 50, 400)
    x_backward = np.linspace(C_opt*-1, x_min-0.1, 400) # C_opt * -1 is the end bound (where we dont go below zero in our square root)
    x_common = np.append(x_common, x_forward)
    x_common = np.insert(x_common, 0, x_backward)

    y_values = sqrt_func(x_common,A_opt,C_opt,D_opt)

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
    plt.title('Square Root Fitting')
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

def ln_average(x,y):
    def print_ln():
        if C < 0 and D < 0:
            print(f"{A}ln({B}x - {abs(C)}) - {abs(D)}")
        elif C > 0 and D < 0:
            print(f"{A}ln({B}x + {C}) - {abs(D)}")
        elif C < 0 and D > 0:
            print(f"{A}ln({B}x - {abs(C)}) + {D}")
        else:
            print(f"{A}ln({B}x + {C}) + {D}")

    def ln_func(x, A, B, C, D):
        return A * np.log(B * x + C) + D

    def fit_ln(x,y):
        # guesses for A,B,C,D given y = Aln(Bx + C) + D
        A_guess = (np.max(y) - np.min(y)) / np.log(np.max(x) + 1)
        B_guess = 1 / (np.max(x) - np.min(x))
        C_guess = abs(np.min(x)) + 1  # shift to ensure the logarithm argument is positive
        D_guess = np.mean(y) - A_guess * np.log(B_guess * np.mean(x) + C_guess)
        initial_guess = [A_guess, B_guess, C_guess, D_guess]

        # fit the curve for ln
        params, _ = curve_fit(ln_func, x, y, p0=initial_guess, maxfev=30000)
        
        return params


    A, B, C, D = fit_ln(x,y)

    print_ln()



    x_min = np.min(x)
    x_max = np.max(x)
    y_max = np.max(y)
    y_min = np.min(y)

    x_common = np.linspace(x_min, x_max, 400)

    # continue the graph
    x_forward = np.linspace(x_max+0.1, 50, 400)
    x_backward = np.linspace((0.1-C)/B, x_min-0.1, 400) # C * -1 is the end bound (where we dont go below zero in our square root)
    x_common = np.append(x_common, x_forward)
    x_common = np.insert(x_common, 0, x_backward)

    y_values = ln_func(x_common,A,B,C,D)

    # plotting
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

def predict_function(x,y): # predicts funtion
    
    x,y = sort_array(x,y)
    print(x)
    print(y)
    model = tf.keras.models.load_model("C:/Users/Administrator/func pred/function_prediction/models/model_V1_8.h5")

    points = list(zip(x, y))
    print(points)

    predicted_type = predict_function_type(points, model)
    return predicted_type

def predict_degree(x,y):
    x,y = sort_array(x,y)

    model_degree = tf.keras.models.load_model("C:/Users/Administrator/func pred/function_prediction/models/model_degree_V1.h5")
    points = list(zip(x,y))
    predicted_degree = predict_degree_type(points, model_degree)
    return predicted_degree


def predict_degree_type(points, model):
    points_reshaped = np.array([points])
    prediction = model.predict(points_reshaped)
    predicted_degree = np.argmax(prediction)

    if predicted_degree == 0:
        return 1
    elif predicted_degree == 1:
        return 2
    elif predicted_degree == 2:
        return 3
    elif predicted_degree == 3:
        return 4
    elif predicted_degree == 4:
        return 5

def predict_function_type(points, model): # returns function type

    points_reshaped = np.array([points])  # Reshape to fit model input shape
    prediction = model.predict(points_reshaped)
    predicted_class = np.argmax(prediction)  # Get index of highest probability
    if predicted_class == 0:
        return "ln"
    elif predicted_class == 1:
        return "polynomial"
    elif predicted_class == 2:
        return "exponential"
    elif predicted_class == 3:
        return "sine"
    
