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
def average_graph(x,y,x_graph_limit_low,x_graph_limit_high,y_graph_limit_low,y_graph_limit_high,degree):

    def fit_polynomial(x_points, y_points, degree):
        A = np.vander(x_points, degree + 1) # this creates an a degree+1 x degree+1 matrix with powers of degree+1 decreasing through the row
        b = np.array(y_points)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]  # clever way of solving the matrix, not as computationally heavy as inverse solving
        return coeffs
    
    # Generate all combinations of degree+1 points out of 6  -- (degree+1 because there are that many number of points)
    combinations = list(itertools.combinations(range(len(x)), degree+1))

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


    all_y_values = []
    for combo in combinations:
        x_points = [x[i] for i in combo]
        y_points = [y[i] for i in combo]
        coeffs = fit_polynomial(x_points, y_points, degree)
        y_values = np.polyval(coeffs, x_common)  # every possible polynomial
        all_y_values.append(y_values) 

    # average of all those polynomials
    y_average = np.mean(all_y_values, axis=0)

    #avg_coeffs = np.mean(np.array([fit_polynomial([x[i] for i in combo], [y[i] for i in combo], 2) for combo in combinations]), axis=0)
    #print("Your function (average for x^2): ", np.round(avg_coeffs[0], decimals=5), "x^2 + ", np.round(avg_coeffs[1], decimals=5), "x + ", np.round(avg_coeffs[2], decimals=5))
    

    # plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points',zorder=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('User Points')
    plt.grid(True)
    plt.xlim(abs(x_min)*x_graph_limit_low, abs(x_max)*x_graph_limit_high) # for limits on the plots
    plt.ylim(abs(y_min)*y_graph_limit_low, abs(y_max)*y_graph_limit_high)
    plt.legend()
    plt.show()


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


def exp_average(x,y,x_graph_limit_low,x_graph_limit_high,y_graph_limit_low,y_graph_limit_high):
    print('WIP')