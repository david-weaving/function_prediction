import math
import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [3,18,2,25,35,41]  # x values from the user
y = [1,5,8,15,25,40] # y values from the user



x_graph_limit_low = -10
x_graph_limit_high = 1.2
y_graph_limit_low = -5.5
y_graph_limit_high = 1.6


# -----------------------------------------------------------------







n = 3 # degree of polynomial


func.average_graph(x,y,x_graph_limit_low,x_graph_limit_high,y_graph_limit_low,y_graph_limit_high,n)
sys.exit()

x,y = func.rearrange_arrays(x,y)  # reorders the array, used to grab the very first and very last x value for plotting, does not matter when finding average graphs

# below is legacy code, for now it is unused but making a place for ----------------------------------------------------

A = np.empty((0, n+1))
b = np.empty((0, 1))


i = 1   # Outer while loop that appends the matrix
while i != n+2:  # this while loop takes any degree polynomial and builds an n+1 x n+1 matrix (to solve system of equations, matrix of x's), along with returning the b vector (column of y)
    b = np.append(b, [[y[i-1]]], axis=0)
    row = []
    k = 0   # Power indexing
    g = 1   # Very inner while loop that populates rows
    while g != n+2:
        row.append(pow(x[i-1], n-k)) # starts with row highest power of polynomial, ex: x_2^2  x_2^1 x_2^0
        k += 1
        g += 1

    row = np.array(row).reshape(1, -1)
    A = func.append_row(A, row)
    i += 1


# Calculate the inverse of A
A_inv = np.linalg.inv(A)

result = A_inv @ b



i=0
solutions = []
while i != n+1: # grabbing solutions ax^n + bx^n-1 + .....
    #solutions.append(np.round(result[i, 0], decimals = 2))
    solutions.append(result[i, 0])
    i=i+1

# need to grab the highest and lowest x and y points 
x_max = np.max(x)
x_min = np.min(x)
y_max = np.max(y)
y_min = np.min(y)

x_n = np.linspace(x_min, x_max, 400) # create 400 points between x_1 and x_f
x_forward = np.linspace(x_max+0.1, 50, 400)
x_backward = np.linspace(-50, x_min-0.1, 400)
x_n = np.append(x_n, x_forward)
x_n = np.insert(x_n, 0, x_backward)

# Compute the polynomial values for each x
y_n = np.polyval(solutions, x_n)  #Plots any degree polynomial using A,B,C,D,E ... solutions



# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, color='blue', alpha=0.6, marker='o', label='Your Points',zorder=2)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('User Points')
# plt.grid(True)
# plt.xlim(abs(x_min)*x_graph_limit_low, abs(x_max)*x_graph_limit_high) # for limits on the plots
# plt.ylim(abs(y_min)*y_graph_limit_low, abs(y_max)*y_graph_limit_high)
# plt.legend()
# plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x,y, color='blue', alpha=0.6, marker='o',label="Your Points",zorder=2)
plt.plot(x_n,y_n, color='red', label='Graph',zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph and Points (Not Averaged)')
plt.grid(True)
plt.xlim(abs(x_min)*x_graph_limit_low, abs(x_max)*x_graph_limit_high) # for limits on the plots
plt.ylim(abs(y_min)*y_graph_limit_low, abs(y_max)*y_graph_limit_high)
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(x_n,y_n, color='red', label='New Graph',zorder=1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Full Graph')
plt.grid(True)
plt.legend()
plt.show()
# -------------------------------------------------------
