import math
import numpy as np
import matplotlib.pyplot as plt
from math import pow

# y = ax + b

# Define the matrices

# np.matrix([[(x_1)^2, x_1, 1], [(x_2)^2, x_2, 1], [(x_3)^2, x_3, 1]])

x = [-2,-1.5,-1,0,1,2]
y = [0,-1.6875,-77,0,3,32]


def append_row(matrix, new_row):  # function to append rows into matrix
    return np.vstack([matrix, new_row])



i = 1   # Outer while loop that appends the matrix
n = 5   # degree of polynomial

A = np.empty((0, n+1))
b = np.empty((0, 1))

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
    A = append_row(A, row)
    i += 1


# Calculate the inverse of A
A_inv = np.linalg.inv(A)

result = A_inv @ b



i = 0
solutions = []
while i != n+1:
    solutions.append(np.round(result[i, 0], decimals = 2))
    i=i+1


i=0
while i != n+1:
    if solutions[i] < 0.01 and solutions[i] >= 0:
        solutions[i] = 0
    i=i+1


# Check if values are less than 0.1, round them

print("Your function: ", solutions[0], "x^5 + ", solutions[1], "x^4 + ", solutions[2], "x^3 + ", solutions[3], "x^2 + ", solutions[4], "x + ", solutions[5])



x_n = x.copy()
y_n = y.copy()

# gross! make it into a function
i = 1
g = 1
k=0
sum = 0

while i < 50:   # 50 forward    
    x_n.append(x[5] + i)
    
    while g != n+1: # used to sum up all the values of degree n
        sum = solutions[g-1]*pow(x_n[5+i], n-k) + sum
        g = g+1
        k=k+1

    y_n.append(sum) # put sum into array
    i = i + 1
    g=1
    k=0
    sum=0

# gross! make it into a function
i = 1
j = -3  #NOTE FIX THIS
g=1
k=1
sum=0
while i < 50:   # 50 backward
  x_n = np.insert(x_n, 0, j)

  while g!=n+1:
    sum = solutions[g-1]*pow(x_n[0], n-k) + sum
    g=g+1
    k=k+1

  y_n = np.insert(y_n, 0, sum)
  i = i + 1
  j = j - 1
  g=1
  k=0
  sum=0



plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', alpha=0.6, marker='o', label='Data Points')
plt.show()
plt.plot(x_n,y_n, color='red', label='Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.ylim(-5000, 5000) # for limits on the plots
plt.xlim(-100, 100)
plt.show()

# -------------------------------------------------------
