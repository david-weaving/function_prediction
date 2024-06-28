import math
import numpy as np
import matplotlib.pyplot as plt
from math import pow

# y = ax + b

# Define the matrices

# np.matrix([[(x_1)^2, x_1, 1], [(x_2)^2, x_2, 1], [(x_3)^2, x_3, 1]])

x = [-1.525,-1.037,0,0.508,0.844,1]
y = [0,13.038,7,10.409,20.098,29]


def append_row(matrix, new_row):  # function to append rows into matrix
    return np.vstack([matrix, new_row])



i = 1   # Outer while loop that appends the matrix
n = 5  # degree of polynomial

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

print(A)
print(b)
# Calculate the inverse of A
A_inv = np.linalg.inv(A)

result = A_inv @ b

# Extract the coefficients
#NOTE I can take a,b,c,d,...... and put them in  their own arrays and then use a loop to do math ect
a = np.round(result[0, 0], decimals = 2)
b = np.round(result[1, 0], decimals = 2)
c = np.round(result[2, 0], decimals = 2)
d = np.round(result[3,0],  decimals = 2)
e = np.round(result[4,0],  decimals = 2)
f = np.round(result[5,0],  decimals = 2)

if a < 0.01 and a >= 0:
    a = 0
if b < 0.01 and b >= 0:
    b = 0
if c < 0.01 and c >= 0:
    c = 0
if d < 0.01 and d >= 0:
    d = 0
if e < 0.01 and e >= 0:
    e = 0
if f < 0.01 and f >= 0:
    f = 0

# Check if values are less than 0.1, round them

print("Your function: ", a, "x^5 + ", b, "x^4 + ", c, "x^3 + ", d, "x^2 + ", e, "x + ", f)



x_n = x.copy()
y_n = y.copy()

i = 1
while i < 50:   # 50 forward    ANOTHER NOTE: MAKE THIS AUTOMATIC TO SYNC UP WITH N (polynomial)
    x_n.append(x[5] + i)
    y_new = (a*x_n[5 + i]*x_n[5 + i]*x_n[5 + i]*x_n[5 + i]*x_n[5 + i]) + (b * x_n[5+i]*x_n[5+i]*x_n[5 + i]*x_n[5 + i]) + (c*x_n[5+i]*x_n[5+i]*x_n[5 + i]) + (d*x_n[5 + i]*x_n[5 + i]) + (e*x_n[5 + i]) + f
    y_n.append(y_new)
    i = i + 1

i = 1
j = -1


while i < 50:   # 50 backward
  x_n = np.insert(x_n, 0, j)
  y_new = (a*x_n[0]*x_n[0]*x_n[0]*x_n[0]*x_n[0]) + (b * x_n[0]*x_n[0]*x_n[0]*x_n[0]) + (c*x_n[0]*x_n[0]*x_n[0]) + (d*x_n[0]*x_n[0]) + (e*x_n[0]) + f
  y_n = np.insert(y_n, 0, y_new)
  i = i + 1
  j = j - 1

plt.plot(x, y)
plt.show()
plt.plot(x_n,y_n)
plt.show()

print(x_n)
print(y_n)
# -------------------------------------------------------