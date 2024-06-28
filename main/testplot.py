import math
import numpy as np
import matplotlib.pyplot as plt
from math import pow

# y = ax + b

# Define the matrices

# np.matrix([[(x_1)^2, x_1, 1], [(x_2)^2, x_2, 1], [(x_3)^2, x_3, 1]])

x = [2,5,8,10,11,14]
y = [25,70,133,185,214,313]


A = np.matrix([[pow(x[0],2), x[0], 1], [pow(x[1],2), x[1], 1], [pow(x[2],2), x[2], 1]]) 
b = np.matrix([[y[0]], [y[1]], [y[2]]])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

result = A_inv @ b

# Extract the coefficients

a = np.round(result[0, 0], decimals = 5)
b = np.round(result[1, 0], decimals = 5)
c = np.round(result[2, 0], decimals = 5)

if a < 0.01:
    a = 0
if b < 0.01:
    b = 0
if c < 0.01:
    c = 0

# Check if values are less than 0.1, round them

print("Your function: ", a, "x^2 + ", b, "x + ", c)



x_n = x.copy()
y_n = y.copy()

i = 1
while i < 50:   # 50 forward
    x_n.append(x[5] + i)
    y_new = (a*x_n[5 + i]*x_n[5 + i]) + b * x_n[5+i] + c
    y_n.append(y_new)
    i = i + 1

i = 1
j = -1


while i < 50:   # 50 backward
  x_n = np.insert(x_n, 0, j)
  y_new = (a*x_n[0]*x_n[0]) + b*x_n[0] + c
  y_n = np.insert(y_n, 0, y_new)
  i = i + 1
  j = j - 1

plt.plot(x, y)
plt.show()
plt.plot(x_n,y_n)
plt.show()
# -------------------------------------------------------