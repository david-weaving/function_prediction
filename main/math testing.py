import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import exit
import func


x = [1,2,3,4,5,6]
y= [0.5,1,1.5,2,2.5,3]

def fit_linear(x_points, y_points):
    A = np.vander(x_points, 2) # this creates a matrix of len(x_points) x degree+1. IT CAPTURES EVERY POINT.
    b = np.array(y_points)
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]  # clever way of solving the matrix, not as computationally heavy as inverse solving
    return coeffs

A,B = fit_linear(x,y)

print(f"Your Linear Function: {A}x + {B}")