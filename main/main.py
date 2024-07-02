import numpy as np
import matplotlib.pyplot as plt
from math import pow
import func
import sys

x = [-4.2, -3.5, -2.8, -1.6, -0.9, 0.0, 1.2, 2.3, 3.4, 4.7]  # x values from the user
y = [0.5, 1.1, 2.3, 4.7, 9.2, 15.0, 27.5, 50.3, 91.8, 168.4] # y values from the user

n = 3 # degree of polynomial

func.poly_avgerage(x,y,n)
func.exp_average(x,y)
sys.exit()

