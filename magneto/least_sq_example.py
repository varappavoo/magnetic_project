#!/usr/bin/python3

import numpy as np

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
# Solve the system of equations 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8:

a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)
print(x)

x = np.linalg.lstsq(a, b, rcond=-1)
print(x[0])