#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

def periodic_corr_np(x, y):
  #
  # src: https://stackoverflow.com/questions/28284257/circular-cross-correlation-python
  # circular cross correlation python
  #
  #
    """Periodic correlation, implemented using np.correlate.

    x and y must be real sequences with the same length.
    """
    return np.correlate(x, np.hstack((y[1:], y)), mode='valid')



x = np.linspace(0, np.pi, 10)
x_corr = np.linspace(0, 2*np.pi, 19)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(x_corr)
# ax.set_yticks(np.arange(0, 2*np.pi, 1))

# >>> x
# array([-3.14159265, -2.44346095, -1.74532925, -1.04719755, -0.34906585,        0.34906585,  1.04719755,  1.74532925,  2.44346095,  3.14159265])
# >>> plt.plot(x, np.sin(x))
# [<matplotlib.lines.Line2D object at 0x7f58a2659160>]
# >>> plt.show()
# sin_x = np.sin(x)
shift_by = -4
sin_x = np.array([-1.22464680e-16, -6.42787610e-01, -9.84807753e-01, -8.66025404e-01, -3.42020143e-01,  3.42020143e-01,  8.66025404e-01,  9.84807753e-01, 6.42787610e-01,  1.22464680e-16])
sin_x_shift = np.roll(sin_x, shift_by)
print(np.round(sin_x,2))
print(np.round(sin_x_shift,2))
sin_x_shift = sin_x_shift+1
# sin_x_shift = np.array([   9.84807753e-01, 6.42787610e-01,  1.22464680e-16, -1.22464680e-16, -6.42787610e-01, -9.84807753e-01, -8.66025404e-01, -3.42020143e-01,  3.42020143e-01, 8.66025404e-01])
# sin_x_shift = np.array([9.84807753e-01,    6.42787610e-01,  1.22464680e-16, -1.22464680e-16, -6.42787610e-01, -9.84807753e-01, -8.66025404e-01, -3.42020143e-01,  3.42020143e-01,  8.66025404e-01])
corr_cir = periodic_corr_np(sin_x, sin_x_shift)
corr_cir_2 = periodic_corr_np(sin_x_shift, sin_x)
corr_pad = correlate(sin_x, sin_x_shift)

print("samples: ", len(sin_x))
print("shift: " ,shift_by, ", shift detected:", corr_cir_2.argmax())
corr_cir_2 = periodic_corr_np(sin_x, sin_x_shift)
print("shift: " ,shift_by, ", shift detected:", corr_cir_2.argmax())

plt.plot(x, sin_x, label='x')
plt.plot(x, sin_x_shift, label='shift')
plt.plot(x, corr_cir, label='corr_periodic sin_x sin_x_shift')
plt.plot(x, corr_cir_2, label='corr_periodic sin_x_shift sin_x')
plt.plot(x_corr, corr_pad, label='corr_padding')
plt.legend()
plt.grid()
plt.show()
