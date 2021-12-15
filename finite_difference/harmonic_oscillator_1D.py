import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix
from scipy.special import hermite
import matplotlib.pyplot as plt


# h bar = 1, and 2m=1
omega = 1  # angular frequency
x_min, x_max, num_grid_points = -1, 1, 202
x_len, dx, x_center = x_max - x_min, (x_max - x_min) / (num_grid_points-1), 0.5 * (x_max + x_min)
hx = 1 / np.square(dx)

x = np.linspace(x_min, x_max, num_grid_points)
potential = 0.25 * np.square(omega * (x-x_center))


diagonals = [
    np.ones(num_grid_points-3), -2 * np.ones(num_grid_points-2), np.ones(num_grid_points-3)
]
offset = [-1, 0, 1]
hamiltonian = -csr_matrix(diags(diagonals, offset)) * hx + np.diag(potential[1:-1])

eigs, numerical_wave_function = linalg.eigsh(hamiltonian, k=2, which='SM')
numerical_wave_function = numerical_wave_function / np.sqrt(dx) 


plt.xlim([-1 + x_min, x_max + 1])
plt.ylim([-0.01, 1.1 * np.sqrt(2.0/x_len)])

plt.plot(x, potential/np.max(potential), '-b', linewidth=2, zorder=1)
plt.plot(x[1:-1], np.square(np.absolute(numerical_wave_function[:, 0])), '--r', linewidth=1, zorder=2)
plt.plot(x[1:-1], np.square(np.absolute(numerical_wave_function[:, 1])), '--g', zorder=3)

plt.vlines(x_min, 0, np.sqrt(2.0/x_len), colors='k', linestyles='dashed', linewidth=2, zorder=4)
plt.vlines(x_max, 0, np.sqrt(2.0/x_len), colors='k', linestyles='dashed', linewidth=2, zorder=5)
plt.show()

