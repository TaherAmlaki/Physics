import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix
import matplotlib.pyplot as plt


# h bar = 1, and 2m=1
x_min, x_max, num_grid_points = 2, 5, 201
x_len, dx, x_center = x_max - x_min, (x_max - x_min) / (num_grid_points-1), 0.5 * (x_max + x_min)
hx = 1 / np.square(dx)

x = np.linspace(x_min, x_max, num_grid_points)

analytical_wave_function = np.sqrt(2.0/x_len) * np.sin(np.pi*(x-x_center+0.5*x_len)/x_len)

diagonals = [
    np.ones(num_grid_points-3), -2 * np.ones(num_grid_points-2), np.ones(num_grid_points-3)
]

offset = [-1, 0, 1]
hamiltonian = -csr_matrix(diags(diagonals, offset)) * hx

eigs, numerical_wave_function = linalg.eigs(hamiltonian, k=1, which='SR')
numerical_wave_function = numerical_wave_function / np.sqrt(dx)  # vecs are normalized to num_grid_points-2 points, so we normalize it to len
numerical_wave_function = np.concatenate(([[0]], numerical_wave_function, [[0]]))

plt.xlim([-1 + x_min, x_max + 1])
plt.ylim([-0.01, 1.1 * np.sqrt(2.0/x_len)])

plt.plot(x, np.square(np.absolute(numerical_wave_function)), 'ob', linewidth=1, zorder=2)
plt.plot(x, np.square(np.absolute(analytical_wave_function)), '--r', zorder=1)

plt.vlines(x_min, 0, np.sqrt(2.0/x_len), colors='k', linestyles='dashed', linewidth=3, zorder=3)
plt.vlines(x_max, 0, np.sqrt(2.0/x_len), colors='k', linestyles='dashed', linewidth=3, zorder=4)
plt.show()

