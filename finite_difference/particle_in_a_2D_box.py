import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix, eye, kron
import matplotlib.pyplot as plt

# h bar = 1, and 2m=1
x_min, x_max, num_grid_points_x = -1, 1, 41
y_min, y_max, num_grid_points_y = -2, 2, 81

x_len, dx, x_center = x_max - x_min, (x_max - x_min) / (num_grid_points_x - 1), 0.5*(x_max + x_min)
y_len, dy, y_center = y_max - y_min, (y_max - y_min) / (num_grid_points_y - 1), 0.5*(y_max + y_min)
dim = (num_grid_points_x - 2) * (num_grid_points_y - 2)
hx, hy = 1 / np.square(dx), 1 / np.square(dy)

x = np.linspace(x_min, x_max, num_grid_points_x)
y = np.linspace(y_min, y_max, num_grid_points_y)
xx, yy = np.meshgrid(x, y)

analytical_x = np.sqrt(2.0 / x_len) * np.sin(np.pi * (xx - x_center + 0.5 * x_len) / x_len)
analytical_y = np.sqrt(2.0 / y_len) * np.sin(np.pi * (yy - y_center + 0.5 * y_len) / y_len)
analytical_wave_function = analytical_x * analytical_y

# building hamiltonian
offset = [-1, 0, 1]
diagonals_x = [
    np.ones(num_grid_points_x-3), -2*np.ones(num_grid_points_x-2), np.ones(num_grid_points_x-3)
]
tx = diags(diagonals_x, offset)
e2 = eye(num_grid_points_y-2)
tx = kron(e2, tx)
tx = tx * hx

offset = [-num_grid_points_x+2, 0, num_grid_points_x-2]
diagonals_x = [np.ones(dim-num_grid_points_x+2), -2*np.ones(dim), np.ones(dim-num_grid_points_x+2)]
ty = diags(diagonals_x, offset)
ty = ty * hy
hamiltonian = -csr_matrix(tx + ty)

eigs, numerical_wave_function = linalg.eigs(hamiltonian, k=1, which='SR')
numerical_wave_function = numerical_wave_function / np.sqrt(dx * dy)
# vecs are normalized to num_grid_points-2 points, so we normalize it to len
numerical_wave_function = numerical_wave_function.reshape(
    (num_grid_points_y - 2, num_grid_points_x - 2)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
plt.axis('off')
vmax = 4/(x_len*y_len)
axes[0].imshow(np.square(np.absolute(numerical_wave_function)), interpolation='bilinear',
               aspect='equal', cmap='hot', vmin=0.0, vmax=vmax, zorder=1, origin='upper')
axes[0].set_title('Numerical Wave Function', fontsize=14)
axes[0].axis('off')

axes[1].imshow(np.square(np.absolute(analytical_wave_function)), interpolation='bilinear',
               aspect='equal', cmap='hot', vmin=0.0, vmax=vmax, zorder=1, origin='upper')
axes[1].set_title('Analytical Wave Function', fontsize=14)

plt.show()
