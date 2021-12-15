import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix, eye, kron, dia_matrix
import matplotlib.pyplot as plt

# h bar = 1, and 2m=1
omega_x, omega_y = 1, 1
x_min, x_max, num_grid_points_x = -1, 1, 41
y_min, y_max, num_grid_points_y = -2, 2, 81

x_len, dx, x_center = x_max - x_min, (x_max - x_min) / (num_grid_points_x - 1), 0.5 * (
            x_max + x_min)
y_len, dy, y_center = y_max - y_min, (y_max - y_min) / (num_grid_points_y - 1), 0.5 * (
            y_max + y_min)
dim = (num_grid_points_x - 2) * (num_grid_points_y - 2)
hx, hy = 1 / np.square(dx), 1 / np.square(dy)

x = np.linspace(x_min, x_max, num_grid_points_x)
y = np.linspace(y_min, y_max, num_grid_points_y)
xx, yy = np.meshgrid(x, y)

# building hamiltonian
offset = [-1, 0, 1]
diagonals_x = [
    np.ones(num_grid_points_x - 3), -2 * np.ones(num_grid_points_x - 2),
    np.ones(num_grid_points_x - 3)
]
tx = diags(diagonals_x, offset)
e2 = eye(num_grid_points_y - 2)
tx = kron(e2, tx)
tx = tx * hx

offset = [-num_grid_points_x + 2, 0, num_grid_points_x - 2]
diagonals_x = [np.ones(dim - num_grid_points_x + 2), -2 * np.ones(dim),
               np.ones(dim - num_grid_points_x + 2)]
ty = diags(diagonals_x, offset)
ty = ty * hy
potential = 0.25 * (np.square(omega_x * (xx - x_center)) + np.square(omega_y * (yy - y_center)))
potential = potential[1:-1, 1:-1].flatten()
hamiltonian = -csr_matrix(tx + ty) + dia_matrix(np.diag(potential), dtype='float')

eigs, numerical_wave_function = linalg.eigs(hamiltonian, k=4, which='SR')
numerical_wave_function = numerical_wave_function / np.sqrt(dx * dy)
# vecs are normalized to num_grid_points-2 points, so we normalize it to len
numerical_wave_function = numerical_wave_function[:, 3].reshape(
    (num_grid_points_y - 2, num_grid_points_x - 2)
)

fig, ax = plt.subplots(figsize=(12, 12))
plt.axis('off')
vmax = 4 / (x_len * y_len)
ax.imshow(np.square(np.absolute(numerical_wave_function)), interpolation='bilinear',
          aspect='equal', cmap='hot', vmin=0.0, vmax=vmax, zorder=1, origin='upper')
ax.set_title('Numerical Wave Function', fontsize=14)
ax.axis('off')

plt.show()
