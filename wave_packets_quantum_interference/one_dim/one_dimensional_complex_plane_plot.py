import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fps, num_seconds, dt = 30, 15, 0.05
total_frames = fps * num_seconds

x_min, x_max, num_points = -50, 50, 8000
x_c, ko, sigma = -45, 1, 1.0
x = np.linspace(x_min, x_max, num_points)
dx = (x_max - x_min) / (num_points - 1)
w2_max = 1 / np.power(2 * np.pi * sigma ** 2, 0.5)
w_factor = np.power(0.5 * sigma ** 2 / np.pi, 0.25)


def calculate_psi(t, xo, k, s):
    s2 = s**2
    psi = np.exp(-0.25 * np.square(x - xo - 2j * k * s2) / (s2 + 1j * t))
    psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * t)
    psi = np.power(0.5 * s2 / np.pi, 0.25) * psi
    return psi


current_time = 0
current_psi = calculate_psi(current_time, x_c, ko, sigma)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.axis("off")
line, = ax.plot3D(x, np.real(current_psi), np.imag(current_psi), 'r', linewidth=3)


def anim(frame):
    global current_psi, current_time
    current_time += dt
    current_psi = calculate_psi(current_time, x_c, ko, sigma)
    line.set_data(x, np.real(current_psi))
    line.set_3d_properties(np.imag(current_psi))
    return line,


anim = FuncAnimation(fig, anim, frames=total_frames, interval=1000 / fps, blit=False, repeat=True)
plt.show()
