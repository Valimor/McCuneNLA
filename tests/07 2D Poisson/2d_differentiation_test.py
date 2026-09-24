import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

N = 64
D, x = cv.chebyshev_diff_matrix(N)

X, Y = np.meshgrid(x, x)

# gotta figure out why. probably something to do with my meshgrid order
Dy_2d = np.kron(D, np.eye(N + 1))
Dx_2d = np.kron(np.eye(N + 1), D)

# use numpy reshape/ravel to work on this
f = np.sin(np.pi * X) * np.cos(np.pi * Y)
f_dya = -np.pi * np.sin(np.pi * X) * np.sin(np.pi * Y)
f_dy = np.reshape(Dy_2d @ f.ravel(), (N + 1, N+1))
f_dxa = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)
f_dx = np.reshape(Dx_2d @ f.ravel(), (N + 1, N+1))

print(f"Difference in x: {np.linalg.norm(f_dx - f_dxa)}")
print(f"Difference in y: {np.linalg.norm(f_dy - f_dya)}")

# 3D axes
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(X, Y, f_dx, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()