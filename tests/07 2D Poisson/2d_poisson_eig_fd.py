import numpy as np
import scipy
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen

def build_fd_laplacian_2d(N, h):
    n = N + 1
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T /= h**2

    I = np.eye(n)
    L_fd = np.kron(I, T) + np.kron(T, I)
    return L_fd

N = 48
x_fd = np.linspace(-1,1,N+1)
h = x_fd[1]-x_fd[0]
X, Y = np.meshgrid(x_fd, x_fd)

L_2d_fd = build_fd_laplacian_2d(N, h)

eigvals, eigvecs = eigen.get_n_eigenmodes_arnoldi(-L_2d_fd, m=256)

# sorting
print("computed eigenmodes")
perm = np.argsort(np.abs(eigvals))
eigvals = eigvals[perm]
eigvecs = eigvecs[:, perm]

# 5. Plot 3x3 grid of first 9 eigenmodes
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for k in range(16):
    ax = axes[k]
    
    # Reconstruct 2D mode from column eigenvector k
    mode_flat = eigvecs[:, k]
    mode_mesh = np.reshape(mode_flat, (N + 1, N + 1))
    
    # Normalize peak value to +1 for consistent colormap scaling
    peak = mode_mesh[np.unravel_index(np.argmax(np.abs(mode_mesh)), mode_mesh.shape)]
    mode_mesh /= peak
    
    # Plot 2D filled contour
    cf = ax.contourf(X, Y, mode_mesh, levels=30, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.contour(X, Y, mode_mesh, levels=[0], colors='black', linewidths=0.8, linestyles='--')  # Nodal lines
    
    # Label mode and numerical eigenvalue
    ax.set_title(f'Mode {k+1}\n$\lambda_{{{k+1}}} = {eigvals[k]:.2f}$', fontsize=11)
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle('First 16 Eigenmodes of Poisson Equation on $[-1,1]^2$', fontsize=14, y=0.98)
fig.tight_layout()
plt.show()