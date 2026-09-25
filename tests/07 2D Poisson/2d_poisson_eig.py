import numpy as np
import scipy
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen

# i might need to start doing fft for this...

N = 128
D, x = cv.chebyshev_diff_matrix(N)

X, Y = np.meshgrid(x, x)

# make the differentiation matrices
print("Making differentiation matrices")
Dx_2d = np.kron(np.eye(N + 1), D)
Dy_2d = np.kron(D, np.eye(N + 1))
print("Making Laplacian")
L_2d = Dx_2d @ Dx_2d + Dy_2d @ Dy_2d

# now, build out the basis
# this is what it grows in!
print("Making basis")
n_basis = 32
Phi = cv.build_dirichlet_basis(x, n_basis)
W = np.diag(cv.clenshaw_curtis_weights(N))

# develop Phi, LaplacianPhi, and inner product weights
Phi_2d = np.kron(Phi, Phi)
Phi_2dL = L_2d @ Phi_2d
W_2d = np.kron(W, W)

# solving now
print("Computing K, M")
K = -Phi_2d.T @ W_2d @ Phi_2dL
M = Phi_2d.T @ W_2d @ Phi_2d
print("computing eigenmodes")
eigvals, eigvecs = eigen.generalized_eigen(K, M)

# sorting
print("computed eigenmodes")
perm = np.argsort(np.abs(eigvals))
eigvals = eigvals[perm]
eigvecs = eigvecs[:, perm]
idx = 4

# 5. Plot 3x3 grid of first 9 eigenmodes
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for k in range(16):
    ax = axes[k]
    
    # Reconstruct 2D mode from column eigenvector k
    mode_flat = Phi_2d @ eigvecs[:, k]
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

plt.suptitle('First 9 Eigenmodes of Poisson Equation on $[-1,1]^2$', fontsize=14, y=0.98)
fig.tight_layout()
plt.show()