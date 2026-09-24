import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

N = 32
D, x = cv.chebyshev_diff_matrix(N)

X, Y = np.meshgrid(x, x)

# make the differentiation matrices
Dx_2d = np.kron(np.eye(N + 1), D)
Dy_2d = np.kron(D, np.eye(N + 1))
L_2d = Dx_2d @ Dx_2d + Dy_2d @ Dy_2d

# now, build out the basis
n_basis = N
Phi = cv.build_dirichlet_basis(x, n_basis)
W = np.diag(cv.clenshaw_curtis_weights(N))

# develop Phi, LaplacianPhi, and inner product weights
Phi_2d = np.kron(Phi, Phi)
Phi_2dL = L_2d @ Phi_2d
W_2d = np.kron(W, W)

# solving now
u = np.sin(np.pi * (X + 1) / 2) * np.sin(np.pi * (Y + 1)/2)
f = -np.pi ** 2 / 2 * u
M = Phi_2d.T @ W_2d @ Phi_2dL
f_flat = f.ravel()
fphi_flat = Phi_2d.T @ W_2d @ f_flat # no way this was the only fix! that's so cool
c = np.linalg.solve(M, fphi_flat)
u_approx_flat = Phi_2d @ c
u_approx = np.reshape(u_approx_flat, (N + 1, N + 1))

# using infinity norm to avoid scaling
print(f"Error: {np.max(u - u_approx)}")

fig = plt.figure(figsize=(16, 5))

# 1. Exact Solution
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
ax1.set_title('Analytic Solution $u$')

# 2. Numerical Solution
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_approx, cmap='viridis', edgecolor='none')
ax2.set_title('Spectral Approximation $u_{approx}$')

# 3. Pointwise Absolute Error
ax3 = fig.add_subplot(133, projection='3d')
err = np.abs(u_approx - u)
err_surf = ax3.plot_surface(X, Y, err, cmap='inferno', edgecolor='none')
fig.colorbar(err_surf, ax=ax3, shrink=0.5, pad=0.1)
ax3.set_title('Absolute Error $|u - u_{approx}|$')

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
error = np.abs(u_approx - u)

# Logarithmic color scale to view machine precision errors
contour = plt.contourf(X, Y, np.log10(error + 1e-16), levels=20, cmap='magma')
plt.colorbar(contour, label=r'$\log_{10}(\text{Absolute Error})$')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Log10 Absolute Error Distribution')
plt.axis('equal')
plt.show()