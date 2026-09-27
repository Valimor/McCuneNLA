import numpy as np
import matplotlib.pyplot as plt

def build_fd_laplacian_2d(N, h):
    n = N + 1
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T /= h**2

    I = np.eye(n)
    L_fd = np.kron(I, T) + np.kron(T, I)
    return L_fd

def build_fd_T_interior(N, h):
    n_interior = N - 1
    main = -2.0 * np.ones(n_interior)
    off = np.ones(n_interior - 1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T /= h**2
    return T

def solve_sylvester_fd(T, F):
    n = T.shape[0]
    eigvals, Q = np.linalg.eigh(T)   # symmetric eigendecomposition -- fast, and exact for symmetric T

    F_tilde = Q.T @ F @ Q             # change of basis for the right-hand side

    denom = eigvals[:, None] + eigvals[None, :]   # matrix of lambda_i + lambda_j, via broadcasting
    U_tilde = F_tilde / denom

    U = Q @ U_tilde @ Q.T              # change back to the original basis
    return U

sylvester = True

N = 64
x_fd = np.linspace(-1,1,N+1)
h = x_fd[1]-x_fd[0]
X, Y = np.meshgrid(x_fd, x_fd)
u = np.sin(np.pi * (X + 1) / 2) * np.sin(np.pi * (Y + 1)/2)
f = -np.pi ** 2 / 2 * u
f_flat = f.ravel()

# solving
if sylvester:
    T_int = build_fd_T_interior(N, h)
    f_interior = f[1:-1, 1:-1]              # drop the boundary ring -- this is your F

    U_interior = solve_sylvester_fd(T_int, f_interior)

    u_approx = np.zeros((N + 1, N + 1))     # boundary stays exactly 0 (homogeneous Dirichlet)
    u_approx[1:-1, 1:-1] = U_interior
else:
    L_2d_fd = build_fd_laplacian_2d(N, h)
    u_approx = np.reshape(np.linalg.solve(L_2d_fd, f_flat), (N + 1, N + 1))

# using infinity norm to avoid scaling
print(f"Error: {np.max(np.abs(u - u_approx))}")

fig = plt.figure(figsize=(16, 5))

# 1. Exact Solution
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
ax1.set_title('Analytic Solution $u$')

# 2. Numerical Solution
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_approx, cmap='viridis', edgecolor='none')
ax2.set_title('Finite Difference Approximation $u_{approx}$')

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