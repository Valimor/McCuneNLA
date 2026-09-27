import numpy as np
import time
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

def build_fd_laplacian_2d(N, h):
    n = N + 1
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T /= h**2
    I = np.eye(n)
    return np.kron(I, T) + np.kron(T, I)

def build_fd_T_interior(N, h):
    n_interior = N - 1
    main = -2.0 * np.ones(n_interior)
    off = np.ones(n_interior - 1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T /= h**2
    return T

def solve_sylvester_general(T, F):
    eigvals, Q = np.linalg.eig(T)         # general eigendecomposition -- NOT eigh
    Q_inv = np.linalg.inv(Q)               # no shortcut here, unlike the orthogonal case

    F_tilde = Q_inv @ F @ Q_inv.T
    U_tilde = F_tilde / (eigvals[:, None] + eigvals[None, :])
    U_complex = Q @ U_tilde @ Q.T
    U = U_complex.real
    return U

def solve_sylvester_fd(T, F):
    n = T.shape[0]
    eigvals, Q = np.linalg.eigh(T)   # symmetric eigendecomposition -- fast, and exact for symmetric T
    F_tilde = Q.T @ F @ Q             # change of basis for the right-hand side
    denom = eigvals[:, None] + eigvals[None, :]   # matrix of lambda_i + lambda_j, via broadcasting
    U_tilde = F_tilde / denom
    U = Q @ U_tilde @ Q.T              # change back to the original basis
    return U

Ns = np.logspace(2, 7, base=2, num=10, dtype=np.int64)

# n_basis chosen large enough that Galerkin sits at its representability
# floor across this whole N sweep -- this is a smooth, non-singular solution
# (product of sines), so unlike the Neumann/Robin eigenmode work, spectral
# convergence in n_basis should be fast and the conditioning wall shouldn't
# bite until much larger n_basis. Worth confirming with Study B before
# trusting this choice blindly.
n_basis = 24

errors_sylvester = []
errors_galerkin = []
setup_times_sylvester = []
solve_times_sylvester = []
setup_times_galerkin = []
solve_times_galerkin = []

for N in Ns:
    D, x = cv.chebyshev_diff_matrix(N)
    X, Y = np.meshgrid(x, x)

    u = np.sin(np.pi * (X + 1) / 2) * np.sin(np.pi * (Y + 1) / 2)
    f = -np.pi ** 2 / 2 * u

    # --- Sylvester (collocation) ---
    t0 = time.time()
    D2 = D @ D
    T_interior = D2[1:-1, 1:-1]
    eigvals, Q = np.linalg.eig(T_interior)
    Q_inv = np.linalg.inv(Q)
    t1 = time.time()

    f_interior = f[1:-1, 1:-1]              # drop the boundary ring -- this is your F
    u_interior_sylvester = solve_sylvester_general(T_interior, f_interior)
    u_approx_sylvester = np.zeros((N + 1, N + 1))
    u_approx_sylvester[1:-1, 1:-1] = u_interior_sylvester 
    t2 = time.time()

    errors_sylvester.append(np.max(np.abs(u - u_approx_sylvester)))
    setup_times_sylvester.append(t1 - t0)
    solve_times_sylvester.append(t2 - t1)

    # --- Galerkin ---
    t0 = time.time()
    Phi = cv.build_dirichlet_basis(x, n_basis)
    W = np.diag(cv.clenshaw_curtis_weights(N))

    Dx_2d = np.kron(np.eye(N + 1), D)
    Dy_2d = np.kron(D, np.eye(N + 1))
    L_2d = Dx_2d @ Dx_2d + Dy_2d @ Dy_2d

    Phi_2d = np.kron(Phi, Phi)
    Phi_2dL = L_2d @ Phi_2d
    W_2d = np.kron(W, W)

    M = Phi_2d.T @ W_2d @ Phi_2dL
    t1 = time.time()

    f_flat = f.ravel()
    fphi_flat = Phi_2d.T @ W_2d @ f_flat
    c = np.linalg.solve(M, fphi_flat)
    u_approx_galerkin = np.reshape(Phi_2d @ c, (N + 1, N + 1))
    t2 = time.time()

    errors_galerkin.append(np.max(np.abs(u - u_approx_galerkin)))
    setup_times_galerkin.append(t1 - t0)
    solve_times_galerkin.append(t2 - t1)

    print(f"N={N}: sylvester error={errors_sylvester[-1]:.3e}, galerkin error={errors_galerkin[-1]:.3e}")

# --- plotting ---
fig, (ax_err, ax_setup, ax_solve) = plt.subplots(1, 3, figsize=(17, 5))

ax_err.loglog(Ns, errors_sylvester, "o-", label="Sylvester (collocation)")
ax_err.loglog(Ns, errors_galerkin, "s-", label="Galerkin")
ax_err.set_xlabel("N")
ax_err.set_ylabel("Max absolute error")
ax_err.set_title("Error vs. N")
ax_err.legend()

ax_setup.loglog(Ns, setup_times_sylvester, "o-", label="Sylvester (eig decomp)")
ax_setup.loglog(Ns, setup_times_galerkin, "s-", label="Galerkin (assemble M)")
ax_setup.set_xlabel("N")
ax_setup.set_ylabel("Setup time (s)")
ax_setup.set_title("Setup Time vs. N")
ax_setup.legend()

ax_solve.loglog(Ns, solve_times_sylvester, "o-", label="Sylvester")
ax_solve.loglog(Ns, solve_times_galerkin, "s-", label="Galerkin")
ax_solve.set_xlabel("N")
ax_solve.set_ylabel("Solve time (s)")
ax_solve.set_title("Solve Time vs. N (per right-hand side)")
ax_solve.legend()

fig.suptitle(f"2D Poisson: Sylvester vs. Galerkin (n_basis={n_basis})")
fig.tight_layout()
plt.show()