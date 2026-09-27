import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

# --- Study B: Galerkin error vs. n_basis, at a fixed, generously large N ---
# Goal: isolate the representability-vs-conditioning tradeoff for the 2D
# Galerkin solve, independent of grid resolution -- the 2D analog of the
# n_basis sweep done earlier for the Neumann/Robin 1D eigenmode work.

N_fixed = 128   # large enough that grid resolution is never the bottleneck;
                 # worth confirming this via a quick separate check (e.g. rerun
                 # with N_fixed=192 and see if results change) before trusting it

n_basis_values = np.arange(4, 48, 2)

D, x = cv.chebyshev_diff_matrix(N_fixed)
X, Y = np.meshgrid(x, x)

u = np.sin(np.pi * (X + 1) / 2) * np.sin(np.pi * (Y + 1) / 2)
f = -np.pi ** 2 / 2 * u
f_flat = f.ravel()

W = np.diag(cv.clenshaw_curtis_weights(N_fixed))
W_2d = np.kron(W, W)

Dx_2d = np.kron(np.eye(N_fixed + 1), D)
Dy_2d = np.kron(D, np.eye(N_fixed + 1))
L_2d = Dx_2d @ Dx_2d + Dy_2d @ Dy_2d

errors = []
cond_numbers = []

for n_basis in n_basis_values:
    Phi = cv.build_dirichlet_basis(x, n_basis)

    Phi_2d = np.kron(Phi, Phi)
    Phi_2dL = L_2d @ Phi_2d

    M = Phi_2d.T @ W_2d @ Phi_2dL
    fphi_flat = Phi_2d.T @ W_2d @ f_flat

    cond_M = np.linalg.cond(M)
    c = np.linalg.solve(M, fphi_flat)
    u_approx = np.reshape(Phi_2d @ c, (N_fixed + 1, N_fixed + 1))

    err = np.max(np.abs(u - u_approx))
    errors.append(err)
    cond_numbers.append(cond_M)

    print(f"n_basis={n_basis}: error={err:.3e}, cond(M)={cond_M:.3e}")

# --- plotting: error and conditioning side by side, sharing the n_basis axis ---
fig, (ax_err, ax_cond) = plt.subplots(1, 2, figsize=(12, 5))

ax_err.semilogy(n_basis_values, errors, "o-")
ax_err.set_xlabel("n_basis")
ax_err.set_ylabel("Max absolute error")
ax_err.set_title(f"Galerkin Error vs. Basis Size (N={N_fixed} fixed)")

ax_cond.semilogy(n_basis_values, cond_numbers, "o-", color="darkorange")
ax_cond.set_xlabel("n_basis")
ax_cond.set_ylabel("cond(M)")
ax_cond.set_title(f"Conditioning vs. Basis Size (N={N_fixed} fixed)")

fig.suptitle("2D Galerkin: representability vs. conditioning tradeoff")
fig.tight_layout()
plt.show()