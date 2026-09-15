import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen

N = 64
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

Phid = cv.build_dirichlet_basis(x, N-1)
Phid_dp = cv.build_dirichlet_basis_dp(x, N-1, D)

Phin = cv.build_neumann_basis(x, N-1)
Phin_dp = cv.build_neumann_basis_dp(x, N-1,D)

interior = slice(1, -1)  # drop the two boundary points

Phid_int = Phid[interior, :]
Phid_dp_int = Phid_dp[interior, :]
Phin_int = Phin[interior, :]
Phin_dp_int = Phin_dp[interior, :]

# this is solving generalized eigenvalue problem for 
# -Phi_dp_interior @ c = \lambda @ Phi_interior @ c
# since phi interior is square and invertible (ask about non-square) we can find
# -Phi_interior^{-1} @ Phi_dp_interior @ c = \lambda @ c
# and solve as an eigenvalue problem

A_deig = -np.linalg.solve(Phid_int, Phid_dp_int)
eigvals_d, eigvecs_d = eigen.QR_eigen_givens_algorithm(A_deig)
eigvals_d = np.sort(eigvals_d)
A_neig = -np.linalg.solve(Phin_int, Phin_dp_int)
eigvals_n, eigvecs_n = eigen.QR_eigen_givens_algorithm(A_neig)
eigvals_n = np.sort(eigvals_n)

max_idx = N - 15

plt.title("Eigenvalue distributions for dirichlet and neumann")
plt.scatter(range(max_idx), eigvals_d[:max_idx], label="Eigvalues for dirichlet")
plt.scatter(range(max_idx), eigvals_n[:max_idx], label="Eigvalues for neumann")
plt.plot(range(max_idx), (np.arange(max_idx) * np.pi/2) ** 2, label="Analytic eigenvalue", linestyle="--", color="green")
plt.xlabel("n")
plt.ylabel("n-th Eigenvalue")
plt.legend()
plt.show()