import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen

N = 128
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

n_basis = 16
Phi = cv.build_neumann_basis(x, n_basis)
Phi_dp = cv.build_neumann_basis_dp(x, n_basis, D)

# drop the two boundary points. figure out why...
interior = slice(1, -1)  

#Phi_int = Phi[interior, :]
#Phi_dp_int = Phi_dp[interior, :]

Phi_int = Phi.copy()
Phi_dp_int = Phi_dp.copy()

# build the generalized eigenvalue problem Kv = \lambda Mv
K = -Phi_int.T @ Phi_dp_int
M = Phi_int.T @ Phi_int

eigvals, eigvecs = eigen.generalized_eigen(K, M)

# keep the same order for both!
order = np.argsort(eigvals.real)
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

low_k = 4
eigvals_low = eigvals_sorted[:low_k]
eigvecs_low = eigvecs_sorted[:, :low_k]

for (eigval, eigvec) in zip(eigvals_low, eigvecs_low.T):
    # print(eigvec)
    plt.plot(x, Phi @ eigvec, label=f"Eigenvalue: {eigval:.3f}")
plt.legend()
plt.show()