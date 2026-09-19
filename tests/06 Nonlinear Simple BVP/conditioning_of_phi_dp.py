import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

# collocation points
N = 256

# computing the number of points to check
n_basis_arr = np.logspace(3,8,num=16,base=2, dtype=np.int64)
cond_arr = np.zeros((n_basis_arr.shape[0],), dtype=np.float64)
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

for idx, n_basis in enumerate(n_basis_arr):
    Phi = cv.build_dirichlet_basis(x, n_basis)
    Phi_dp = D2 @ Phi
    cond_arr[idx] = np.linalg.cond(Phi_dp)

print(cond_arr)

plt.loglog(n_basis_arr, cond_arr)
plt.ylabel("Condition number")
plt.xlabel("Number of bases")
plt.title("Condition of $\Phi''$ vs. number of bases")
plt.show()