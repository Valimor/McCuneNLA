import numpy as np
import matplotlib.pyplot as plt

from linalg import decomposition as decomp

from testing import test_matrices

rng = np.random.default_rng(seed=10)

ns = np.logspace(1, 3, num=5, dtype=np.int64)
conds = np.logspace(1, 3, num=32)
errors = np.zeros((ns.shape[0], conds.shape[0]), dtype=np.float64)
for i, n in enumerate(ns):
    print(n)
    for j, cond in enumerate(conds):
        M = test_matrices.random_spd_with_condition(n, cond, rng)
        L = decomp.compute_cholesky(M)

        errors[i,j] = np.linalg.norm(L @ L.T - M)

for idx, error in enumerate(errors):
    plt.loglog(conds, error, label=f"Matrix dimension: ${ns[idx]}\\times{ns[idx]}$")
plt.title("Reconstruction error vs. Condition Number")
plt.xlabel("Condition number")
plt.ylabel("$||LL^T - M||$")
plt.legend()
plt.show()