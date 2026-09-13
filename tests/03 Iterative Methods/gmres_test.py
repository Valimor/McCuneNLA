import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from linalg import iterative
from linalg import decomposition

rng = np.random.default_rng(seed=10)

def random_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs)

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

def gmres_residual_history(A, b, N):
    x0 = np.zeros(A.shape[0])
    residuals = np.zeros((N,))

    Qi, Hi = iterative.arnoldi_step(A, b)
    x_1, res_1 = iterative.gmres_step(A, Qi, Hi, b, x0)
    residuals[0] = res_1

    for i in range(N - 1):
        Qi, Hi = iterative.arnoldi_step(A, b, Q=Qi, H=Hi)
        x_i, res_i = iterative.gmres_step(A, Qi, Hi, b, x0)
        residuals[i + 1] = res_i

    return residuals

N = 200
condition_numbers = [1e1, 1e3, 1e5, 1e7, 1e9]

fig, ax = plt.subplots(figsize=(8, 6))

for cond_number in condition_numbers:
    A = random_spd_with_condition(N, cond_number, rng)
    b = rng.normal(size=(N,))
    residuals = gmres_residual_history(A, b, N)
    ax.plot(np.arange(N), np.log10(residuals), label=f"$\\kappa$ = {cond_number:.0e}")

ax.set_ylabel("$log_{10}$||Residual||")
ax.set_xlabel("Number of steps")
ax.set_title("GMRES convergence vs. condition number")
ax.legend()
plt.show()