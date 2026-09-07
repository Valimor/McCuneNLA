import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from linalg import iterative

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

rng = np.random.default_rng(seed=10)

N = 100
cond_numbers = np.logspace(0, 8, 15)  # κ from 10^0 to 10^8

iters_to_converge = []
final_residuals = []
residual_histories = {}  # keep full history only for a few select cond_numbers

selected_indices = set(np.linspace(0, len(cond_numbers) - 1, 5, dtype=int))

residual_histories = {}
for idx, k in enumerate(cond_numbers):
    A = random_spd_with_condition(N, cond_number=k, rng=rng)
    b = rng.normal(size=N)
    x_cg, residuals = iterative.conj_gradient(A, b, tol=1e-10, convergence_padding=10)
    iters_to_converge.append(len(residuals))
    final_residuals.append(residuals[-1])
    if idx in selected_indices:
        residual_histories[k] = residuals

plt.figure()
plt.semilogx(cond_numbers, iters_to_converge, 'o-', label="Empirical")
plt.xlabel("Condition number κ")
plt.ylabel("Iterations to converge")
plt.title("Condition number vs. Iterations for Convergence")
plt.legend()
plt.show()

plt.figure()
for k, hist in residual_histories.items():
    plt.semilogy(hist, label=f"κ={k:.0e}")
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Error vs. Iteration for different Condition Numbers")
plt.legend()
plt.show()