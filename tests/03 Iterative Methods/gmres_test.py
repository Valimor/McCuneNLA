import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from linalg import iterative

rng = np.random.default_rng(seed=10)

def random_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs)

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

N = 200
condition_number = 10
A = random_spd_with_condition(N, condition_number, rng)
print(np.linalg.cond(A))
b = rng.normal(size=(N,))

# solving
"""
Q, H = iterative.arnoldi_iteration(A, b, m)
x0 = np.zeros(A.shape[0])
x_exact, res = iterative.gmres_step(A, Q, H, b, x0)
x = np.linalg.solve(A, b)
print(np.linalg.norm(x - x_exact))
"""

x0 = np.zeros(A.shape[0])
Qi, Hi = iterative.arnoldi_step(A, b)
residuals = np.zeros((N,))
x_1, res_1 = iterative.gmres_step(A, Qi, Hi, b, x0)
residuals[0] = res_1

for i in range(N - 1):
    # make the Q and H
    Qi, Hi = iterative.arnoldi_step(A, b, Q=Qi, H=Hi)
    x_i, res_i = iterative.gmres_step(A, Qi, Hi, b, x0)
    residuals[i + 1] = res_i

# print the first index where the minimum was found.
try:
    print(np.min(np.arange(N)[residuals < residuals[-1]]))
except:
    print("Smallest residual was final")

plt.plot(np.arange(N), np.log(residuals), label="ln(Residuals)")
plt.hlines(np.log(residuals[-2]), xmin=0, xmax=N-1, linestyles="--", color="black", label="Penultimate residual")
plt.ylabel("ln||Residual||")
plt.xlabel("Number of steps")
plt.legend()
plt.show()