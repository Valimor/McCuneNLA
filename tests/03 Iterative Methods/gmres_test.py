import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from linalg import iterative

rng = np.random.default_rng(seed=10)

N = 5
A = rng.normal(size=(N,N))
A = A @ A.T # lets see how this plays
A = scipy.linalg.hilbert(N)
# b = np.ones((N,), dtype=np.float64) # equivalent to solving Ax = vector of ones
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

plt.plot(np.arange(N), np.log(residuals), label="ln(Residuals)")
plt.hlines(np.log(residuals[-2]), xmin=0, xmax=N-1, linestyles="--", color="black", label="Penultimate residual")
plt.ylabel("ln||Residual||")
plt.xlabel("Number of steps")
plt.legend()
plt.show()