from linalg import decomposition as decomp
import numpy as np

rng = np.random.default_rng(seed=11)

n = 8
A = rng.normal(size = (n,n))

"""
Q, H = decomp.compute_householder_hessenberg(A.copy())
print(np.linalg.norm(Q.T @ A @ Q - H))          # similarity relationship holds
print(np.linalg.norm(Q.T @ Q - np.eye(n)))       # Q still orthogonal
print(np.sort(np.linalg.eigvals(A)))
print(np.sort(np.linalg.eigvals(H)))              # eigenvalues preserved exactly
np.set_printoptions(suppress=True)

# the problem is at least with the compute_householder_hessenberg function
H0, Q0 = decomp.compute_householder_hessenberg(A.copy())
print(np.abs(np.tril(H0, -2)).max())
"""

Q0, H0 = decomp.compute_householder_hessenberg(A.copy())
print(np.abs(np.tril(H0, -2)).max())