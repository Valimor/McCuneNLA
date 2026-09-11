import numpy as np

from linalg import eigen
from linalg import decomposition as decomp

"""
A = np.array([
    [1, 1, 1],
    [2, 0, 1],
    [0, 2, 2]
], dtype=np.float64)
"""
n = 3
A = np.random.normal(size=(n,n))


A_new_givens, rotations = eigen.hessenberg_qr_step(A.copy())

Q, R = decomp.compute_householder_QR(A.copy())
A_new_householder = R @ Q

# don't do the frobenius norm to compare these - the determinants are different!
print(np.trace(A_new_givens), np.trace(A_new_householder))     # should match — trace is similarity-invariant
print(np.linalg.det(A_new_givens), np.linalg.det(A_new_householder))  # should match too
print(np.sort(np.linalg.eigvals(A_new_givens)))
print(np.sort(np.linalg.eigvals(A_new_householder)))
print(np.sort(np.linalg.eigvals(A)))   # all three of these should agree

# testing the step
m = n
A_test = A.copy()
Q_total = np.eye(n)

A_new, Q_total_new = eigen.QR_eigen_step_shifted_givens(A_test.copy(), Q_total, m)

# check: does Q_total_new correctly satisfy the similarity relationship for ONE step?
# it does!
"""
print(np.linalg.norm(Q_total_new.T @ A_test @ Q_total_new - A_new))
print(np.linalg.norm(Q_total_new.T @ Q_total_new - np.eye(n))) 
"""

# compute_householder_hessenberg is broken somehow. this is because the 0s are not all zero
H0, Q0 = decomp.compute_householder_hessenberg(A.copy())
print(np.abs(np.tril(H0, -2)).max())