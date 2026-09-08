import numpy as np

# my prewritten stuff
import linalg.decomposition as decomp # probably going to rename this.
import iterative

rng = np.random.default_rng(seed=20)

n = 6
R = rng.normal(size=(n,n))
Q, _ = decomp.compute_householder_QR(R)

# bet
M = Q.T @ np.diag([3.0, 1.5, 1.2, 1.1, 0.5, 0.7]) @ Q

shifted = True

steps = 200
A = np.copy(M)
A_np = np.copy(M)
Q_total = np.eye(*A.shape)
for k in range(steps):
    if shifted:
        mu = A[-1,-1]
        Q, R = decomp.compute_householder_QR(A - mu * np.eye(*A.shape))
        A = R @ Q + mu * np.eye(*A.shape)
    else:
        Q, R = decomp.compute_householder_QR(A)
        A = R @ Q
    Q_total = Q_total @ Q
    if k % 5 == 0:
        eigs = np.sort(np.diag(A))
        print(np.abs(eigs[3] - eigs[2]))

for i in range(n):
    lam = A[i,i]
    v = Q_total[:, i]
    print(f"Eigenvalue at idx {i} has error {np.linalg.norm(M @ v - lam * v)}")