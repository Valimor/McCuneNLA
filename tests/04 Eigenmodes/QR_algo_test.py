import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

from linalg import decomposition as decomp
from linalg import eigen
from linalg import iterative

rng = np.random.default_rng(seed=20)

n = 4
R = rng.normal(size=(n,n))
Q, _ = decomp.compute_householder_QR(R)

# bet
# M = Q.T @ np.diag([3.0, 1.5, 1.2, 1.1, 0.5, 0.7]) @ Q
theta = np.pi/3
M = np.array([
    [np.cos(theta), np.sin(theta), 0, 0],
    [-np.sin(theta), np.cos(theta), 0, 0],
    [0, 0, 2.0, 0],
    [0, 0, 0, 3.0]
])
M = rng.normal(size=(n,n))
M = Q.T @ M @ Q
Q_total = np.eye(*M.shape)

shifted = True
steps = 100
A = np.copy(M)
evals, evecs = eigen.QR_eigen_algorithm(A)
print(evals)