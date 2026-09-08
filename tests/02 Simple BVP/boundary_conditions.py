import numpy as np
import scipy
import matplotlib.pyplot as plt

from linalg import decomposition as decomp

def chebyshev_diff_matrix(N):
    """
    Returns the (N+1)x(N+1) Chebyshev differentiation matrix D
    and the Chebyshev points x, on [-1, 1].
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])

    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)

    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1.0) ** j

    X = np.tile(x, (N + 1, 1)).T          # each column is a copy of x
    dX = X - X.T                           # dX[i,j] = x[i] - x[j]

    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))  # off-diagonal entries
    np.fill_diagonal(D, 0)
    np.fill_diagonal(D, -np.sum(D, axis=1))           # diagonal entries

    return D, x

N = 32 # i like

I = np.eye(N + 1)
D, x = chebyshev_diff_matrix(N)
D2 = D @ D

# make a fun matrix
a, b, c = (0, 0, 1)
C = a * I + b * D + c * D2 # f = u'' + u' + u

# add neumann BC (corresponds to a slope of 1 at 1 and a value of zero)
# gotta understand this
C[0,:] = D[0,:]
C[-1,:] = 0
C[-1,-1] = 1

# u'' = f
# let f be a constant and see what happens!
f = np.zeros_like(x) - 1.0
f[0] = 1.0 
f[-1] = 1.0 

u = np.linalg.solve(C, f)
plt.plot(x, u)
plt.show()