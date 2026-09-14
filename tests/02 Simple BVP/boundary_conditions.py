import numpy as np
import scipy
import matplotlib.pyplot as plt

from linalg import decomposition as decomp
from spectral import chebyshev

N = 256 # i like

I = np.eye(N + 1)
D, x = chebyshev.chebyshev_diff_matrix(N)
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