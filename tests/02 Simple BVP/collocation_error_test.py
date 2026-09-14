import numpy as np
import scipy
import matplotlib.pyplot as plt

from linalg import decomposition as decomp
from spectral import chebyshev

# functions
def rescaled_sin(x):
    return -1 * np.pi **2 * np.sin(np.pi * x)

def rescaled_sin_soln(x):
    return np.sin(np.pi * x)

def constant(x):
    return -1 * np.ones_like(x)

def constant_soln(x):
    return 1/2 * (1 - x ** 2)

def exponential(x):
    return np.exp(x)

def exponential_soln(x):
    return np.exp(x)

def high_freq_sin(x, k=5):
    return -1 * np.pi **2 * k ** 2 * np.sin(np.pi * x * k)

def high_freq_sin_soln(x, k=5):
    return np.sin(np.pi * x * k)

def smooth_v_kink(x):
    return 6 * np.abs(x)

def smooth_v_kink_soln(x):
    return np.abs(x) ** 3

N = 64
sizes = np.logspace(0, 12, num=N, base=2, dtype=np.int64)
conds = np.zeros_like(sizes, dtype=np.float64)

simulated_functions = [
    rescaled_sin,
    constant,
    exponential,
    high_freq_sin,
    smooth_v_kink
]

functions_bc = {
    rescaled_sin:(0,0),
    constant:(0,0),
    exponential:(np.exp(-1), np.exp(1)),
    high_freq_sin:(0,0),
    smooth_v_kink:(1,1)
}

functions_names = {
    rescaled_sin:"Sinusoid",
    constant:"Constant",
    exponential:"Exponential",
    high_freq_sin:"Higher Frequency Sinusoid",
    smooth_v_kink:"Smooth with Kink"
}

solutions_dict = {
    constant:constant_soln,
    rescaled_sin:rescaled_sin_soln,
    exponential:exponential_soln,
    high_freq_sin:high_freq_sin_soln,
    smooth_v_kink:smooth_v_kink_soln
}

errors_dict = {
    funct:np.zeros_like(sizes, dtype=np.float64) for funct in simulated_functions
}

# try different boundary conditions

for idx, size in enumerate(sizes):

    D, x = chebyshev.chebyshev_diff_matrix(size)
    D2 = D @ D

    # add dirichlet BC
    D2[0,:] = 0
    D2[0,0] = 1
    D2[-1,:] = 0
    D2[-1,-1] = 1

    conds[idx] = np.linalg.cond(D2)

    print(f"Computed D2 condition for {size}x{size} at index {idx}")

    for function in simulated_functions:
        f = function(x)

        # add boundary conditions
        bc = functions_bc[function]
        f[-1] = bc[0]
        f[0] = bc[1]

        # compute soln
        # soln = decomp.solve_QR(Q, R, f)
        soln = np.linalg.solve(D2, f)

        # compute error
        error = np.max(np.abs(solutions_dict[function](x) - soln))
        errors_dict[function][idx] = error

for function in simulated_functions:
    plt.loglog(sizes, errors_dict[function], label=functions_names[function])
plt.loglog(sizes, conds, label="Condition numbers")
plt.legend()
plt.show()