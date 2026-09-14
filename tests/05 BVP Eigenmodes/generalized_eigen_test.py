import numpy as np

from linalg import decomposition as decomp
from linalg import eigen

# TODO: make a library of test functions
#   - new module in linalg? call it test_funcs or something
def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T


rng = np.random.default_rng(seed=10)
n = 30

M_test = random_spd_with_condition(n, 10, rng)
K_test = random_spd_with_condition(n, 1, rng)

# write the defining equation out explicitly in a comment directly above any residual check
# i messed this up a few times. keep it consistent!
g_evals, g_evecs = eigen.generalized_eigen(K_test, M_test)

for geval, gevec in zip(g_evals, g_evecs.T):
    print(np.linalg.norm((K_test - geval * M_test) @ gevec ))