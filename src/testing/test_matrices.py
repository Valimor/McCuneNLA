import numpy as np

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

def normal_distributed_random(n, rng):
    return rng.normal(size=(n, n))

def wilkinson_growth_matrix(n):
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = -1
    A[:, -1] = 1
    return A

def pivot_triggering_matrix(n, seed=0):
    rng_local = np.random.default_rng(seed)
    A = rng_local.normal(size=(n, n))
    A[0, 0] = 1e-8
    A[-1, 0] = 8.0
    return A