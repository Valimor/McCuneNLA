import numpy as np

from linalg import decomposition, iterative, eigen

def eigenmode_check(A, evals, evecs):
    for (eval, evec) in zip(evals, evecs.T):
        print(f"Error for eval: {eval}: {np.linalg.norm(A @ evec - eval * evec)}")

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

def random_spd_bigger_eigs(n, l, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(l, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

rng = np.random.default_rng(seed=10)

n = 100
M = random_spd_bigger_eigs(n, 1, 10, rng)
A = np.copy(M)

# intialize b. this is arbitrary for now
b = np.ones((n,), dtype=np.float64)

# compute the n-dimensional arnoldi decomposition
Q, H = iterative.arnoldi_iteration(A, b, 50)
if H.shape[0] != H.shape[1]:
    H_square = H[:-1,:]
    Q_reshaped = Q[:,:-1]
else:
    H_square = H
    Q_reshaped = Q
ritz_values, ritz_vecs = eigen.QR_eigen_givens_algorithm(H_square)

approx_evecs = Q_reshaped @ ritz_vecs

# check error for H_square
eigenmode_check(A, ritz_values, approx_evecs)