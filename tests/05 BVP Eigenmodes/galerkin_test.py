import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen
from testing import eigenmode_validation

N = 256
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

n_basis = 64
Phi = cv.build_neumann_basis(x, n_basis)   # self-normalized (unit-norm columns), as originally written
Phi_x = D @ Phi   # first derivative only, not second. integration by parts

w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

K = Phi_x.T @ W @ Phi_x
M = Phi.T @ W @ Phi

eigvals, eigvecs = eigen.generalized_eigen(K, M)

# keep the same order for both!
order = np.argsort(eigvals.real)
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

low_k = 8
eigvals_low = eigvals_sorted[:low_k]
eigvecs_low = eigvecs_sorted[:, :low_k]

# TODO: write up about finding this bug with the neumann behavior
# 1. wiggly. changed the formulation of Phi_dp
# 2. Wrong eigenvalue. I think this might actually be a bug in the matrix...?
#       - its consistent with the bad conditioning. I will rerun again with a smaller N
# 3. Solution - wrong inner product. did not take the spacing of the chebyshev nodes into account!
#       - i should re-run the dirichlet
#       - takeaway. the naive innre product is not always correct - validate!

# this is a good thing to get debugging practice with!
# use rayleigh quotient to approximate eigenvalue. good call actually!
# make an actual test suite. spend some time on this as well

for n, (eigval, eigvec) in enumerate(zip(eigvals_low, eigvecs_low.T)):
    reconstructed_eigvec = Phi @ eigvec
    reconstructed_eigvec /= np.linalg.norm(reconstructed_eigvec)  # fix scale
    if reconstructed_eigvec[np.argmax(np.abs(reconstructed_eigvec))] < 0:
        reconstructed_eigvec = -reconstructed_eigvec               # fix sign
    plt.plot(x, reconstructed_eigvec, label=f"$\\lambda$ = {eigval:.3f}")

    analytic = np.cos(n * np.pi/2 * (x + 1))
    analytic /= np.linalg.norm(analytic)              # same normalization convention
    if np.sign(analytic[0]) != np.sign(reconstructed_eigvec[0]):
        analytic *= -1
    plt.plot(x, analytic, "--", label=f"Analytic $\\lambda$ = {(np.pi * n/2) ** 2:.3f}")    
plt.legend()
plt.show()