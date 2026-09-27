import numpy as np
import math

from linalg import eigen

def polynomial_companion_matrix(p_coefs):
    # recursion
    if p_coefs.shape[0] == 0:
        return np.nan
    if np.abs(p_coefs[0]) < 1e-14:
        return polynomial_companion_matrix(p_coefs[1:])

    # protect
    p_coefs = p_coefs.copy()

    # monic
    p_coefs = p_coefs/p_coefs[0]
    n = p_coefs.shape[0] - 1

    # make the block matrix
    C = np.zeros((n,n), dtype=np.float64)
    C[1:n,0:n-1] = np.eye(n-1)
    C[:,-1] = -p_coefs[1:][::-1]
    return C

def make_callable_poly(p):
    def polynomial(x):
        total = 0
        for coef in p:
            total *= x
            total += coef
        return total
    return polynomial

p = np.array([1/math.factorial(i) for i in reversed(range(1,5))]) # truncated exponential. finding when it's 1
polynomial = make_callable_poly(p)
roots, _ = eigen.QR_eigen_givens_algorithm(polynomial_companion_matrix(p))
print(np.min(roots))