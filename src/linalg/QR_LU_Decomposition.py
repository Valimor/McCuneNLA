import numpy as np

# use LU decomposition to set up gaussian elimination
# make a version with and without partial pivoting
# also make a function that does the triangle solution
"""
=====================================================================================
DECOMPOSERS
=====================================================================================
"""

# from Algorithm 6.1 in the book
# no partial pivoting.. could be a problem if A[k,k] is zero
# claude suggested version. uses an outer product to replace the i and j loops
# gaussian elimination
def compute_LU(A):
    n1, n2 = A.shape
    assert n1 == n2, "LU Decomposition only works on square matrices"
    A = A.astype(np.float64)
    n = n1
    for k in range(n - 1):
        if abs(A[k,k]) < 1e-14:
            raise ValueError(f"Zero pivot encountered at step {k}; partial pivoting required")
        A[k+1:, k] = A[k+1:, k] / A[k,k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])
    return A

def compute_LU_pivot(A):
    n1, n2 = A.shape
    assert n1 == n2, "LU Decomposition only works on square matrices"
    n = n1
    A = A.astype(np.float64)
    P = np.arange(n)
    for k in range(n - 1):
        # find the index of the maximum value
        max_idx = np.argmax(np.abs(A[k:,k])) + k

        # swap the values
        A[[k, max_idx], :] = A[[max_idx, k], :]
        P[[k, max_idx]] = P[[max_idx, k]]

        # check the magnitude
        if abs(A[k,k]) < 1e-14:
            raise ValueError(f"Zero pivot encountered in step {k}. Matrix may be singular")
        A[k+1:, k] = A[k+1:, k] / A[k,k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])
    return A, P

# make two versions of this
# one with gram-schmidt, one with householder
def compute_gram_schmidt_QR(A):
    n = A.shape[0]
    Q = np.zeros_like(A, dtype=np.float64)
    R = np.zeros_like(A, dtype=np.float64)

    # i vectorized this with the help of claude
    for i in range(n):
        Q[:,i] = A[:,i] # loading the next value in
        R[:i,i] = Q[:,:i].T @ A[:,i] # finding the projections onto the previous values
        Q[:,i] -= R[:i,i] @ Q[:,:i].T # subtracting the projections off
        R[i,i] = np.linalg.norm(Q[:,i]) # getting magnitude of the residual
        Q[:,i] = Q[:,i] / R[i,i] # divide by magnitude of residual

    return Q, R

def compute_modified_gram_schmidt_QR(A):
    n = A.shape[0]
    Q = np.zeros_like(A, dtype=np.float64)
    R = np.zeros_like(A, dtype=np.float64)

    # i vectorized this with the help of claude
    for i in range(n):
        Q[:,i] = A[:,i] # loading the next value in
        for k in range(i):
            R[k,i] = Q[:,k] @ Q[:,i] # finding the projections onto the previous values. do this immediately
            Q[:,i] -= R[k,i] * Q[:,k] # subtracting the projections off
        R[i,i] = np.linalg.norm(Q[:,i]) # getting magnitude of the residual
        Q[:,i] = Q[:,i] / R[i,i] # divide by magnitude of residual

    return Q, R

# helper function for householder
def compute_h_small(x):
    # initialize the v
    v = np.zeros_like(x, dtype=np.float64)
    sign = 1.0 if x[0] >= 0 else -1.0
    alpha = -sign * np.linalg.norm(x)
    v[0] = alpha
    v = x - v

    return np.eye(len(v)) - 2 * np.outer(v, v) / (v @ v)

def compute_householder_QR(A):
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)
    R = np.copy(A)

    # do a series of reflections to zero out the below-diagonal elements
    for k in range(n - 1):
        h_small = compute_h_small(R[k:,k])
        R[k:,k:] = h_small @ R[k:,k:]
        Q[:,k:] = Q[:,k:] @ h_small

    return Q, R

"""
=====================================================================================
SOLVERS
=====================================================================================
"""

# claude suggested version. I understand this, so it's ok
def solve_triangular(T, b, lower=True):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)

    if np.any(np.abs(np.diag(T)) < 1e-14): 
        raise ValueError("Any zero pivots will cause errors. partial pivoting required")

    if lower:
        for i in range(n):
            x[i] = (b[i] - T[i, :i] @ x[:i]) / T[i, i]
    else:
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - T[i, i+1:] @ x[i+1:]) / T[i, i]

    return x

def solve_LU(LU, b, P=None):
    n = len(b)
    y = np.zeros_like(b, dtype=np.float64)
    x = np.zeros_like(b, dtype=np.float64)

    # partial pivoting support. literally just a permutation
    if not (P is None):
        b = b[P]

    if np.any(np.abs(np.diag(LU)) < 1e-14): 
        raise ValueError("Any zero pivots will cause errors. partial pivoting required")

    for i in range(n):
        y[i] = (b[i] - LU[i, :i] @ y[:i])

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - LU[i, i+1:] @ x[i+1:]) / LU[i, i]

    return x

def solve_QR(Q, R, b):
    QTb = Q.T @ b
    return solve_triangular(R, QTb, lower=False)

# TODO: come back to banded matrices
# chebyshev matrices are not banded

"""
=====================================================================================
HELPERS
=====================================================================================
"""
def make_L_U_from_LU(LU):
    L = np.eye(*LU.shape) + np.tril(LU, k=-1)
    U = np.triu(LU)
    return L, U

def reconstruct_from_LU(LU):
    L, U = make_L_U_from_LU(LU)
    return L @ U

def reconstruct_from_LU_pivot(LU, P):
    A = reconstruct_from_LU(LU)
    inv_P = np.empty_like(P)
    inv_P[P] = np.arange(len(P))
    return A[inv_P]

def reconstruct_QR(Q, R):
    return Q @ R