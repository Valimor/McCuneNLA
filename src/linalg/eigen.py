import numpy as np

# my prewritten stuff
import linalg.decomposition as decomp
import linalg.iterative

"""
====================================================================================
SIMPLE QR FUNCTIONS
====================================================================================
"""

# this won't use shifting nor deflation. can compare!
def QR_eigen_step(A, Q_total=None):
    Q, R = decomp.compute_householder_QR(A)
    if Q_total is None:
        Q_total = Q
    else:
        Q_total = Q_total @ Q
    A = R @ Q
    return A, Q_total

# the idea is to use A - muI to converge faster
def QR_eigen_step_shifted(A, Q_total=None, m=None):
    if m is None:
        m = A.shape[0]
    Asub = A[:m, :m]
    mu = Asub[-1, -1]
    Q, R = decomp.compute_householder_QR(Asub - mu * np.eye(m))
    Asub_new = R @ Q + mu * np.eye(m)
    A = A.copy()
    A[:m, :m] = Asub_new

    if Q_total is not None:
        Q_full = np.eye(A.shape[0])
        Q_full[:m, :m] = Q
        Q_total = Q_total @ Q_full

    return A, Q_total

def get_evals_from_2x2(A, tol=1e-12):
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(A[i+1, i]) < tol:
            eigenvalues.append(A[i, i])
            i += 1
        else:
            a, b, c, d = A[i,i], A[i,i+1], A[i+1,i], A[i+1,i+1]
            disc = (a+d)**2 - 4*(a*d - b*c)
            sqrt_disc = np.emath.sqrt(disc)   # automatically deals w complex
            eigenvalues.append((a+d + sqrt_disc)/2)
            eigenvalues.append((a+d - sqrt_disc)/2)
            i += 2
    return np.array(eigenvalues)

def eigvec_2x2(B, lam):
    a, b, c, d = B[0,0], B[0,1], B[1,0], B[1,1]
    if abs(b) > 1e-10:
        w = np.array([b, lam - a], dtype=complex)
    else:
        w = np.array([lam - d, c], dtype=complex)
    return w / np.linalg.norm(w)

# note: this also has to deal with the Q_total if it is to return the e-vectors
def reduce_2x2_block_to_complex(A, Q_total, m=None):
    if m is None:
        m = A.shape[0]
    B = A[m-2:m, m-2:m]
    a, b, c, d = B[0,0], B[0,1], B[1,0], B[1,1]
    disc = (a+d)**2 - 4*(a*d - b*c)
    sqrt_disc = np.emath.sqrt(disc)
    lam1 = (a+d + sqrt_disc) / 2
    lam2 = (a+d - sqrt_disc) / 2
    w1 = eigvec_2x2(B, lam1)
    v1 = Q_total[:, m-2:m] @ w1
    w2 = eigvec_2x2(B, lam2)
    v2 = Q_total[:, m-2:m] @ w2
    return (lam1, v1), (lam2, v2)

# go through and comment this to understand
# note: this is the simplest version and it takes a long time to converge
def QR_eigen_algorithm(A, tol=1e-10, max_steps=500):
    n = A.shape[0]
    A = A.copy() # why?
    Q_total = np.eye(n)
    m = n

    # might make these np.array
    eigenvalues = []
    eigenvectors = []

    while m > 0:
        if m == 1:
            eigenvalues.append(A[0, 0])
            eigenvectors.append(Q_total[:, 0])
            m -= 1
            continue

        for _ in range(max_steps):
            A, Q_total = QR_eigen_step_shifted(A, Q_total, m)
            if abs(A[m-1, m-2]) < tol:
                break  # trailing 1x1 has converged

        if abs(A[m-1, m-2]) < tol:
            eigenvalues.append(A[m-1, m-1])
            eigenvectors.append(Q_total[:, m-1])
            m -= 1        # deflate: trailing eigenvalue is done
        else:
            (lam1, v1), (lam2, v2) = reduce_2x2_block_to_complex(A, Q_total, m)
            eigenvalues.extend([lam1, lam2])
            eigenvectors.extend([v1, v2])
            m -= 2        # trailing 2x2 block (likely complex pair), deflate both at once

    return np.array(eigenvalues), np.array(eigenvectors).T # transpose so columns!

"""
====================================================================================
HESSENBERG QR FUNCTIONS
====================================================================================
"""
# TODO: make a version of QR that starts by reducing a matrix to upper hessenberg
# - then it finds the evals
# TODO: determine givens vs householder rotation
# - looks like givens is better for sparse and householder better for dense
# - will start with householder. i am not trying to build a full sparse library too
def hessenberg(A):
    # converts a matrix to an upper hessenberg form with householder rotations.
    # unclear if this should return the rotation matrix as well
    # will likely be similar to the householder QR function
    pass

"""
====================================================================================
IMPLICIT QR FUNCTIONS
====================================================================================
"""
# TODO: implement bulge chasing. does the same math as the above, but more stably
# - same process

"""
====================================================================================
ARNOLDI FUNCTIONS
====================================================================================
"""

# TODO:
# 1. implement a version of this that first transforms the matrix to upper hessenberg form
#   the use the existing methods on that
#   a. shift
#   b. sequence of givens rotations to zero out (i + 1, i) for the shifted matrix
#   c. multiply R by the givens rotations on the right
#   d. undo the shift
# 2. implement an implicit QR algorithm
#   a. bulge-chasing algorthim.
# 3. implement Arnoldi on top of the existing algorithms