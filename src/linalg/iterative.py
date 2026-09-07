import numpy as np
import scipy

def jacobi_iteration(A, b, x0=None, tol=1e-13, max_iter=1000):
    D = np.diag(A)
    N = np.diag(D) - A
    
    # Broadcast division along columns: M^{-1} N
    M_inv_N = N / D[:, None] 
    c = b / D

    if x0 is None:
        xk = np.ones((A.shape[0],), dtype=np.float64)
    else:
        xk = np.copy(x0)

    for k in range(max_iter):
        residual = np.linalg.norm(A @ xk - b)
        if residual < tol:
            return xk

        xk = M_inv_N @ xk + c
    return xk

def gauss_seidel_iteration(A, b, x0=None, tol=1e-13, max_iter=10000):
    M = np.tril(A)
    N = np.triu(A, k=1)

    if x0 is None:
        xk = np.ones((A.shape[0],), dtype=np.float64)
    else:
        xk = np.copy(x0)

    # get c
    c = scipy.linalg.solve_triangular(M, b, lower=True)

    for k in range(max_iter):
        residual = np.linalg.norm(A @ xk - b)
        if residual < tol:
            print(f"Converged in {k} iterations")
            return xk

        xk = -scipy.linalg.solve_triangular(M, N @ xk, lower=True) + c
    print("Failed to converge")
    return xk

# conjugate gradient method time
# turns out I got this right first try - it's just a bad condition number
def conj_gradient(A, b, x0=None, tol=1e-10, convergence_padding=5):
    if x0 is None:
        x = np.ones((A.shape[0],), dtype=np.float64)
    else:
        x = np.copy(x0)

    residuals = []

    r = b - A @ x
    p = np.copy(r)
    gamma = np.dot(r, r)
    for i in range(convergence_padding * A.shape[0]):
        residuals.append(np.sqrt(gamma))
        if residuals[-1] < tol:
            print(f"Conjugate gradient converged early in {i} steps")
            return x, residuals
        y = A @ p
        alpha = gamma / np.dot(y, p)
        x += alpha * p
        r -= alpha * y
        mag_r = np.dot(r, r)
        beta = mag_r / gamma
        gamma = mag_r
        p = r + beta * p
    return x, residuals

def arnoldi_iteration(A, b, m):
    n = A.shape[0]
    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    Q[:, 0] = b / np.linalg.norm(b)

    for k in range(m):
        w = A @ Q[:, k]
        for i in range(k + 1):
            H[i, k] = Q[:, i] @ w
            w -= H[i, k] * Q[:, i]
        H[k + 1, k] = np.linalg.norm(w)
        if H[k + 1, k] < 1e-14 * np.linalg.norm(A): # rescaled version for the tolerance check
            return Q[:, :k+1], H[:k+1, :k+1] 
        Q[:, k + 1] = w / H[k + 1, k]

    return Q, H

def arnoldi_step(A, b, Q=None, H=None):
    n = A.shape[0]
    if Q is None: # initialize with m = 1
        Q = np.zeros((n, 2))
        Q[:, 0] = b / np.linalg.norm(b)
        H = np.zeros((2, 1))
    else: # expand the matrices to increase m by 1
        Q = np.pad(Q, pad_width=((0,0),(0,1))) # just the column
        H = np.pad(H, pad_width=(0,1)) # bofa

    # define k so we can just re-use code
    k = Q.shape[1] - 2
    w = A @ Q[:, k]
    for i in range(k + 1):
        H[i, k] = Q[:, i] @ w
        w -= H[i, k] * Q[:, i]
    H[k + 1, k] = np.linalg.norm(w)
    if H[k + 1, k] < 1e-14 * np.linalg.norm(A): # rescaled version for the tolerance check
        return Q[:, :k+1], H[:k+1, :k+1] 
    Q[:, k + 1] = w / H[k + 1, k]
    return Q, H   

def gmres_step(A, Q, H, b, x0):
    beta = np.linalg.norm(b)
    e1 = np.zeros(H.shape[0])
    e1[0] = beta
    y, _, _, _ = np.linalg.lstsq(H, e1, rcond=None)
    x = x0 + Q[:, :H.shape[1]] @ y
    return x, np.linalg.norm(b - A @ x)

def gmres(A, b, tol=1e-10, convergence_padding=5):
    pass