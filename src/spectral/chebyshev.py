import numpy as np

def chebyshev_points(N):
    # might need to validate this.
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)

    return x

def chebyshev_diff_matrix(N):
    """
    Returns the (N+1)x(N+1) Chebyshev differentiation matrix D
    and the Chebyshev points x, on [-1, 1].
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])

    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)

    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1.0) ** j

    X = np.tile(x, (N + 1, 1)).T          # each column is a copy of x
    dX = X - X.T                           # dX[i,j] = x[i] - x[j]

    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))  # off-diagonal entries
    np.fill_diagonal(D, 0)
    np.fill_diagonal(D, -np.sum(D, axis=1))           # diagonal entries

    return D, x

def build_dirichlet_basis(x, n_basis):
    # x: Chebyshev points (length N+1), N: grid parameter (N-1 basis functions, k=0..N-2)
    n_points = len(x)
    theta = np.arccos(x)  # recover the angle underlying each Chebyshev point
    Phi = np.zeros((n_points, n_basis))
    for k in range(n_basis):
        Phi[:, k] = np.cos(k * theta) - np.cos((k + 2) * theta)
    return Phi

def build_neumann_basis(x, n_basis):
    n_points = len(x)
    theta = np.arccos(x)
    Phi = np.zeros((n_points, n_basis))
    for k in range(n_basis):
        a = 1
        b = - k ** 2 / (k + 2) ** 2
        phi_k = a * np.cos(k * theta) + b * np.cos((k + 2) * theta)
        Phi[:, k] = phi_k / np.linalg.norm(phi_k)   # normalize
    return Phi

def build_mixed_basis(x, n_basis):
    # Dirichlet at x=-1, Neumann at x=1
    n_points = len(x)
    theta = np.arccos(x)
    Phi = np.zeros((n_points, n_basis))
    for k in range(n_basis):
        b = -(k**2 + (k+1)**2) / ((k+1)**2 + (k+2)**2)
        a = 1 + b
        phi_k = np.cos(k*theta) + a*np.cos((k+1)*theta) + b*np.cos((k+2)*theta)
        Phi[:, k] = phi_k / np.linalg.norm(phi_k)
    return Phi

def build_robin_basis(x, n_basis, alpha=0.0):
    # robin boundary conditions u(-1) = 0, u'(1) + \alpha u'(1) = 0
    n_points = len(x)
    theta = np.arccos(x)
    Phi = np.zeros((n_points, n_basis))
    for k in range(n_basis):
        M = np.array([
            [-1, 1],
            [((k+1)**2 + alpha), ((k+2)**2 + alpha)]
        ])
        V = np.array([-1, -k**2 - alpha])
        c = np.linalg.solve(M, V)
        a = c[0]
        b = c[1]
        phi_k = np.cos(k*theta) + a*np.cos((k+1)*theta) + b*np.cos((k+2)*theta)
        Phi[:, k] = phi_k / np.linalg.norm(phi_k)
    return Phi

def clenshaw_curtis_weights(N):
    """
    Returns quadrature weights w (length N+1) for the Chebyshev points
    x_j = cos(j*pi/N), j = 0,...,N, such that
        integral_{-1}^{1} f(x) dx  ≈  sum_j w_j f(x_j)
    """
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)

    # standard Clenshaw-Curtis weight formula via a cosine sum
    for j in range(N + 1):
        s = 0.0
        for k in range(1, N // 2 + 1):
            c = 2.0 if 2 * k != N else 1.0
            s += c / (4 * k**2 - 1) * np.cos(2 * k * theta[j])
        w[j] = 1.0 - s

    # endpoint and interior scaling
    w /= N
    w[0] /= 2
    w[-1] /= 2
    w *= 2  # overall factor from the [-1,1] interval

    return w