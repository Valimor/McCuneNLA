import numpy as np

from spectral import chebyshev as cv

N = 32
w = cv.clenshaw_curtis_weights(N)
x = np.cos(np.pi * np.arange(N+1) / N)

print(np.sum(w))                    # should be 2.0 (integral of 1 over [-1,1])
print(np.sum(w * x))                 # should be ~0 (integral of x, an odd function)
print(np.sum(w * x**2))              # should be 2/3 (exact integral of x^2 over [-1,1])
print(np.sum(w * np.sin(np.pi*x)))   # compare against a known transcendental integral if you want a non-polynomial check