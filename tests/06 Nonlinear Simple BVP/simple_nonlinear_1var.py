import numpy as np
import scipy

epsilon = 0.01
max_steps = 3000

tol = 1e-14

x = 3.0

# computing lambert w of 2
for step in range(max_steps):
    R = 2 - x * np.exp(x)
    if np.abs(R) < tol:
        break

    x = x + epsilon * R
print(step, x)
print(np.abs(scipy.special.lambertw(2) - x))