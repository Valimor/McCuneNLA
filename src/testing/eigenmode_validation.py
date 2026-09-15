import numpy as np

def rayleigh_quotient(A, v):
    return (v.T @ A @ v) / (v.T @ v)