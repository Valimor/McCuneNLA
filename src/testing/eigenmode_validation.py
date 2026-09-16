import numpy as np

def rayleigh_quotient(A, v):
    return (v.T @ A @ v) / (v.T @ v)

def generalized_rayleigh_quotient(A, B, v):
    return (v.T @ A @ v) / (v.T @ B @ v)