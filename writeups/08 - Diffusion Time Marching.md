# Diffusion Time Marching

## What was implemented

The diffusion equation, which is a time-marching equivalent of the poisson equation from the previous section, was solved via time-marching. To do this a series of three different methods were implemented: Explicit Runge-Kutta 4, Implicit Euler, and Crank-Nicolson. They were implemented using chebyshev collocation, galerkin, and finite differences methods.

(do the finite differences)

## Validation

- how you know it's correct. Comparison against a known analytic solution, against scipy/numpy, or a convergence-order check. This is the most important section for a research audience. it's the difference between "I wrote code" and "I verified my code is right."

## The interesting numerical behavior

- conditioning plots, convergence plots, error vs. N, eigenvalue distributions. This is where the actual research content lives (e.g., "Chebyshev differentiation matrix condition number grows like O(N^4)" is a finding, not a footnote).

## What surprised me or broke

- genuinely valuable for a research portfolio. "I expected X, got Y, here's why" reads as more sophisticated than a clean success story, and it's honest.
Plots, generated from a script you keep alongside the writeup so they're reproducible, not just pasted images
