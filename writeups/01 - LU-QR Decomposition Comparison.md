# Markdown Note Template

*note: to preview the .md files I should press ctrl+k, v*
*to switch between the two I should press ctrl+shift+v*

What each writeup should contain

Note: here are the metrics
QR:
- Runtime
- Reconstruction Error
- Orthogonality Error
- Backwards Error

LU:
- Runtime
- Reconstruction Error
- Backwards Error
- Standard Growth Metric Factor

## What was implemented

- one paragraph, plain description, no code.

## Validation

- how you know it's correct. Comparison against a known analytic solution, against scipy/numpy, or a convergence-order check. This is the most important section for a research audience. it's the difference between "I wrote code" and "I verified my code is right."

## The interesting numerical behavior

- conditioning plots, convergence plots, error vs. N, eigenvalue distributions. This is where the actual research content lives (e.g., "Chebyshev differentiation matrix condition number grows like O(N^4)" is a finding, not a footnote).

## What surprised you or broke

- genuinely valuable for a research portfolio. "I expected X, got Y, here's why" reads as more sophisticated than a clean success story, and it's honest.
Plots, generated from a script you keep alongside the writeup so they're reproducible, not just pasted images
