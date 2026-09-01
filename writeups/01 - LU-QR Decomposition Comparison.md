# 01 - LU and QR Decomposition Comparisons

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

I implemented my own versions of LU and QR decompositions in Python. For LU, this consisted of regular Gaussian Elimination and Gaussian elimination with partial pivoting. For QR, this consisted of Gram-Schmidt, modified Gram-Schmidt, and Householder. Then, I validated them against built-in versions of the same decompositions (in SciPy and NumPy respectively). The tests included a variety of different metrics across a few different kinds of matrices.

## Validation

The written code performed similarly to the built-in methods in virtually every metric for well-behaved matrices.

## The interesting numerical behavior

The QR code behaved much better for hilbert matrices than LU decomposition did. Even with partial pivoting, LU failed to find a reasonable result for n > 12. Furthermore, hilbert matrices provided the worst numerical performance of them all and QR matrices consistently had lower reconstruction and backwards errors. Runtime was also consistent across all schemes, irrelevant of matrix type.

## What surprised you or broke

I was surprised that the LU decompositions would stop working for such a low n. I was also surprised that the backwards error for the QR matrices stopped working in the solution step for all kinds of QR decompositions except for the rudimentary Gram-Schmidt. The low numerical accuracy kept the pivots above the 1e-14 threshold to be considered zeros and therefore allowed it to bypass the solution errors caused by the condition number of the Hilbert matrix.