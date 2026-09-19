# Markdown Note Template

## What was implemented

I implemented a numerical solver for 1D nonlinear BVPs. It used newton-raphson iteration to linearize the problem at each step, solve for the update via the linearization, then construct a new linear problem. Each update would ideally improve the residual until convergence was met. It was validated on 3 different nonlinear BVPs, each of which had different quirks.

## Validation

For all of the test cases I compared them against known analytic solutions. They are listed out below:

1. $u" + \lambda e^u = 0$, $u(\pm 1) = 0$ This has a known analytic solution in terms of hyperbolic cosines and natural logs, so I plotted against that solution.
2. $u" = 6u^2$, $u(0) = 1$, $u(1) = \frac{1}{4}$. This has a known analytic solution $u(x) = \frac{1}{x^2 + 1}$, so I plotted against that solution.
3. $(u^mu')' + 1 = 0$, $u(\pm 1) = 0$. This has a known analytical solution when $m=1$, which is $u(x) = \sqrt{1 - x^2}$, so I plotted against that solution.

## The interesting numerical behavior

The analytic solution for 1. depended on a parameter $\theta$, which required solving a nonlinear equation to find. This was interesting, as to find the analytic solution I had to make some numerical approximations. I also decided to switch to using sympy to compute the newton-raphson step here, as it was more reliable than doing all the calculations by hand.

It was interesting that the convergence for 2. depended on the choice of the initial guess. Of all of the guesses I tried, the linear interpolation between the two endpoints converged the fastest (in about 4 steps), while variations of that took closer to 8 or 9. In all of the cases, I noticed that if the problem was going to converge it usually only took fewer than 10 steps, after which a floor was hit in the precision. I was also surprised at how high that floor was - usually around 1e-10 to 1e-12 has been the norm for this project, but 1e-8 was more normal for the nonlinear validation.

## What surprised me or broke

I was surprised that the problem $(u^mu')' + 1 = 0$ with dirichlet boundary conditions broke. I expected it to be very easily solvable because of the dirichlet boundary conditions and the fact that the singularities only appeared at the edges. However, that last fact ended up being a huge problem regardless. Due to this property, the chebyshev polynomials were not able to achieve a reasonable level of precision relative to the analytic solution, implying that the analytic solution was not easily expressible in chebyshev polynomials. This was further exacerbated by the fact that the required step to improve approximation was increasing the number of chebyshev points, which in turn caused the conditioning of the second derivative matrix to grow past machine precision. Since the condition is $\mathcal{O}(k^4)$ where $k$ is the number of chebyshev points, this quickly became a problem. Ultimately those two factors made it so that although a solution theoretically existed for this problem, the technique I was hoping to use to find it was not functional.

In retrospect that should have been more clear. The code had already been sort of telling me that after I implemented a line search on the newton-raphson iteration. Originally my newton-raphson had found a two-cycle where it would bounce between two separate approximations for a solution but it wouldn't improve. Once I imposed the improvement condition on it via the line search, it never picked an improvement that had length of more than zero. It was interesting how this apparent bug could actually be pointing to something deeper.

I was also surprised at how simple it was to incorporate inhomogenous boundary conditions for the problem $u" = 6u^2$. I expected it to be relatively complex, but after thinking about it more I just tried to have an initial guess that fit the inhomogenous boundary conditions. My reasoning was since the basis functions were all homogenous, they couldn't change the inhomogenous conditions on the solution. It turns out that that was correct, so I generated my solution!