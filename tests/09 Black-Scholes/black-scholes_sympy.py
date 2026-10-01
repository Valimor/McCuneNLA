import sympy as sp

S, tau, sigma, r, K, x = sp.symbols('S tau sigma r K x', positive=True)
V = sp.Function('V')

# original PDE, in terms of S
V_S = V(S, tau)
pde_original = (
    sp.diff(V_S, tau)
    - sp.Rational(1,2)*sigma**2*S**2*sp.diff(V_S, S, 2)
    - r*S*sp.diff(V_S, S)
    + r*V_S
)

# substitute S = exp(x), express everything in terms of a new function W(x, tau) = V(e^x, tau)
x_sym = sp.symbols('x')
W = sp.Function('W')
V_of_x = W(x_sym, tau)

dVdS = sp.diff(V_of_x, x_sym) / S
d2VdS2 = sp.diff(dVdS.subs(x_sym, sp.log(S)), S)  # careful chain -- easier to verify the OPERATOR identity directly:

# cleaner approach: verify the operator identity symbolically rather than
# substituting a generic function, since sympy handles this more reliably
f = sp.Function('f')(x_sym)
lhs = sp.diff(f, x_sym, 2) - sp.diff(f, x_sym)   # the claimed (d^2/dx^2 - d/dx) f
# check: does S^2 * d^2/dS^2 [f(ln S)] equal (d^2/dx^2 - d/dx) f(x), with x=ln(S)?
g = f.subs(x_sym, sp.log(S))
rhs = sp.simplify(S**2 * sp.diff(g, S, 2))
print(sp.simplify(rhs - lhs.subs(x_sym, sp.log(S))))   # should be 0