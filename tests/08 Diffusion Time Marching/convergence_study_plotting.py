import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./tests/08 Diffusion Time Marching/outputs/convergence_vs_dt.txt", skiprows=1)
dts, error_rk4, error_ie, error_cn = data.T

def mask_valid(dts, errors, ceiling=1e10):
    valid = np.isfinite(errors) & (errors < ceiling)
    return dts[valid], errors[valid]

dts_rk4, error_rk4_clean = mask_valid(dts, error_rk4)
dts_ie, error_ie_clean = mask_valid(dts, error_ie)
dts_cn, error_cn_clean = mask_valid(dts, error_cn)

if len(dts_rk4) < len(dts):
    breakdown_dt = dts[len(dts_rk4)]  # first dt where RK4 failed, assuming dts is sorted ascending
    plt.axvline(breakdown_dt, color="gray", linestyle=":", alpha=0.6, label=f"RK4 instability onset ($\\Delta t\\approx${breakdown_dt:.1e})")

plt.loglog(dts_rk4, error_rk4_clean, "o-", label="RK4")
plt.loglog(dts_ie, error_ie_clean, "s-", label="Implicit Euler")
plt.loglog(dts_cn, error_cn_clean, "^-", label="Crank-Nicolson")
plt.legend()
plt.title("Error vs. dt")
plt.xlabel("dt")
plt.ylabel("Final error")
plt.show()