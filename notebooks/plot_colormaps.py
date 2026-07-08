"""Density colormaps for the 2D step function and RTI on the current branch.

Sanity check of the layout-cleanup branch: runs both problems with the SDFB
scheme on GPU and saves initial/final density colormaps.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spd.initial_conditions as ic
import spd.utils.visualization as vsd
from spd.spd_simulator import SPD_Simulator

fig, axs = plt.subplots(1, 4, figsize=(19, 4.6),
                        gridspec_kw={"width_ratios": [1, 1, 0.45, 0.45]})

# ----------------------------------------------------------------------
# 1) Advected step function (periodic): at t=1 it must return to the IC.
# ----------------------------------------------------------------------
p = 3
s = SPD_Simulator(
    p=p, N=(128 // (p + 1),) * 2, use_cupy=True, FB=True,
    time_integrator="rk3", fallback="MUSCL", verbose=False,
    cfl_coeff=0.4,
    init_fct=ic.ic2d.step_function,
)
W0 = np.asarray(s.dm.W_cv).copy()

plt.sca(axs[0])
vsd.plot_field(s, W0, 0, plot_title="step function: density, t=0")
plt.gca().set_aspect("equal")

s.perform_time_evolution(1.0)

plt.sca(axs[1])
vsd.plot_field(s, s.dm.W_cv, 0,
               plot_title=f"step function: density, t={s.time:.2f} "
                          "(one period)")
plt.gca().set_aspect("equal")

# ----------------------------------------------------------------------
# 2) Rayleigh-Taylor instability (setup from notebooks/RTI.ipynb)
# ----------------------------------------------------------------------
NDOF = 192
N = NDOF // (p + 1)
s = SPD_Simulator(
    p=p,
    N=(N // 4, N),
    xlim=(0.0, 0.25),
    ylim=(0.0, 1.0),
    BC=(("periodic", "periodic"), ("reflective", "reflective")),
    init_fct=ic.RTI(P0=1, gamma=5 / 3),
    cfl_coeff=0.4,
    use_cupy=True,
    time_integrator="rk3",
    fallback="MUSCL",
    scheme="SDFB",
    potential=True,
    NAD="",
    PAD=True,
    SED=True,
    blending=False,
    riemann_solver_sd="hllc",
    riemann_solver_fv="hllc",
    limiting_variables=[0, 1, 2, 4],
    tolerance=1e-5,
    verbose=False,
)
W0 = np.asarray(s.dm.W_cv).copy()

plt.sca(axs[2])
vsd.plot_field(s, W0, 0, plot_title="RTI: density, t=0")
plt.gca().set_aspect("equal")

s.perform_time_evolution(1.95)

plt.sca(axs[3])
vsd.plot_field(s, s.dm.W_cv, 0,
               plot_title=f"RTI: density, t={s.time:.2f}")
plt.gca().set_aspect("equal")

plt.tight_layout()
plt.savefig("colormaps_layout_cleanup.png", dpi=130, bbox_inches="tight")
print("saved colormaps_layout_cleanup.png")
