# Hydro tutorial

Every hydro initial condition bundled in `spd.initial_conditions`, run with sensible
configurations:

| IC | dim | notes |
|---|---|---|
| `sod()` | 1D | shock tube (shock, contact, rarefaction) |
| `sine_wave(A, vx, ...)` | 1/2/3D | smooth advection -- used for a convergence test |
| `step_function(vx, ...)` | 1/2/3D | advected square wave |
| `KH_instability()` | 2D | shear layer, double periodic |
| `implosion(...)` | 2D | Liska-Wendroff implosion, reflective box, symmetry test |
| `RTI(P0, gamma, ...)` | 2D | Rayleigh-Taylor with gravity (`potential=True`) |
| `double_mach_reflection()` | 2D | Mach-10 shock on a wall, special BCs |

See [Introduction notebook](../notebooks/Introduction) for what each knob does.

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import spd
import spd.utils.visualization as vsd
import spd.initial_conditions as ic
from spd.spd_simulator import SPD_Simulator

try:
    import cupy as cp
    gpu = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    gpu = False

np.seterr(all="ignore");
mpl.rcParams["font.size"] = 13
print(f"backend: {'cupy (GPU)' if gpu else 'numpy (CPU)'}")
```

## 1. Sod shock tube (1D)

The standard shock-tube: left state (1, 0, 1), right state (0.125, 0, 0.1),
outflow boundaries.

```python
s = SPD_Simulator(p=3, N=(64,), init_fct=ic.sod(),
                  BC=(("gradfree", "gradfree"),), time_integrator="rk3",
                  riemann_solver="hllc", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.2)
vsd.plot_fields(s, s.dm.W_cv)
```

## 2. Sine wave: convergence of the SD scheme

A smooth density wave advected for one period returns to the initial condition, so
the L1 error measures the order of the scheme. Expect ~`(p+1)`-order convergence
until the fallback or round-off interferes.

```python
def l1_error(p, N):
    s = SPD_Simulator(p=p, N=(N,), init_fct=ic.sine_wave(A=0.125),
                      FB=False, use_cupy=gpu, verbose=False)
    W0 = np.array(s.dm.asnumpy(s.dm.W_cv[0]).copy())
    s.perform_time_evolution(1.0)
    return float(np.mean(np.abs(s.dm.asnumpy(s.dm.W_cv[0]) - W0)))

plt.figure(figsize=(7, 4.5))
Ns = np.array([4, 8, 16, 32])
for p in (1, 2, 3):
    errs = [l1_error(p, int(N)) for N in Ns]
    dof = Ns * (p + 1)
    plt.loglog(dof, errs, "o-", label=f"p={p}")
    plt.loglog(dof, errs[0] * (dof / dof[0]) ** -(p + 1.0), "k:", lw=0.8)
plt.xlabel("degrees of freedom"); plt.ylabel("L1 error")
plt.title("Advected sine wave, 1 period (dotted: ideal order p+1)")
plt.legend();
```

## 3. Step function: advected square wave

A discontinuous profile advected diagonally across the periodic box -- the classic
stress test for the limiter. After one period it should return to the initial
square, as sharp as possible and without over/undershoots.

```python
# 1D
s = SPD_Simulator(p=3, N=(32,), init_fct=ic.step_function(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(7, 4))
W = s.transpose_to_fv(s.regular_mesh(s.dm.W_cv))
plt.plot(s.regular_centers()[0], W[0], ".-")
plt.xlabel("x"); plt.ylabel(r"$\rho$"); plt.title("1D square wave after 1 period");
```

```python
# 2D, advected diagonally (vx=vy=1)
s = SPD_Simulator(p=3, N=(32, 32), init_fct=ic.step_function(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 0)
```

```python
# 3D (kept small; mid-plane slice)
s = SPD_Simulator(p=1, N=(24, 24, 24), init_fct=ic.step_function(),
                  time_integrator="rk2", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.25)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 0, dim="z")
```

## 4. Kelvin-Helmholtz instability (2D)

A double shear layer with a seeded sinusoidal perturbation; both directions periodic
(the defaults). High order pays off here: the billows stay crisp.

```python
s = SPD_Simulator(p=3, N=(64, 64), init_fct=ic.KH_instability(),
                  time_integrator="rk3", riemann_solver="hllc",
                  use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(5.5, 4.5)); vsd.plot_field(s, s.dm.W_cv, 0, cmap="RdBu_r")
```

## 5. Implosion (2D, Liska & Wendroff 2003)

A low-density, low-pressure triangle below the diagonal `x + y = 0.15` in a
reflective box `[0, 0.3]^2`. The imploding shock reflects off the origin and, by
`t = 2.5`, a jet has propagated along the diagonal. The problem is mirror-symmetric
about `x = y`, so the transpose-asymmetry of the density is an exact measure of how
well the scheme preserves symmetry -- watch it stay at round-off.

(This is the longest run in the notebook: ~18k steps, a few minutes on a GPU.)

```python
s = SPD_Simulator(p=3, N=(100, 100), init_fct=ic.implosion(),
                  xlim=(0, 0.3), ylim=(0, 0.3), gamma=1.4,
                  BC=(("reflective", "reflective"), ("reflective", "reflective")),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(2.5)

W = s.dm.asnumpy(s.transpose_to_fv(s.regular_mesh(s.dm.W_cv)))
print(f"diagonal symmetry: max|rho - rho^T| = {np.abs(W[0] - W[0].T).max():.2e}")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
for ax, var, name in ((axs[0], 0, r"$\rho$"), (axs[1], 4, "P")):
    im = ax.imshow(W[var], origin="lower", extent=(0, 0.3, 0, 0.3), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"{name} at t=2.5"); ax.set_aspect("equal")
```

## 6. Rayleigh-Taylor instability (2D, gravity)

Heavy fluid on top of light fluid in a constant gravitational field. This exercises

- `potential=True`: static gravity, with the potential sampled from
  `init_fct(xyz, -1)` (the `RTI` factory provides it),
- `reflective` walls in y,
- `limiting_variables=[0, 1, 2, 4]` (include velocities in the NAD check).

```python
NDOF = 256
p = 3
N = NDOF // (p + 1)
s = SPD_Simulator(p=p, N=(N // 4, N), xlim=(0.0, 0.25), ylim=(0.0, 1.0),
                  BC=(("periodic", "periodic"), ("reflective", "reflective")),
                  init_fct=ic.RTI(P0=1, gamma=5/3), gamma=5/3,
                  potential=True, blending=False,
                  time_integrator="rk3", fallback="MUSCL",
                  limiting_variables=[0, 1, 2, 4], tolerance=1e-5,
                  riemann_solver="hllc",
                  cfl_coeff=0.4, use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.8)
plt.figure(figsize=(3.2, 8)); vsd.plot_field(s, s.dm.W_cv, 0)
plt.gca().set_aspect("equal")
```

## 7. Double Mach Reflection (2D)

A Mach-10 shock hitting a reflecting wall at 60 degrees (Woodward & Colella 1984).
Uses the dedicated `"doublemach"` boundary conditions, which impose the exact moving
shock trace on the top boundary. Domain `[0, 4] x [0, 1]`.

```python
NDOF = 400
p = 3
N = NDOF // (p + 1)
s = SPD_Simulator(p=p, N=(N, N // 4), xlim=(0.0, 4.0), ylim=(0.0, 1.0),
                  BC=(("doublemach", "doublemach"), ("doublemach", "doublemach")),
                  init_fct=ic.double_mach_reflection(),
                  time_integrator="rk3", fallback="MUSCL", blending=False,
                  limiting_variables=[0, 1, 2, 4], tolerance=1e-5,
                  riemann_solver="hllc",
                  cfl_coeff=0.4, use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.2)
plt.figure(figsize=(12, 3.2)); vsd.plot_field(s, s.dm.W_cv, 0)
plt.gca().set_aspect("equal"); plt.xlim(0, 3)
```

## Notes

- All of these runs use the SDFB scheme; swap `scheme="FV"` (+ RK integrator) or
  `FB=False` to compare (see Introduction.ipynb section 3).
- For a well-balanced hydrostatic run, pass the same `RTI` factory as `eq_fct` with
  `WB=True`: the unperturbed atmosphere then stays steady to machine precision.
