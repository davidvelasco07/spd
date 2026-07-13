# Induction tutorial

`soe="induction"` evolves *only* the magnetic field, with a prescribed (frozen)
velocity field:

$$ \partial_t \mathbf{B} = \nabla \times (\mathbf{v} \times \mathbf{B}) $$

using **constrained transport (CT)**: B lives on faces, the electric field on
edges, and the update is a discrete curl -- so div(B) = 0 holds to machine
precision *by construction*, forever.

Setup differs from hydro in two ways:

- `init_fct` provides rho, v, P (rho/P are only carried along; v drives the induction),
- `vectorpot_fct(mesh, j)` provides the vector potential **A**; the initial
  face-staggered B is its discrete curl (divergence-free from the start).

The scheme knob is the same `scheme` as everywhere else: `"SD"` (default),
`"FV"`, or `"FVFB"` (FV with
fallback, RK integrators). Both ADER and RK (`"rk1"`-`"rk5"`) time integration work.

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

## 1. Field-loop advection

The classic CT test (Gardiner & Stone 2005): a weak magnetic loop is advected
diagonally through the periodic box. A good scheme preserves the loop's shape;
div(B) must stay zero.

```python
def loop_flow(vx=1.0, vy=0.5):
    """Uniform background flow; rho = P = 1 (carried along, not evolved)."""
    def init_fct(mesh, var):
        shape = mesh[0].shape
        if var == 0 or var == 4:
            return np.ones(shape)
        if var == 1:
            return vx * np.ones(shape)
        if var == 2:
            return vy * np.ones(shape)
        return np.zeros(shape)
    return init_fct

def loop_Az(A0=1e-3, R=0.3):
    """Vector potential of a flat magnetic loop of radius R at the box center."""
    def vectorpot(mesh, j):
        if j == 2:
            x, y = mesh[0] - 0.5, mesh[1] - 0.5
            return A0 * np.maximum(R - np.sqrt(x**2 + y**2), 0.0)
        return np.zeros(mesh[0].shape)
    return vectorpot

def B2_image(s):
    """Cell-centered B^2 as a plain 2D image (handles SD and FV layouts)."""
    B2 = s.ho_scheme.compute_B2()
    if B2.ndim == 2:      # FV scheme: cell-centered with Nghc ghost layers
        g = s.Nghc
        return s.dm.asnumpy(B2)[g:-g, g:-g]
    return s.dm.asnumpy(s.ho_scheme.transpose_to_fv(B2)[0])
```

```python
s = SPD_Simulator(soe="induction", scheme="SD", p=3, N=(32, 32),
                  init_fct=loop_flow(), vectorpot_fct=loop_Az(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)

B2_0 = B2_image(s)
s.perform_time_evolution(2.0)   # two full crossings in x, one in y
B2_1 = B2_image(s)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, B2, t in ((axs[0], B2_0, 0.0), (axs[1], B2_1, 2.0)):
    im = ax.imshow(B2, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"$B^2$ at t={t}")
```

## 2. div(B) is zero by construction

The face-staggered divergence is a plain finite difference of the face fields --
check it directly:

```python
na = np.newaxis
ho = s.ho_scheme
divB = ho.compute_sp_from_dfp(s.dm.Bx_fp[na], "x")[0] / s.h["x"]
divB += ho.compute_sp_from_dfp(s.dm.By_fp[na], "y")[0] / s.h["y"]
print(f"max |div B| after {s.n_step} steps: {float(abs(divB).max()):.3e}")
print(f"total B^2: initial {B2_0.sum():.6e} -> final {B2_1.sum():.6e} "
      f"(numerical dissipation only)")
```

## 3. FV induction scheme

`scheme="FV"` swaps the SD induction scheme for a MUSCL-based cell-centered
scheme with face CT (one control volume per `N`, so raise `N` to compare at fixed
resolution). `"FVFB"` adds trouble detection + fallback on top (use RK).

```python
s_fv = SPD_Simulator(soe="induction", scheme="FV", p=1, N=(64, 64),
                     init_fct=loop_flow(), vectorpot_fct=loop_Az(),
                     time_integrator="rk2", use_cupy=gpu, verbose=False)
s_fv.perform_time_evolution(2.0)
B2_fv = B2_image(s_fv)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, B2, name in ((axs[0], B2_1, "SD p=3, N=32"), (axs[1], B2_fv, "FV, N=64")):
    im = ax.imshow(B2, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"$B^2$ at t=2, {name}")
```

## 4. Rotating flow

Anything you can write as `v(x, y)` works -- e.g. rigid rotation, which should spin
the loop without deforming it (until numerical diffusion smears it).

```python
def rotation(omega=2 * np.pi):
    def init_fct(mesh, var):
        x, y = mesh[0] - 0.5, mesh[1] - 0.5
        if var == 0 or var == 4:
            return np.ones(x.shape)
        if var == 1:
            return -omega * y
        if var == 2:
            return omega * x
        return np.zeros(x.shape)
    return init_fct

def off_center_Az(A0=1e-3, R=0.15):
    def vectorpot(mesh, j):
        if j == 2:
            x, y = mesh[0] - 0.7, mesh[1] - 0.5
            return A0 * np.maximum(R - np.sqrt(x**2 + y**2), 0.0)
        return np.zeros(mesh[0].shape)
    return vectorpot

s = SPD_Simulator(soe="induction", scheme="SD", p=3, N=(32, 32),
                  init_fct=rotation(), vectorpot_fct=off_center_Az(),
                  time_integrator="rk3", cfl_coeff=0.3, use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.5)   # half a rotation
B2 = B2_image(s)
plt.figure(figsize=(5.5, 4.5))
plt.imshow(B2, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
plt.colorbar(); plt.title("$B^2$ after half a rotation");
```

## Notes

- 3D works the same way: all three `vectorpot_fct(mesh, j)` components are used and
  B has three face-staggered components.
- The induction fallback (`scheme="FVFB"`, or `FB=True`) detects trouble from the
  variation of |B|^2 and blends the edge electric fields, so the corrected update is
  still a curl -- div(B) stays zero even in limited cells. The same machinery is what
  the MHD fallback uses (see the [MHD tutorial](mhd)).
- In this pure-induction mode the velocity field is frozen: the momentum equation is
  not evolved.
