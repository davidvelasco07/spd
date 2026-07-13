# MHD tutorial

`soe="mhd"` couples the hydro solver to constrained-transport induction:

- fluid variables update with SD/FV fluxes (all 8 conserved variables, including
  cell-centered B),
- the in-plane face-staggered B updates with edge electric fields (CT), so
  div(B) = 0 to machine precision,
- the MHD-aware fallback limits *both*: MUSCL flux blending / MOOD cascade for the
  fluid and low-order edge-E levels for the induction -- the corrected B update is
  still a curl.

All hydro knobs apply. MHD-specific points:

- `vectorpot_fct` is required for a divergence-free initial B,
- `riemann_solver` (or `riemann_solver_ho` / `riemann_solver_lo`): `"llf"` or `"hlld"`,
- `scheme="FV"`: pure MUSCL(-Hancock) CT under RK (no ADER for MHD FV),
- variables: `rho, vx, vy, vz, P, Bx, By, Bz` (indices via `s.b["x"]`, ...).

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

```python
from spd.initial_conditions.initial_conditions_2d import orszag_tang, orszag_tang_Az

OT = dict(soe="mhd", init_fct=orszag_tang, vectorpot_fct=orszag_tang_Az,
          xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), gamma=5/3, cfl_coeff=0.4)

def divB_max(s):
    """Max |div B| of the face-staggered CT field (SD or FV layout)."""
    Bx, By = s.dm.asnumpy(s.dm.Bx_fp), s.dm.asnumpy(s.dm.By_fp)
    if Bx.ndim == 2:   # FV: staggered faces incl. Nghc ghost layers
        d = np.diff(Bx, axis=-1) / s.h["x"] + np.diff(By, axis=-2) / s.h["y"]
        g = s.Nghc
        return float(abs(d[g:-g, g:-g]).max())
    na = np.newaxis
    d = s.ho_scheme.compute_sp_from_dfp(s.dm.Bx_fp[na], "x")[0] / s.h["x"]
    d += s.ho_scheme.compute_sp_from_dfp(s.dm.By_fp[na], "y")[0] / s.h["y"]
    return float(abs(d).max())
```

## 1. Orszag-Tang vortex, SDFB + MOOD

The standard 2D MHD test: a smooth velocity/field vortex that develops shocks and a
central current sheet. Domain `[-1/2, 1/2]^2`, periodic; `rho = gamma^2`,
`P = gamma`, `B0 = 1` (code units). The pressure map at `t = 0.5` is the classic
figure to compare against the literature.

```python
N = 64
p = 3
s = SPD_Simulator(p=p, N=(N, N), scheme="SDFB", time_integrator="rk3",
                  blending=False,        # MOOD cascade (the MHD default)
                  riemann_solver="llf",
                  use_cupy=gpu, verbose=False, **OT)
s.perform_time_evolution(0.5)
print(f"max |div B| = {divB_max(s):.3e}")

W = s.dm.asnumpy(s.transpose_to_fv(s.regular_mesh(s.dm.W_cv)))
plt.figure(figsize=(6, 5))
plt.imshow(W[s._p_], origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="jet")
plt.colorbar(); plt.title(f"OT pressure, SDFB/MOOD p={p} N={N}, t=0.5");
```

## 2. All the fields

The primitive vector now includes the cell-centered magnetic field. (The
face-staggered `s.dm.Bx_fp` / `By_fp` are the divergence-free CT representation;
the cell-centered rows are their interpolation.)

```python
fig, axs = plt.subplots(1, 4, figsize=(19, 4))
for ax, (var, name) in zip(axs, [(0, r"$\rho$"), (4, "P"),
                                 (s.b["x"], "$B_x$"), (s.b["y"], "$B_y$")]):
    im = ax.imshow(W[var], origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="jet")
    fig.colorbar(im, ax=ax); ax.set_title(name)
```

## 3. Riemann solvers: LLF and HLLD

`"hlld"` (Miyoshi & Kusano 2005) resolves the full MHD wave fan and is markedly less
diffusive than `"llf"` -- but its integration here is **experimental**: on
Orszag-Tang it currently develops striping and drives the pressure to its floor
around `t ~ 0.2`. Use `"llf"` for production runs; the cell below only smoke-tests
the HLLD knob for a few steps.

```python
sr = SPD_Simulator(p=3, N=(32, 32), scheme="SDFB", time_integrator="rk3",
                   blending=False, riemann_solver="hlld",
                   use_cupy=gpu, verbose=False, **OT)
sr.perform_iterations(50)
W = sr.dm.asnumpy(sr.dm.W_cv)
print(f"hlld after {sr.n_step} steps: t={sr.time:.4f}, "
      f"minP={W[sr._p_].min():.3e}, max|divB|={divB_max(sr):.2e}")
```

## 4. Godunov mode and the FV CT scheme

- `godunov=True`: the MUSCL fallback flux (and level-1 edge E) is used *everywhere* --
  a robust ~2nd-order baseline on the SD subcell mesh.
- `scheme="FV"`: pure MUSCL / MUSCL-Hancock with face CT on the FV mesh, under RK
  (`time_integrator="rk1"` + `fallback="MUSCL-Hancock"` is the classic second-order
  CT scheme; `"rk2"` + `"MUSCL"` shown here).

```python
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))

sg = SPD_Simulator(p=3, N=(32, 32), scheme="SDFB", godunov=True,
                   time_integrator="rk3", use_cupy=gpu, verbose=False, **OT)
sg.perform_time_evolution(0.5)
Wg = sg.dm.asnumpy(sg.transpose_to_fv(sg.regular_mesh(sg.dm.W_cv)))
im = axs[0].imshow(Wg[sg._p_], origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="jet")
fig.colorbar(im, ax=axs[0]); axs[0].set_title("SDFB godunov=True (128 DOF)")

sf = SPD_Simulator(p=1, N=(128, 128), scheme="FV", fallback="MUSCL",
                   time_integrator="rk2", use_cupy=gpu, verbose=False, **OT)
sf.perform_time_evolution(0.5)
Wf = sf.dm.asnumpy(sf.transpose_to_fv(sf.dm.W_cv))
im = axs[1].imshow(Wf[sf._p_], origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="jet")
fig.colorbar(im, ax=axs[1]); axs[1].set_title("FV MUSCL CT, N=128")
print(f"divB: godunov {divB_max(sg):.2e}, FV {divB_max(sf):.2e}")
```

## Notes

- The MOOD cascade (`blending=False`, the MHD default) revises troubled cells down
  to MUSCL and then first order; cap the sweeps with `max_revs`.
  Neighbor-spreading theta-blending (`blending=True`) is a hydro feature -- for MHD
  the detection stays on cell-centered quantities.
- `limiting_variables` defaults to density and pressure; add the active B rows
  (e.g. `[0, 4, s.b["x"], s.b["y"]]`) for stricter control of magnetic oscillations.
- HLLD is experimental: on Orszag-Tang it currently develops grid-aligned striping
  and hits the pressure floor. If a run with `"hlld"` misbehaves, use `"llf"`.
- ADER time integration also works for `scheme="SD"`/`"SDFB"` MHD (it is the
  default); the RK path is what the recent benchmarks use.
