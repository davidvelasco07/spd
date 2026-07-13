"""Generate tutorial notebooks and MyST doc pages.

Run from ``notebooks/``:

- ``python _build_tutorials.py`` -- rebuild all ``*.ipynb`` sources
- ``python _build_tutorials.py --myst`` -- write ``docs/tutorials/*.ipynb`` for the docs site
- ``python _build_tutorials.py Hydro`` -- rebuild one notebook
"""
from pathlib import Path

import nbformat as nbf

KERNELSPEC = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}


DOCS_TUTORIALS = Path(__file__).resolve().parent.parent / "docs" / "tutorials"


def build(path, cells, *, tag_first_code=None):
    nb = nbf.v4.new_notebook()
    nb.metadata.update(KERNELSPEC)
    nb.cells = []
    first_code = True
    for kind, src in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(_myst_links(src)))
        else:
            cell = nbf.v4.new_code_cell(src)
            if first_code and tag_first_code:
                cell.metadata["tags"] = [tag_first_code]
            first_code = False
            nb.cells.append(cell)
    nbf.write(nb, path)
    print("wrote", path)


LIVE_CODE_BANNER = """\
```{thebe-button} Live Code
```

Starts a Binder kernel (CPU NumPy; no GPU). When the kernel is ready, click **Run** on each
code cell below. The first cell runs automatically to import `spd` and set up plotting."""


def write_doc_notebook(name, cells, out_dir=DOCS_TUTORIALS):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_cells = [("md", LIVE_CODE_BANNER)] + cells
    build(
        out_dir / f"{name.lower()}.ipynb",
        doc_cells,
        tag_first_code="thebe-init",
    )


def _myst_links(text):
    """Rewrite notebook cross-links for Sphinx MyST / nbsphinx pages."""
    return (
        text.replace(
            "[Introduction.ipynb](Introduction.ipynb)",
            "[Introduction notebook](../notebooks/Introduction)",
        )
        .replace(
            "[Hydro.ipynb](Hydro.ipynb)",
            "[Hydro tutorial](hydro)",
        )
        .replace(
            "[Induction.ipynb](Induction.ipynb)",
            "[Induction tutorial](induction)",
        )
        .replace(
            "[MHD.ipynb](MHD.ipynb)",
            "[MHD tutorial](mhd)",
        )
        .replace(
            "(see RTI in Hydro.ipynb)",
            "(see RTI in the [Hydro tutorial](hydro))",
        )
        .replace(
            "(see MHD.ipynb)",
            "(see the [MHD tutorial](mhd))",
        )
        .replace(
            "(MHD only, see MHD.ipynb)",
            "(MHD only; see the [MHD tutorial](mhd))",
        )
        .replace(
            "(see Introduction.ipynb section 3)",
            "(see the [Introduction notebook](../notebooks/Introduction), section 3)",
        )
    )


MYST_FRONTMATTER = """\
---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
"""


def cells_to_myst(cells):
    """Legacy MyST-markdown export (prefer write_doc_notebook for docs)."""
    parts = [
        MYST_FRONTMATTER,
        LIVE_CODE_BANNER + "\n",
    ]
    first_code = True
    for kind, src in cells:
        if kind == "md":
            parts.append(_myst_links(src))
        elif first_code:
            parts.append(
                f"```{{code-cell}} ipython3\n"
                f":tags: [thebe-init]\n\n"
                f"{src}\n```"
            )
            first_code = False
        else:
            parts.append(f"```{{code-cell}} ipython3\n{src}\n```")
    return "\n\n".join(parts) + "\n"


def write_myst(name, cells, out_dir=DOCS_TUTORIALS):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name.lower()}.md"
    path.write_text(cells_to_myst(cells), encoding="utf-8")
    print("wrote", path)


SETUP = '''\
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
print(f"backend: {'cupy (GPU)' if gpu else 'numpy (CPU)'}")'''


# ======================================================================
# Introduction.ipynb
# ======================================================================
intro = []
intro.append(("md", '''\
# SPD: Introduction and user guide

`spd` solves systems of conservation laws with a high-order **Spectral Difference (SD)**
scheme, stabilized where needed by a low-order **finite-volume (MUSCL) fallback**, on
CPU (NumPy) or GPU (CuPy).

The main axes of configuration are:

| Axis | Knob | Options |
|---|---|---|
| Equations | `soe` | `"hydro"`, `"induction"`, `"mhd"` |
| Spatial scheme | `scheme` | `"SD"`, `"SDFB"` (SD + fallback), `"FV"` |
| Space order | `p` | polynomial degree; the scheme is order `p+1` |
| Time integrator | `time_integrator` | `"ader"` (default), `"rk1"` ... `"rk5"` |
| Dimension | `N` | `(Nx,)`, `(Nx, Ny)`, `(Nx, Ny, Nz)` |
| Riemann solver | `riemann_solver` (or `_ho` / `_lo`) | `"llf"`, `"hllc"` (hydro), `"hlld"` (MHD) |
| Slope limiter | `slope_limiter` | `"minmod"`, `"moncen"` |
| Fallback reconstruction | `fallback` | `"MUSCL"`, `"MUSCL-Hancock"` |
| Limiting strategy | `blending`, `godunov`, `max_revs` | theta-blend / MOOD cascade / pure Godunov |

This notebook walks through every knob on small, fast problems.
The companion notebooks apply them:

- **[Hydro tutorial](../tutorials/hydro)** -- all bundled hydro initial conditions,
- **[Induction tutorial](../tutorials/induction)** -- constrained-transport induction solver,
- **[MHD tutorial](../tutorials/mhd)** -- coupled MHD (Orszag-Tang vortex).'''))
intro.append(("code", SETUP))
intro.append(("md", '''\
## 1. Entry point

`SPD_Simulator` is a factory: the `soe` keyword ("system of equations") dispatches to
`HydroSimulator` (default), `InductionSimulator`, or `MHDSimulator`, forwarding all
other arguments. You can also import those classes directly.

## 2. Quick start: Sod shock tube

Defaults: `soe="hydro"`, `scheme="SDFB"` (SD with fallback), `time_integrator="ader"`,
`p=1`. Initial conditions are callables `f(xyz, var)`; the bundled ones in
`spd.initial_conditions` are factories, so call them (optionally with parameters) to
get the function: `ic.sod()`, `ic.step_function(vx=2)`, ...

`N` counts SD *elements*; each element carries `(p+1)` solution points per dimension,
so the resolution in degrees of freedom is `N*(p+1)`.'''))
intro.append(("code", '''\
s = SPD_Simulator(
    p=3,                # degree 3 -> 4th order in space
    N=(32,),            # 32 elements x 4 solution points = 128 DOF
    init_fct=ic.sod(),
    BC=(("gradfree", "gradfree"),),   # outflow at both ends
    use_cupy=gpu,
)
s.perform_time_evolution(0.2)
vsd.plot_fields(s, s.dm.W_cv)   # primitive variables: rho, vx, vy, vz, P'''))
intro.append(("md", '''\
The solution lives in the data manager `s.dm`:

- `s.dm.W_cv` / `s.dm.U_cv` -- primitive / conserved control-volume averages,
- rows are indexed by `s.variables` (density `s._d_ = 0`, velocities `s.vels`,
  pressure `s._p_`).

`perform_time_evolution(t_end)` runs to a fixed time; `perform_iterations(n)` runs a
fixed number of steps.

## 3. `scheme`: SD, SDFB, FV

- `"SD"` -- pure high-order spectral difference. Crisp on smooth flows, oscillates at shocks.
- `"SDFB"` -- SD plus the MUSCL fallback: troubled cells are detected each step and
  their fluxes replaced/blended with a robust low-order scheme. **Recommended default.**
- `"FV"` -- pure second-order MUSCL / MUSCL-Hancock finite volume on `N` cells
  (use an RK integrator for FV).

The `FB` boolean is a convenience flag: `FB=False` strips the fallback from the scheme
name, `FB=True` adds it.'''))
intro.append(("code", '''\
fig, axs = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
runs = [("SD", "ader"), ("SDFB", "ader"), ("FV", "rk2")]
for ax, (scheme, ti) in zip(axs, runs):
    s = SPD_Simulator(p=3, N=(32,), scheme=scheme, time_integrator=ti,
                      init_fct=ic.sod(), BC=(("gradfree", "gradfree"),),
                      use_cupy=gpu, verbose=False)
    s.perform_time_evolution(0.2)
    W = s.transpose_to_fv(s.regular_mesh(s.dm.W_cv))
    x = s.regular_centers()[0] if scheme != "FV" else np.linspace(0, 1, W.shape[1])
    ax.plot(x, W[0], ".-", ms=3)
    ax.set_title(f"{scheme} ({ti})")
    ax.set_xlabel("x")
axs[0].set_ylabel(r"$\\rho$");'''))
intro.append(("md", '''\
Pure SD rings at the shock, SDFB stays clean at high order, FV is robust but more
diffusive.

## 4. `p` and `m`: space and time order

`p` is the polynomial degree per element (space order `p+1`). `m` is the time order;
by default `m = p`. For RK integrators the digit in the name sets the order
(`"rk3"` = 3 stages, up to `"rk5"`).

Compare orders at *fixed degrees of freedom* -- higher `p` on fewer elements resolves
smooth features better. At the deliberately coarse 32 DOF below, `p=1` visibly damps
the wave after one period while `p=3` and `p=7` sit on the exact curve:'''))
intro.append(("code", '''\
NDOF = 32
plt.figure(figsize=(7, 4))
for p in (1, 3, 7):
    N = NDOF // (p + 1)
    s = SPD_Simulator(p=p, N=(N,), init_fct=ic.sine_wave(A=0.5),
                      use_cupy=gpu, verbose=False)
    s.perform_time_evolution(1.0)   # one advection period
    W = s.transpose_to_fv(s.regular_mesh(s.dm.W_cv))
    plt.plot(s.regular_centers()[0], W[0], ".-", ms=4, label=f"p={p}, N={N}")
x = np.linspace(0, 1, 400)
plt.plot(x, 1 + 0.5*np.sin(2*np.pi*x), "k--", lw=1, label="exact")
plt.legend(); plt.xlabel("x"); plt.ylabel(r"$\\rho$")
plt.title(f"Advected sine wave, {NDOF} DOF, t=1");'''))
intro.append(("md", '''\
## 5. `time_integrator`: ADER or Runge-Kutta

- `"ader"` (default) -- single-step, arbitrary-order ADER predictor/corrector.
  The fallback then defaults to `fallback="MUSCL-Hancock"` (its half-step time
  prediction matches the ADER update).
- `"rk1"` ... `"rk5"` -- explicit method-of-lines RK. The fallback defaults to
  plain `fallback="MUSCL"` (a pure spatial RHS).

You can override the fallback reconstruction explicitly with
`fallback="MUSCL"` or `"MUSCL-Hancock"`.

Note: pure-FV hydro and MHD schemes want RK integrators (`"FV"` + ADER is not wired up).'''))
intro.append(("code", '''\
for ti in ("ader", "rk3"):
    s = SPD_Simulator(p=3, N=(32,), time_integrator=ti, init_fct=ic.sod(),
                      BC=(("gradfree", "gradfree"),), use_cupy=gpu, verbose=False)
    s.perform_time_evolution(0.2)
    print(f"{ti:5s}: fallback={s.lo_scheme.scheme!r:16s} steps={s.n_step}")'''))
intro.append(("md", '''\
## 6. Dimension

The length of `N` sets the dimension. Boundary conditions (`BC`), limits
(`xlim`/`ylim`/`zlim`) and the plotting helpers follow along.'''))
intro.append(("code", '''\
# 2D
s = SPD_Simulator(p=3, N=(32, 32), init_fct=ic.step_function(),
                  use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.5)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 0)'''))
intro.append(("code", '''\
# 3D (small, one step -- just to show the shape conventions)
s = SPD_Simulator(p=1, N=(16, 16, 16), init_fct=ic.step_function(),
                  use_cupy=gpu, verbose=False)
s.perform_iterations(5)
plt.figure(figsize=(5, 4))
vsd.plot_field(s, s.dm.W_cv, 0, dim="z")   # mid-plane slice; integrate=True averages'''))
intro.append(("md", '''\
## 7. Riemann solvers

`riemann_solver` sets the interface flux everywhere. For finer control there are
separate knobs for the two schemes -- when `riemann_solver` is given it supersedes
both:

- `riemann_solver_ho` -- flux of the primary (high-order) scheme,
- `riemann_solver_lo` -- flux of the low-order fallback scheme.

Options: `"llf"` (local Lax-Friedrichs, most diffusive, most robust), `"hllc"`
(hydro only), `"hlld"` (MHD only, see MHD.ipynb).'''))
intro.append(("code", '''\
plt.figure(figsize=(7, 4))
for rs in ("llf", "hllc"):
    s = SPD_Simulator(p=1, N=(64,), scheme="FV", time_integrator="rk2",
                      riemann_solver=rs, init_fct=ic.sod(),
                      BC=(("gradfree", "gradfree"),), use_cupy=gpu, verbose=False)
    s.perform_time_evolution(0.2)
    plt.plot(s.regular_centers()[0], s.transpose_to_fv(s.dm.W_cv)[0], label=rs)
plt.legend(); plt.xlabel("x"); plt.ylabel(r"$\\rho$")
plt.title("FV Sod: llf vs hllc (contact is sharper with hllc)");'''))
intro.append(("md", '''\
## 8. `slope_limiter`: minmod or moncen

Limits the MUSCL slopes of the FV/fallback reconstruction. `"minmod"` is the most
dissipative and safest; `"moncen"` (monotonized-central) is sharper.'''))
intro.append(("code", '''\
plt.figure(figsize=(7, 4))
for sl in ("minmod", "moncen"):
    s = SPD_Simulator(p=1, N=(64,), scheme="FV", time_integrator="rk2",
                      riemann_solver="hllc", slope_limiter=sl,
                      init_fct=ic.sod(), BC=(("gradfree", "gradfree"),),
                      use_cupy=gpu, verbose=False)
    s.perform_time_evolution(0.2)
    plt.plot(s.regular_centers()[0], s.transpose_to_fv(s.dm.W_cv)[0], label=sl)
plt.legend(); plt.xlabel("x"); plt.ylabel(r"$\\rho$"); plt.title("FV Sod: slope limiters");'''))
intro.append(("md", '''\
## 9. Trouble detection (the "FB" in SDFB)

After each candidate update the fallback checks every control volume:

- **NAD** (numerical admissibility): the new value must stay inside the local
  min/max bounds of the previous solution, relaxed by `tolerance`.
  - `NAD=""` -- relative tolerance; `NAD="delta"` -- tolerance scaled by the global range.
  - `NAD_neighbors="1st"` (face neighbors) or `"2nd"` (includes diagonals) sets the
    bounds stencil.
  - `SED=True` -- smooth-extrema detection: genuine smooth extrema are *not* flagged
    (prevents clipping of smooth peaks).
- **PAD** (physical admissibility): flags densities/pressures outside
  `[min_rho, max_rho]` / `min_P` (and NaNs).
- `limiting_variables` -- which rows are NAD-checked (default: density and pressure,
  `[0, 4]`). For more aggressive limiting include the velocities, e.g. `[0, 1, 2, 4]`.

The trouble mask of the last step is in `s.lo_scheme.dm.troubles`:'''))
intro.append(("code", '''\
s = SPD_Simulator(p=3, N=(32, 32), init_fct=ic.KH_instability(),
                  time_integrator="rk3", tolerance=1e-5, use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.8)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
plt.sca(axs[0]); vsd.plot_field(s, s.dm.W_cv, 0)
tr = s.dm.asnumpy(s.lo_scheme.dm.troubles)[0]
axs[1].imshow(tr, origin="lower", cmap="Reds", extent=(0, 1, 0, 1))
axs[1].set_title(f"troubled cells (last step): {int(tr.sum())}");'''))
intro.append(("md", '''\
## 10. Limiting strategy: blend, MOOD, or Godunov

What happens to flagged cells is set by two booleans:

| `blending` | `godunov` | behaviour |
|---|---|---|
| `True` (default) | `False` | **theta-blend**: one sweep; high- and low-order fluxes are convex-combined, and the trouble indicator is spread to neighbors with decaying weights (0.75 faces / 0.5 edges / 0.375 corners) |
| `False` | `False` | **MOOD cascade**: iterative detect -> recompute; flagged cells drop to MUSCL, then to first order. `max_revs` caps the number of revision sweeps (the "revisions" knob) |
| -- | `True` | **pure Godunov**: the fallback flux is used everywhere (robust baseline, order ~2) |'''))
intro.append(("code", '''\
fig, axs = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
configs = [
    ("theta-blend", dict(blending=True)),
    ("MOOD, 2 revisions", dict(blending=False, max_revs=2)),
    ("Godunov", dict(godunov=True)),
]
for ax, (name, kw) in zip(axs, configs):
    s = SPD_Simulator(p=3, N=(32,), init_fct=ic.sod(),
                      BC=(("gradfree", "gradfree"),),
                      time_integrator="rk3", use_cupy=gpu, verbose=False, **kw)
    s.perform_time_evolution(0.2)
    W = s.transpose_to_fv(s.regular_mesh(s.dm.W_cv))
    ax.plot(s.regular_centers()[0], W[0], ".-", ms=3)
    ax.set_title(name); ax.set_xlabel("x")
axs[0].set_ylabel(r"$\\rho$");'''))
intro.append(("md", '''\
## 11. Boundary conditions

`BC` is a per-dimension tuple of `(left, right)` strings:

- `"periodic"` (default)
- `"reflective"` -- wall (normal velocity flipped)
- `"gradfree"` -- zero-gradient outflow
- `"doublemach"` -- special time-dependent BCs of the Double Mach Reflection test

Domain extents are `xlim`, `ylim`, `zlim` (default `(0, 1)` each).

The Liska-Wendroff implosion is the classic workout for reflective walls: a
low-pressure corner triangle launches a shock that keeps bouncing around the box.
Because the setup is symmetric about the diagonal, any asymmetry the scheme (or the
BCs) introduces is immediately visible.'''))
intro.append(("code", '''\
# Liska-Wendroff implosion: reflective walls on all four sides
s = SPD_Simulator(p=3, N=(32, 32), init_fct=ic.implosion(),
                  xlim=(0, 0.3), ylim=(0, 0.3), gamma=1.4,
                  BC=(("reflective", "reflective"), ("reflective", "reflective")),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.5)
plt.figure(figsize=(5, 4.2)); vsd.plot_field(s, s.dm.W_cv, 0)
plt.gca().set_aspect("equal")

W = s.dm.asnumpy(s.transpose_to_fv(s.regular_mesh(s.dm.W_cv)))
print(f"diagonal symmetry: max|rho - rho^T| = {np.abs(W[0] - W[0].T).max():.2e}")'''))
intro.append(("md", '''\
## 12. Physics and numerics parameters

- `gamma` -- adiabatic index (default 1.4).
- `cfl_coeff` -- CFL number (default 0.4; for `scheme="FV"` it is divided by `p+1`
  internally).
- `min_rho`, `min_c2` -- floors used in primitive conversions.
- `passives` -- list of passive-scalar names, advected with the flow and appended
  to the variable vector.
- `potential=True` -- static gravity; the potential is sampled from `init_fct(xyz, -1)`
  (see RTI in Hydro.ipynb).
- `WB=True` -- well-balanced mode: evolves the perturbation around an equilibrium
  given by `eq_fct` (hydrostatic setups stay exactly steady).
- `viscosity` / `nu` and `thdiffusion` / `chi` -- Navier-Stokes viscous flux and
  thermal diffusion terms (this code path is currently being reworked -- expect
  rough edges).'''))
intro.append(("code", '''\
# A passive scalar tracing the initial top/bottom split of a KH run
def kh_with_dye(xyz, var):
    if var == 5:
        return np.where(np.abs(xyz[1] - 0.5) < 0.25, 1.0, 0.0)
    return ic.KH_instability()(xyz, var)

s = SPD_Simulator(p=3, N=(32, 32), init_fct=kh_with_dye, passives=["dye"],
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.8)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 5)'''))
intro.append(("md", '''\
## 13. GPU execution

`use_cupy=True` keeps every array on the GPU for the whole run (CuPy); results are
copied back to host at the end. If CuPy is not installed the flag silently falls
back to NumPy. `verbose=False` suppresses the per-100-step progress line.

The final printout reports wall time and throughput (`bzcps` = billions of
zone-cycles per second).'''))
intro.append(("code", '''\
if gpu:
    s = SPD_Simulator(p=3, N=(64, 64), init_fct=ic.step_function(),
                      time_integrator="rk3", use_cupy=True, verbose=False)
    s.perform_time_evolution(0.2)   # prints steps, wall time, bzcps'''))
intro.append(("md", '''\
## 14. Outputs and visualization

- `s.output()` writes `W_cv` snapshots to `folder` (default `outputs/`);
  `s.load_output()` restores the latest one.
- `vsd.plot_field(s, M, var)` plots one variable (2D pcolormesh / 1D line;
  for 3D pass `dim=` for the slice normal or `integrate=True`).
- `vsd.plot_fields(s, M)` plots all variables side by side.
- `s.regular_mesh(W)` interpolates SD solution points onto a regular grid, and
  `s.transpose_to_fv(M)` flattens the element layout `(nvar, Ny, Nx, ny, nx)` to a
  plain image `(nvar, Ny*ny, Nx*nx)` -- useful for custom plotting, as in the cells
  above.

## Where next

- **[Hydro tutorial](../tutorials/hydro)** -- Sod, sine wave (convergence), step function,
  Kelvin-Helmholtz, Rayleigh-Taylor with gravity, Double Mach Reflection.
- **[Induction tutorial](../tutorials/induction)** -- the constrained-transport induction
  solver: field-loop advection, SD vs FV, div(B) = 0.
- **[MHD tutorial](../tutorials/mhd)** -- full MHD: Orszag-Tang with the SDFB/MOOD scheme,
  HLLD, and the FV CT scheme.'''))


# ======================================================================
# Hydro.ipynb
# ======================================================================
hydro = []
hydro.append(("md", '''\
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

See [Introduction.ipynb](Introduction.ipynb) for what each knob does.'''))
hydro.append(("code", SETUP))
hydro.append(("md", '''\
## 1. Sod shock tube (1D)

The standard shock-tube: left state (1, 0, 1), right state (0.125, 0, 0.1),
outflow boundaries.'''))
hydro.append(("code", '''\
s = SPD_Simulator(p=3, N=(64,), init_fct=ic.sod(),
                  BC=(("gradfree", "gradfree"),), time_integrator="rk3",
                  riemann_solver="hllc", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.2)
vsd.plot_fields(s, s.dm.W_cv)'''))
hydro.append(("md", '''\
## 2. Sine wave: convergence of the SD scheme

A smooth density wave advected for one period returns to the initial condition, so
the L1 error measures the order of the scheme. Expect ~`(p+1)`-order convergence
until the fallback or round-off interferes.'''))
hydro.append(("code", '''\
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
plt.legend();'''))
hydro.append(("md", '''\
## 3. Step function: advected square wave

A discontinuous profile advected diagonally across the periodic box -- the classic
stress test for the limiter. After one period it should return to the initial
square, as sharp as possible and without over/undershoots.'''))
hydro.append(("code", '''\
# 1D
s = SPD_Simulator(p=3, N=(32,), init_fct=ic.step_function(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(7, 4))
W = s.transpose_to_fv(s.regular_mesh(s.dm.W_cv))
plt.plot(s.regular_centers()[0], W[0], ".-")
plt.xlabel("x"); plt.ylabel(r"$\\rho$"); plt.title("1D square wave after 1 period");'''))
hydro.append(("code", '''\
# 2D, advected diagonally (vx=vy=1)
s = SPD_Simulator(p=3, N=(32, 32), init_fct=ic.step_function(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 0)'''))
hydro.append(("code", '''\
# 3D (kept small; mid-plane slice)
s = SPD_Simulator(p=1, N=(24, 24, 24), init_fct=ic.step_function(),
                  time_integrator="rk2", use_cupy=gpu, verbose=False)
s.perform_time_evolution(0.25)
plt.figure(figsize=(5, 4)); vsd.plot_field(s, s.dm.W_cv, 0, dim="z")'''))
hydro.append(("md", '''\
## 4. Kelvin-Helmholtz instability (2D)

A double shear layer with a seeded sinusoidal perturbation; both directions periodic
(the defaults). High order pays off here: the billows stay crisp.'''))
hydro.append(("code", '''\
s = SPD_Simulator(p=3, N=(64, 64), init_fct=ic.KH_instability(),
                  time_integrator="rk3", riemann_solver="hllc",
                  use_cupy=gpu, verbose=False)
s.perform_time_evolution(1.0)
plt.figure(figsize=(5.5, 4.5)); vsd.plot_field(s, s.dm.W_cv, 0, cmap="RdBu_r")'''))
hydro.append(("md", '''\
## 5. Implosion (2D, Liska & Wendroff 2003)

A low-density, low-pressure triangle below the diagonal `x + y = 0.15` in a
reflective box `[0, 0.3]^2`. The imploding shock reflects off the origin and, by
`t = 2.5`, a jet has propagated along the diagonal. The problem is mirror-symmetric
about `x = y`, so the transpose-asymmetry of the density is an exact measure of how
well the scheme preserves symmetry -- watch it stay at round-off.

(This is the longest run in the notebook: ~18k steps, a few minutes on a GPU.)'''))
hydro.append(("code", '''\
s = SPD_Simulator(p=3, N=(100, 100), init_fct=ic.implosion(),
                  xlim=(0, 0.3), ylim=(0, 0.3), gamma=1.4,
                  BC=(("reflective", "reflective"), ("reflective", "reflective")),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)
s.perform_time_evolution(2.5)

W = s.dm.asnumpy(s.transpose_to_fv(s.regular_mesh(s.dm.W_cv)))
print(f"diagonal symmetry: max|rho - rho^T| = {np.abs(W[0] - W[0].T).max():.2e}")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
for ax, var, name in ((axs[0], 0, r"$\\rho$"), (axs[1], 4, "P")):
    im = ax.imshow(W[var], origin="lower", extent=(0, 0.3, 0, 0.3), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"{name} at t=2.5"); ax.set_aspect("equal")'''))
hydro.append(("md", '''\
## 6. Rayleigh-Taylor instability (2D, gravity)

Heavy fluid on top of light fluid in a constant gravitational field. This exercises

- `potential=True`: static gravity, with the potential sampled from
  `init_fct(xyz, -1)` (the `RTI` factory provides it),
- `reflective` walls in y,
- `limiting_variables=[0, 1, 2, 4]` (include velocities in the NAD check).'''))
hydro.append(("code", '''\
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
plt.gca().set_aspect("equal")'''))
hydro.append(("md", '''\
## 7. Double Mach Reflection (2D)

A Mach-10 shock hitting a reflecting wall at 60 degrees (Woodward & Colella 1984).
Uses the dedicated `"doublemach"` boundary conditions, which impose the exact moving
shock trace on the top boundary. Domain `[0, 4] x [0, 1]`.'''))
hydro.append(("code", '''\
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
plt.gca().set_aspect("equal"); plt.xlim(0, 3)'''))
hydro.append(("md", '''\
## Notes

- All of these runs use the SDFB scheme; swap `scheme="FV"` (+ RK integrator) or
  `FB=False` to compare (see Introduction.ipynb section 3).
- For a well-balanced hydrostatic run, pass the same `RTI` factory as `eq_fct` with
  `WB=True`: the unperturbed atmosphere then stays steady to machine precision.'''))


# ======================================================================
# Induction.ipynb
# ======================================================================
induction = []
induction.append(("md", '''\
# Induction tutorial

`soe="induction"` evolves *only* the magnetic field, with a prescribed (frozen)
velocity field:

$$ \\partial_t \\mathbf{B} = \\nabla \\times (\\mathbf{v} \\times \\mathbf{B}) $$

using **constrained transport (CT)**: B lives on faces, the electric field on
edges, and the update is a discrete curl -- so div(B) = 0 holds to machine
precision *by construction*, forever.

Setup differs from hydro in two ways:

- `init_fct` provides rho, v, P (rho/P are only carried along; v drives the induction),
- `vectorpot_fct(mesh, j)` provides the vector potential **A**; the initial
  face-staggered B is its discrete curl (divergence-free from the start).

The scheme knob is the same `scheme` as everywhere else: `"SD"` (default),
`"FV"`, or `"FVFB"` (FV with
fallback, RK integrators). Both ADER and RK (`"rk1"`-`"rk5"`) time integration work.'''))
induction.append(("code", SETUP))
induction.append(("md", '''\
## 1. Field-loop advection

The classic CT test (Gardiner & Stone 2005): a weak magnetic loop is advected
diagonally through the periodic box. A good scheme preserves the loop's shape;
div(B) must stay zero.'''))
induction.append(("code", '''\
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
    return s.dm.asnumpy(s.ho_scheme.transpose_to_fv(B2)[0])'''))
induction.append(("code", '''\
s = SPD_Simulator(soe="induction", scheme="SD", p=3, N=(32, 32),
                  init_fct=loop_flow(), vectorpot_fct=loop_Az(),
                  time_integrator="rk3", use_cupy=gpu, verbose=False)

B2_0 = B2_image(s)
s.perform_time_evolution(2.0)   # two full crossings in x, one in y
B2_1 = B2_image(s)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, B2, t in ((axs[0], B2_0, 0.0), (axs[1], B2_1, 2.0)):
    im = ax.imshow(B2, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"$B^2$ at t={t}")'''))
induction.append(("md", '''\
## 2. div(B) is zero by construction

The face-staggered divergence is a plain finite difference of the face fields --
check it directly:'''))
induction.append(("code", '''\
na = np.newaxis
ho = s.ho_scheme
divB = ho.compute_sp_from_dfp(s.dm.Bx_fp[na], "x")[0] / s.h["x"]
divB += ho.compute_sp_from_dfp(s.dm.By_fp[na], "y")[0] / s.h["y"]
print(f"max |div B| after {s.n_step} steps: {float(abs(divB).max()):.3e}")
print(f"total B^2: initial {B2_0.sum():.6e} -> final {B2_1.sum():.6e} "
      f"(numerical dissipation only)")'''))
induction.append(("md", '''\
## 3. FV induction scheme

`scheme="FV"` swaps the SD induction scheme for a MUSCL-based cell-centered
scheme with face CT (one control volume per `N`, so raise `N` to compare at fixed
resolution). `"FVFB"` adds trouble detection + fallback on top (use RK).'''))
induction.append(("code", '''\
s_fv = SPD_Simulator(soe="induction", scheme="FV", p=1, N=(64, 64),
                     init_fct=loop_flow(), vectorpot_fct=loop_Az(),
                     time_integrator="rk2", use_cupy=gpu, verbose=False)
s_fv.perform_time_evolution(2.0)
B2_fv = B2_image(s_fv)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, B2, name in ((axs[0], B2_1, "SD p=3, N=32"), (axs[1], B2_fv, "FV, N=64")):
    im = ax.imshow(B2, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(f"$B^2$ at t=2, {name}")'''))
induction.append(("md", '''\
## 4. Rotating flow

Anything you can write as `v(x, y)` works -- e.g. rigid rotation, which should spin
the loop without deforming it (until numerical diffusion smears it).'''))
induction.append(("code", '''\
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
plt.colorbar(); plt.title("$B^2$ after half a rotation");'''))
induction.append(("md", '''\
## Notes

- 3D works the same way: all three `vectorpot_fct(mesh, j)` components are used and
  B has three face-staggered components.
- The induction fallback (`scheme="FVFB"`, or `FB=True`) detects trouble from the
  variation of |B|^2 and blends the edge electric fields, so the corrected update is
  still a curl -- div(B) stays zero even in limited cells. The same machinery is what
  the MHD fallback uses (see MHD.ipynb).
- In this pure-induction mode the velocity field is frozen: the momentum equation is
  not evolved.'''))


# ======================================================================
# MHD.ipynb
# ======================================================================
mhd = []
mhd.append(("md", '''\
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
- variables: `rho, vx, vy, vz, P, Bx, By, Bz` (indices via `s.b["x"]`, ...).'''))
mhd.append(("code", SETUP))
mhd.append(("code", '''\
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
    return float(abs(d).max())'''))
mhd.append(("md", '''\
## 1. Orszag-Tang vortex, SDFB + MOOD

The standard 2D MHD test: a smooth velocity/field vortex that develops shocks and a
central current sheet. Domain `[-1/2, 1/2]^2`, periodic; `rho = gamma^2`,
`P = gamma`, `B0 = 1` (code units). The pressure map at `t = 0.5` is the classic
figure to compare against the literature.'''))
mhd.append(("code", '''\
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
plt.colorbar(); plt.title(f"OT pressure, SDFB/MOOD p={p} N={N}, t=0.5");'''))
mhd.append(("md", '''\
## 2. All the fields

The primitive vector now includes the cell-centered magnetic field. (The
face-staggered `s.dm.Bx_fp` / `By_fp` are the divergence-free CT representation;
the cell-centered rows are their interpolation.)'''))
mhd.append(("code", '''\
fig, axs = plt.subplots(1, 4, figsize=(19, 4))
for ax, (var, name) in zip(axs, [(0, r"$\\rho$"), (4, "P"),
                                 (s.b["x"], "$B_x$"), (s.b["y"], "$B_y$")]):
    im = ax.imshow(W[var], origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="jet")
    fig.colorbar(im, ax=ax); ax.set_title(name)'''))
mhd.append(("md", '''\
## 3. Riemann solvers: LLF and HLLD

`"hlld"` (Miyoshi & Kusano 2005) resolves the full MHD wave fan and is markedly less
diffusive than `"llf"` -- but its integration here is **experimental**: on
Orszag-Tang it currently develops striping and drives the pressure to its floor
around `t ~ 0.2`. Use `"llf"` for production runs; the cell below only smoke-tests
the HLLD knob for a few steps.'''))
mhd.append(("code", '''\
sr = SPD_Simulator(p=3, N=(32, 32), scheme="SDFB", time_integrator="rk3",
                   blending=False, riemann_solver="hlld",
                   use_cupy=gpu, verbose=False, **OT)
sr.perform_iterations(50)
W = sr.dm.asnumpy(sr.dm.W_cv)
print(f"hlld after {sr.n_step} steps: t={sr.time:.4f}, "
      f"minP={W[sr._p_].min():.3e}, max|divB|={divB_max(sr):.2e}")'''))
mhd.append(("md", '''\
## 4. Godunov mode and the FV CT scheme

- `godunov=True`: the MUSCL fallback flux (and level-1 edge E) is used *everywhere* --
  a robust ~2nd-order baseline on the SD subcell mesh.
- `scheme="FV"`: pure MUSCL / MUSCL-Hancock with face CT on the FV mesh, under RK
  (`time_integrator="rk1"` + `fallback="MUSCL-Hancock"` is the classic second-order
  CT scheme; `"rk2"` + `"MUSCL"` shown here).'''))
mhd.append(("code", '''\
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
print(f"divB: godunov {divB_max(sg):.2e}, FV {divB_max(sf):.2e}")'''))
mhd.append(("md", '''\
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
  default); the RK path is what the recent benchmarks use.'''))


import sys

targets = {"Introduction": intro, "Hydro": hydro,
           "Induction": induction, "MHD": mhd}
MYST_TARGETS = ("Hydro", "Induction", "MHD")
args = sys.argv[1:]
myst_only = "--myst" in args
if myst_only:
    args = [a for a in args if a != "--myst"]

if myst_only:
    for name in MYST_TARGETS:
        write_doc_notebook(name, targets[name])
else:
    selected = args or list(targets)
    for name in selected:
        if name in MYST_TARGETS:
            write_doc_notebook(name, targets[name])
        if name in targets:
            build(f"{name}.ipynb", targets[name])
