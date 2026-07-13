# spd

**spd** is a Python framework for compressible hydrodynamics and magnetohydrodynamics
(MHD) based on the **Spectral Difference (SD)** method with **a-posteriori limiting**.
High-order SD fluxes are computed on a sub-cell Gauss–Lobatto mesh; where the
solution is not smooth, a **trouble detector** flags cells and a robust
**finite-volume fallback** (MUSCL or MUSCL–Hancock, with optional theta-blending
or MOOD cascade) replaces or blends the high-order update. The same driver supports
pure FV runs, ADER or Runge–Kutta time integration, and optional **GPU**
execution through CuPy.

## Features

- **Physics:** ideal hydro (`soe="hydro"`), induction-only (`"induction"`), and
  full MHD (`"mhd"`) with constrained transport on staggered face fields.
- **Spatial schemes:** spectral difference (`SD`, `SDFB`) and finite volume
  (`FV`, `FVFB`); polynomial orders `p ≥ 1` in 1D/2D/3D.
- **Limiting:** NAD (numerical admissibility detection), PAD, smooth-extrema
  detection, neighbor spreading or MOOD revisions (`max_revs`), and a pure-Godunov
  mode for baseline comparisons.
- **Riemann solvers:** LLF, HLLC, and related closures; separate high- and
  low-order solvers (`riemann_solver_ho` / `riemann_solver_lo`).
- **Time integration:** ADER (default) or SSP-RK (`rk1`, `rk2`, `rk3`); RK path
  used for recent MHD/induction development.
- **GPU:** optional CuPy backend with fused kernels for flux blending and hydro
  updates (`pip install spd[gpu]`).
- **Initial conditions:** bundled 1D/2D/3D test problems (Sod, Kelvin–Helmholtz,
  Rayleigh–Taylor, Orszag–Tang, etc.) under `spd.initial_conditions`.

## Quick start

Install from the repository root (editable install recommended during
development):

```bash
python -m pip install -e .
# optional GPU extras (CUDA 12.x)
python -m pip install -e ".[gpu]"
```

Minimal 2D hydro run with the high-order + fallback scheme:

```python
import spd.initial_conditions as ic
from spd.spd_simulator import SPD_Simulator

s = SPD_Simulator(
    p=3,
    N=(32, 32),
    init_fct=ic.sod(),
    scheme="SDFB",
    time_integrator="rk3",
    use_cupy=False,
)
s.perform_time_evolution(0.2)
print(s.dm.W_cv.shape)
```

`SPD_Simulator` dispatches on `soe` to `HydroSimulator`, `InductionSimulator`, or
`MHDSimulator`; all mesh, boundary-condition, and limiting kwargs are forwarded
unchanged.

## Project layout

| Path | Role |
|------|------|
| `spd/simulator.py` | Base driver: mesh, BCs, CFL, time loop, I/O |
| `spd/spd_simulator.py` | Top-level factory (`soe` dispatch) |
| `spd/spectral_difference/` | SD spatial operators |
| `spd/finite_volume/` | FV / MUSCL reconstruction |
| `spd/fallback/` | Trouble detection and flux blending |
| `spd/integrators/` | ADER and RK time stepping |
| `spd/hydro/`, `spd/induction/`, `spd/MHD/` | Equation-specific physics and schemes |
| `spd/runtime/` | GPU data management and communication helpers |
| `spd/initial_conditions/` | Parametrised test-problem ICs |
| `test/` | Pytest suite (creation, transforms, GPU smoke tests) |
| `notebooks/Introduction.ipynb` | Tutorial notebook (tracked in git) |

See the architecture guide in the docs for extension points (new schemes,
integrators, or equation sets).

## Tests

```bash
python -m pytest test/
```

GPU tests are skipped automatically when CuPy is not available.

## Documentation

Documentation is built by the GitHub Actions workflow and deployed to GitHub
Pages:

https://davidvelasco07.github.io/spd/

The site covers the modular layout (simulator → scheme → integrator → physics),
configuration options, and the Python API reference.
