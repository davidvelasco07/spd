import sys
import numpy as np
import pytest

sys.path.append("../src")

from amr.tree import FINER
from sdader_simulator import SDADER_Simulator
from initial_conditions_3d import step_function


LEVELS_2D = [{"level": 1, "xmin": 0.52, "xmax": 0.74, "ymin": 0.52, "ymax": 0.74}]
LEVELS_3D = [{
    "level": 1,
    "xmin": 0.52,
    "xmax": 0.74,
    "ymin": 0.52,
    "ymax": 0.74,
    "zmin": 0.52,
    "zmax": 0.74,
}]


def _ref_weights(sim) -> np.ndarray:
    w = np.array([1.0])
    for d in list(sim.dims.keys())[::-1]:
        w = np.multiply.outer(w, np.diff(sim.fp[d]))
    return w.reshape((sim.p + 1,) * sim.ndim)


def _total_mass(sim) -> float:
    U = sim.dm.asnumpy(sim.dm.U_cv)
    rho = U[sim._d_]
    # U_cv can be either SD layout [Nb, ..., pz,py,px] or FV layout
    # [Nb, ..., Nz*nz, Ny*ny, Nx*nx] when persistent FV mode is active.
    if rho.shape[-sim.ndim:] == (sim.p + 1,) * sim.ndim:
        w_ref = _ref_weights(sim)
    else:
        w_ref = np.array([1.0])
        for d in list(sim.dims.keys())[::-1]:
            wd = np.tile(np.diff(sim.fp[d]), sim.NB[d])
            w_ref = np.multiply.outer(w_ref, wd)
        w_ref = w_ref.reshape(rho.shape[1:])
    total = 0.0
    for ib, block in enumerate(sim.forest.blocks):
        vol_elem = 1.0
        for d in sim.dims:
            vol_elem *= block.h[d]
        total += np.sum(rho[ib] * w_ref) * vol_elem
    return float(total)


def _count_coarse_to_fine_faces(sim) -> int:
    count = 0
    for block in sim.forest.blocks:
        for d in sim.dims:
            for side in (0, 1):
                entries = block.neighbors[d][side]
                if entries and entries[0][1] == FINER:
                    count += 1
    return count


def _run_mass_drift(cfg, steps=20):
    sim = SDADER_Simulator(use_cupy=False, verbose=False, **cfg)
    cf_faces = _count_coarse_to_fine_faces(sim)
    assert cf_faces > 0, "AMR coarse-fine interfaces are required for this test."
    sim.init_sim()
    m0 = _total_mass(sim)
    max_abs_drift = 0.0
    for _ in range(steps):
        sim.compute_dt()
        sim.perform_update()
        max_abs_drift = max(max_abs_drift, abs(_total_mass(sim) - m0))
    sim.end_sim()
    return max_abs_drift, cf_faces


STRICT_CONFIGS = [
    # 2D, FB disabled (strict conservation expected).
    dict(p=1, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="SD", FB=False),
    dict(p=1, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=False),
    dict(p=3, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="SD", FB=False),
    dict(p=3, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=False),
    # 3D, FB disabled (strict conservation expected).
    dict(p=1, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="SD", FB=False),
    dict(p=1, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=False),
    dict(p=2, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="SD", FB=False),
    dict(p=2, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=False),
]


FB_DIAGNOSTIC_CONFIGS = [
    # 2D, FB paths (both blended and pure-godunov FB).
    dict(p=1, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=True),
    dict(p=1, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=True, godunov=True),
    dict(p=3, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=True),
    dict(p=3, N=(8, 8), Nb=(4, 4), levels=LEVELS_2D, update="FV", FB=True, godunov=True),
    # 3D, FB paths.
    dict(p=1, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=True),
    dict(p=1, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=True, godunov=True),
    dict(p=2, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=True),
    dict(p=2, N=(4, 4, 4), Nb=(2, 2, 2), levels=LEVELS_3D, update="FV", FB=True, godunov=True),
]


@pytest.mark.parametrize("cfg", STRICT_CONFIGS)
def test_amr_mass_conservation_strict(cfg):
    max_abs_drift, cf_faces = _run_mass_drift(cfg)
    assert max_abs_drift <= 1e-12, (
        f"Strict AMR mass drift too large: max |ΔM|={max_abs_drift:.3e} > 1e-12 "
        f"(cfg={cfg}, cf_faces={cf_faces})"
    )


@pytest.mark.parametrize("cfg", FB_DIAGNOSTIC_CONFIGS)
def test_amr_mass_conservation_fb_diagnostic(cfg):
    # Diagnostic guardrail: keeps current FB/Godunov AMR drift bounded while we
    # ensure future changes preserve machine-conservative closure.
    max_abs_drift, cf_faces = _run_mass_drift(cfg)
    assert max_abs_drift <= 1e-12, (
        f"FB diagnostic drift exceeded strict tolerance: max |ΔM|={max_abs_drift:.3e} > 1e-12 "
        f"(cfg={cfg}, cf_faces={cf_faces})"
    )


FB_STEP_IC_CONFIGS = [
    # 2D FB-only (blended and godunov).
    dict(
        p=3,
        N=(8, 8),
        Nb=(4, 4),
        levels=LEVELS_2D,
        update="FV",
        FB=True,
        godunov=False,
        init_fct=step_function,
    ),
    dict(
        p=3,
        N=(8, 8),
        Nb=(4, 4),
        levels=LEVELS_2D,
        update="FV",
        FB=True,
        godunov=True,
        init_fct=step_function,
    ),
    # 3D FB-only (blended and godunov).
    dict(
        p=2,
        N=(4, 4, 4),
        Nb=(2, 2, 2),
        levels=LEVELS_3D,
        update="FV",
        FB=True,
        godunov=False,
        init_fct=step_function,
    ),
    dict(
        p=2,
        N=(4, 4, 4),
        Nb=(2, 2, 2),
        levels=LEVELS_3D,
        update="FV",
        FB=True,
        godunov=True,
        init_fct=step_function,
    ),
]


@pytest.mark.parametrize("cfg", FB_STEP_IC_CONFIGS)
def test_amr_mass_conservation_fb_step_ic(cfg):
    # FB-only conservation check under a non-smooth IC.
    max_abs_drift, cf_faces = _run_mass_drift(cfg)
    assert max_abs_drift <= 1e-12, (
        f"FB step-IC drift exceeded strict tolerance: max |ΔM|={max_abs_drift:.3e} > 1e-12 "
        f"(cfg={cfg}, cf_faces={cf_faces})"
    )
