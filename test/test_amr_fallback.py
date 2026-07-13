"""Block-based SDFB (MUSCL fallback) runs: uniform-forest equivalence and
dynamic AMR with shock capturing."""
import numpy as np
import pytest

from spd.hydro.hydro_simulator import HydroSimulator


def _pulse(xy, case):
    """Smooth pressure pulse (fallback rarely triggers)."""
    x, y = xy[0], xy[1]
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if case == 0:
        return 1.0 + 0.0 * x
    if case in (1, 2, 3):
        return 0.0 * x
    return 1.0 + 3.0 * np.exp(-r2 / 0.01)


def _sedov(xy, case):
    """Strong (100:1) Gaussian-smoothed blast: needs the fallback."""
    x, y = xy[0], xy[1]
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if case == 0:
        return 1.0 + 0.0 * x
    if case in (1, 2, 3):
        return 0.0 * x
    return 0.1 + 100.0 * np.exp(-r2 / 0.001)


COMMON = dict(scheme="SDFB", p=2, N=(16, 16), time_integrator="rk3",
              cfl_coeff=0.4, use_cupy=False, verbose=False)


def _total_mass(sim):
    """Gauss-weighted density integral over all blocks."""
    w = np.diff(sim.ho_scheme.fp["x"])
    wij = np.outer(w, w)
    W = sim.dm.W_cv
    return sum(
        float(np.einsum("yxji,ji->", W[0, ib], wij)) * b.h["x"] * b.h["y"]
        for ib, b in enumerate(sim.forest.blocks)
    )


def test_uniform_multiblock_sdfb_matches_single_block():
    s1 = HydroSimulator(init_fct=_pulse, **COMMON)
    s1.perform_time_evolution(0.03)

    s2 = HydroSimulator(Nb=(4, 4), init_fct=_pulse, **COMMON)
    assert type(s2.ho_scheme).__name__ == "SD_AMR_Scheme"
    assert type(s2.lo_scheme).__name__ == "FallbackAMRScheme"
    s2.perform_time_evolution(0.03)

    W1 = s1.dm.W_cv
    W2 = s2.dm.W_cv
    W2f = np.zeros_like(W1)
    nb = 4
    for ib, b in enumerate(s2.forest.blocks):
        ix = int(round(b.lim["x"][0] * nb))
        iy = int(round(b.lim["y"][0] * nb))
        W2f[:, iy * 4:(iy + 1) * 4, ix * 4:(ix + 1) * 4] = W2[:, ib]
    assert np.abs(W1 - W2f).max() < 1e-11


def test_dynamic_amr_fallback_shock_conserves_mass():
    def rough(block, W):
        P = W[-1]
        return float(P.max() - P.min()) > 0.5

    def smooth(pl, sibs, sW):
        return max(float(W[-1].max() - W[-1].min()) for W in sW) < 0.1

    s = HydroSimulator(Nb=(4, 4), init_fct=_sedov, riemann_solver="hllc",
                       tolerance=1e-4, refine_fn=rough, derefine_fn=smooth,
                       adapt_interval=2, amr_max_level=2, **COMMON)
    m0 = _total_mass(s)
    troubled_steps = 0
    t_end = 0.015
    while s.time < t_end:
        s.compute_dt()
        if s.time + s.dt > t_end:
            s.dt = t_end - s.time
            s.ho_scheme.dt = s.dt
        s.perform_update()
        if float(s.lo_scheme.dm.troubles.sum()) > 0:
            troubled_steps += 1
        assert np.isfinite(s.dm.W_cv).all()
    m1 = _total_mass(s)

    # The blast must refine and actually engage the MUSCL fallback.
    assert s.forest.max_level >= 1
    assert s.forest.Nblocks > 16
    assert troubled_steps > 0
    # Blended fluxes stay strictly conservative across block interfaces
    # (CF restriction + same-level symmetrization) and across regrids.
    assert abs(m1 - m0) / abs(m0) < 1e-12
    # Positivity held up (PAD + fallback on a 100:1 blast).
    assert float(s.dm.W_cv[0].min()) > 0
    assert float(s.dm.W_cv[-1].min()) > 0


def test_godunov_mode_on_blocks():
    """Pure Godunov (theta=1 everywhere) on a static multi-block forest."""
    s = HydroSimulator(Nb=(2, 2), init_fct=_sedov, godunov=True,
                       riemann_solver="hllc", **COMMON)
    m0 = _total_mass(s)
    s.perform_time_evolution(0.01)
    m1 = _total_mass(s)
    assert np.isfinite(s.dm.W_cv).all()
    assert abs(m1 - m0) / abs(m0) < 1e-12


def test_mood_rejected_on_blocks():
    with pytest.raises(NotImplementedError):
        HydroSimulator(Nb=(2, 2), init_fct=_pulse, blending=False, **COMMON)
