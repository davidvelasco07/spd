"""Block-based SD scheme: uniform-forest equivalence and dynamic AMR."""
import numpy as np
import pytest

from spd.spectral_difference.sd_simulator import SD_Simulator


def _gaussian(xy, case):
    x, y = xy[0], xy[1]
    r2 = (x - 0.3) ** 2 + (y - 0.3) ** 2
    if case == 0:
        return 1.0 + 0.5 * np.exp(-r2 / 0.01)
    if case in (1, 2):
        return 1.0 + 0.0 * x
    if case == 4:
        return 1.0 + 0.0 * x
    return 0.0 * x


COMMON = dict(p=2, N=(16, 16), time_integrator="rk3", cfl_coeff=0.4,
              use_cupy=False, verbose=False, riemann_solver="llf")


def _total_mass(sim):
    """Gauss-weighted density integral over all blocks."""
    w = np.diff(sim.scheme.fp["x"])
    wij = np.outer(w, w)
    W = sim.dm.W_cv
    return sum(
        float(np.einsum("yxji,ji->", W[0, ib], wij)) * b.h["x"] * b.h["y"]
        for ib, b in enumerate(sim.forest.blocks)
    )


def test_uniform_multiblock_matches_single_block():
    s1 = SD_Simulator(**COMMON)
    s1.perform_time_evolution(0.05)

    s2 = SD_Simulator(Nb=(4, 4), **COMMON)
    assert type(s2.scheme).__name__ == "SD_AMR_Scheme"
    s2.perform_time_evolution(0.05)

    W1 = s1.dm.W_cv
    W2 = s2.dm.W_cv
    W2f = np.zeros_like(W1)
    nb = 4
    for ib, b in enumerate(s2.forest.blocks):
        ix = int(round(b.lim["x"][0] * nb))
        iy = int(round(b.lim["y"][0] * nb))
        W2f[:, iy * 4:(iy + 1) * 4, ix * 4:(ix + 1) * 4] = W2[:, ib]
    assert np.abs(W1 - W2f).max() < 1e-12


def test_dynamic_amr_regrids_and_conserves_mass():
    def rough(block, W):
        return float(W[0].max() - W[0].min()) > 0.05

    def smooth(pl, sibs, sW):
        return max(float(W[0].max() - W[0].min()) for W in sW) < 0.02

    s = SD_Simulator(Nb=(4, 4), init_fct=_gaussian,
                     refine_fn=rough, derefine_fn=smooth,
                     adapt_interval=3, amr_max_level=1, **COMMON)
    m0 = _total_mass(s)
    s.perform_time_evolution(0.1)
    m1 = _total_mass(s)

    # The pulse must have triggered actual refinement.
    assert s.forest.max_level == 1
    assert s.forest.Nblocks > 16
    # Regridding (prolong/restrict + CF flux correction) is conservative.
    assert abs(m1 - m0) / abs(m0) < 1e-12
    assert np.isfinite(s.dm.W_cv).all()


def test_ader_rejected_for_block_based_runs():
    with pytest.raises(ValueError):
        SD_Simulator(p=2, N=(16, 16), Nb=(4, 4), time_integrator="ader",
                     use_cupy=False, verbose=False)
