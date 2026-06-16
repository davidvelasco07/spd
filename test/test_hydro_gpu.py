"""NumPy vs CuPy parity for hydro GPU kernels."""

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from spd import hydro


GAMMA = 1.4
MIN_C2 = 1e-10
VELS = np.array([1, 2, 3])
_P_ = 4
_D_ = 0
SHAPE = (5, 8, 8)


def _random_primitives():
    rho = np.random.uniform(0.5, 2.0, SHAPE[1:])
    p = np.random.uniform(0.5, 3.0, SHAPE[1:])
    vx = np.random.uniform(-0.3, 0.3, SHAPE[1:])
    vy = np.random.uniform(-0.3, 0.3, SHAPE[1:])
    vz = np.random.uniform(-0.3, 0.3, SHAPE[1:])
    return np.stack([rho, vx, vy, vz, p])


@pytest.fixture
def primitives():
    return _random_primitives()


def test_primitives_roundtrip(primitives):
    w = primitives.copy()
    u = hydro.compute_conservatives(w, VELS, _P_, GAMMA, _d_=_D_)
    w2 = hydro.compute_primitives(u, VELS, _P_, GAMMA, _d_=_D_, W=w.copy())
    np.testing.assert_allclose(w, w2, rtol=1e-12, atol=1e-12)


def test_conservatives_gpu_matches_cpu(primitives):
    w = primitives
    u_cpu = hydro.compute_conservatives(w, VELS, _P_, GAMMA, _d_=_D_)
    w_g = cp.asarray(w)
    u_g = cp.empty_like(w_g)
    hydro.compute_conservatives(w_g, VELS, _P_, GAMMA, _d_=_D_, U=u_g)
    np.testing.assert_allclose(u_cpu, cp.asnumpy(u_g), rtol=1e-12, atol=1e-12)


def test_primitives_gpu_matches_cpu(primitives):
    w = primitives
    u = hydro.compute_conservatives(w, VELS, _P_, GAMMA, _d_=_D_)
    w_cpu = hydro.compute_primitives(u, VELS, _P_, GAMMA, _d_=_D_)
    u_g = cp.asarray(u)
    w_g = cp.empty_like(u_g)
    hydro.compute_primitives(u_g, VELS, _P_, GAMMA, _d_=_D_, W=w_g)
    np.testing.assert_allclose(w_cpu, cp.asnumpy(w_g), rtol=1e-12, atol=1e-12)


def test_fluxes_gpu_matches_cpu(primitives):
    w = primitives
    f_cpu = hydro.compute_fluxes(w, VELS, _P_, GAMMA, _d_=_D_)
    w_g = cp.asarray(w)
    f_g = cp.empty_like(w_g)
    hydro.compute_fluxes(w_g, VELS, _P_, GAMMA, _d_=_D_, F=f_g)
    np.testing.assert_allclose(f_cpu, cp.asnumpy(f_g), rtol=1e-12, atol=1e-12)


def test_fluxes_from_conservatives_gpu_matches_cpu(primitives):
    w = primitives
    u = hydro.compute_conservatives(w, VELS, _P_, GAMMA, _d_=_D_)
    f_cpu = hydro.compute_fluxes_from_conservatives(u, VELS, _P_, GAMMA, _d_=_D_)
    u_g = cp.asarray(u)
    f_g = cp.empty_like(u_g)
    hydro.compute_fluxes_from_conservatives(u_g, VELS, _P_, GAMMA, _d_=_D_, F=f_g)
    np.testing.assert_allclose(f_cpu, cp.asnumpy(f_g), rtol=1e-12, atol=1e-12)


def test_sound_speed_gpu_matches_cpu(primitives):
    p = primitives[4]
    rho = primitives[0]
    cs_cpu = hydro.compute_cs(p, rho, GAMMA, MIN_C2)
    cs_gpu = hydro.compute_cs(cp.asarray(p), cp.asarray(rho), GAMMA, MIN_C2)
    np.testing.assert_allclose(cs_cpu, cp.asnumpy(cs_gpu), rtol=1e-12, atol=1e-12)
