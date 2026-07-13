import numpy as np

from . import gpu_kernels as gk


def compute_cs2(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float) -> np.ndarray:
    """
    Returns the square of the sound speed, with minimum value min_c2
    """
    c2 = gamma * P / rho
    return np.where(c2 > min_c2, c2, min_c2)


def compute_cs(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float) -> np.ndarray:
    """Returns the sound speed, with minimum value sqrt(min_c2)."""
    return gk.sound_speed(P, rho, gamma, min_c2)


def compute_primitives(
        U: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        _t_=None,
        W=None,
        thdiffusion: bool = False,
        npassive=0,
        **kwargs) -> np.ndarray:
    """Conservative to primitive variables (GPU path for the 5 core fields)."""
    if W is None:
        W = U.copy()
    assert W.shape == U.shape
    W[_d_] = U[_d_]
    gk.primitives(U, W, gamma)
    if thdiffusion:
        W[_t_] = gk.temperature(W[_d_], W[_p_])
    if npassive > 0:
        _ps_ = _p_ + 1
        for i in range(npassive):
            W[_ps_ + i] = gk.primitive(U[_d_], U[_ps_ + i])
    return W


def compute_conservatives(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        _t_=None,
        U=None,
        thdiffusion: bool = False,
        npassive=0,
        **kwargs) -> np.ndarray:
    """Primitive to conservative variables (GPU path for the 5 core fields)."""
    if U is None:
        U = W.copy()
    assert U.shape == W.shape
    U[_d_] = W[_d_]
    gk.conservatives(W, U, gamma)
    if thdiffusion:
        U[_t_] = gk.temperature(W[_d_], W[_p_])
    if npassive > 0:
        _ps_ = _p_ + 1
        for i in range(npassive):
            U[_ps_ + i] = gk.conserve(W[_d_], W[_ps_ + i])
    return U


def compute_fluxes(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        F=None,
        npassive=0,
        **kwargs) -> np.ndarray:
    """Conservative fluxes from primitive variables."""
    if F is None:
        F = W.copy()
    gk.fluxes(W, F, vels, gamma)
    if npassive > 0:
        _ps_ = _p_ + 1
        F[_ps_:_ps_ + npassive, ...] = F[0] * W[_ps_:_ps_ + npassive]
    return F


def compute_fluxes_from_conservatives(
        U: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        F=None,
        npassive=0,
        **kwargs) -> np.ndarray:
    """Conservative fluxes without a primitive round-trip (ADER face states)."""
    if F is None:
        F = U.copy()
    gk.fluxes_from_conservatives(U, F, vels, gamma)
    if npassive > 0:
        _ps_ = _p_ + 1
        F[_ps_:_ps_ + npassive, ...] = F[0] * U[_ps_:_ps_ + npassive] / U[_d_]
    return F


def compute_viscous_fluxes(
        W: np.ndarray,
        vels: np.array,
        dWs: dict,
        _p_: int,
        nu: float,
        beta: float,
        _d_: int = 0,
        F=None,
        npassive=0,
        **kwargs) -> np.ndarray:
    """
    Returns array of viscous fluxes for conservative variables
    """
    if F is None:
        F = W.copy()
    F[...] = 0
    v1 = vels[0]
    dW1 = dWs[v1 - 1]
    F[v1] = 2 * dW1[v1] - beta * dW1[v1]
    F[_p_] = W[v1] * F[v1]
    for vel in vels[1:]:
        idim = vel - 1
        dW = dWs[idim]
        F[v1] -= beta * dW[vel]
        F[vel] = dW1[vel] + dW[v1]
        F[_p_] += W[vel] * F[vel]
    if npassive > 0:
        _ps_ = _p_ + 1
        F[_ps_:_ps_ + npassive] = dW1[_ps_:_ps_ + npassive]
    return F * W[_d_] * nu


def compute_thermal_fluxes(
        W: np.ndarray,
        dW: np.ndarray,
        chi: float,
        _t_: int,
        _d_: int = 0,
        **kwargs) -> np.ndarray:
    return chi * W[_d_] * dW[_t_]
