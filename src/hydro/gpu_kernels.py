"""
CuPy elementwise kernels and numpy fallbacks for 5-variable ideal hydro.

Assumes variable layout: 0=rho, 1-3=velocities, 4=pressure/energy.
Passive scalars and thermal diffusion are handled outside this module.
"""

import numpy as np

from runtime.gpu import CUPY_AVAILABLE, fuse, is_gpu_array

if CUPY_AVAILABLE:
    import cupy as cp


@fuse
def conservatives_f(rho, vx, vy, vz, p, gamma):
    mx = rho * vx
    my = rho * vy
    mz = rho * vz
    k = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    return mx, my, mz, p / (gamma - 1) + k


@fuse
def primitives_f(rho, mx, my, mz, e, gamma):
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    k = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    return vx, vy, vz, (gamma - 1) * (e - k)


@fuse
def fluxes_f(rho, v1, v2, v3, p, gamma):
    m1 = rho * v1
    k = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    e = p / (gamma - 1) + k
    return m1, m1 * v1 + p, m1 * v2, m1 * v3, v1 * (e + p)


@fuse
def conserve_f(rho, primitive):
    return rho * primitive


@fuse
def primitive_f(rho, conserved):
    return conserved / rho


@fuse
def temperature_f(rho, p):
    return p / rho


if CUPY_AVAILABLE:
    cs_k = cp.ElementwiseKernel(
        "T P, T rho, float64 gamma, float64 min_c2",
        "T cs",
        """
        T cs2 = gamma * P / rho;
        cs2 = max(cs2, min_c2);
        cs = sqrt(cs2);
        """,
        "hydro_cs_k",
    )

    conservatives_k = cp.ElementwiseKernel(
        "T rho, T vx, T vy, T vz, T p, float64 gamma",
        "T mx, T my, T mz, T e",
        """
        mx = rho * vx;
        my = rho * vy;
        mz = rho * vz;
        e = p / (gamma - 1) + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
        """,
        "hydro_conservatives_k",
    )

    primitives_k = cp.ElementwiseKernel(
        "T rho, T mx, T my, T mz, T e, float64 gamma",
        "T vx, T vy, T vz, T p",
        """
        vx = mx / rho;
        vy = my / rho;
        vz = mz / rho;
        p = (gamma - 1) * (e - 0.5 * rho * (vx * vx + vy * vy + vz * vz));
        """,
        "hydro_primitives_k",
    )

    fluxes_k = cp.ElementwiseKernel(
        "T rho, T v1, T v2, T v3, T p, float64 gamma",
        "T m1, T Fm1, T Fm2, T Fm3, T Fe",
        """
        double e;
        m1 = v1 * rho;
        Fm1 = m1 * v1 + p;
        Fm2 = m1 * v2;
        Fm3 = m1 * v3;
        e = p / (gamma - 1) + 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3);
        Fe = v1 * (e + p);
        """,
        "hydro_fluxes_k",
    )

    fluxes_ck = cp.ElementwiseKernel(
        "T rho, T m1, T m2, T m3, T e, float64 gamma",
        "T Frho, T Fm1, T Fm2, T Fm3, T Fe",
        """
        T v1 = m1 / rho;
        T p = (gamma - 1) * (e - 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / rho);
        Frho = m1;
        Fm1 = m1 * v1 + p;
        Fm2 = v1 * m2;
        Fm3 = v1 * m3;
        Fe = v1 * (e + p);
        """,
        "hydro_fluxes_ck",
    )


def conservatives(W, U, gamma):
    if is_gpu_array(W[0]):
        conservatives_k(W[0], W[1], W[2], W[3], W[4], gamma, U[1], U[2], U[3], U[4])
    else:
        U[1], U[2], U[3], U[4] = conservatives_f(
            W[0], W[1], W[2], W[3], W[4], gamma
        )


def primitives(U, W, gamma):
    if is_gpu_array(U[0]):
        primitives_k(U[0], U[1], U[2], U[3], U[4], gamma, W[1], W[2], W[3], W[4])
    else:
        W[1], W[2], W[3], W[4] = primitives_f(
            U[0], U[1], U[2], U[3], U[4], gamma
        )


def fluxes(W, F, vels, gamma):
    v1, v2, v3 = int(vels[0]), int(vels[1]), int(vels[2])
    if is_gpu_array(W[0]):
        fluxes_k(W[0], W[v1], W[v2], W[v3], W[4], gamma, F[0], F[v1], F[v2], F[v3], F[4])
    else:
        F[0], F[v1], F[v2], F[v3], F[4] = fluxes_f(
            W[0], W[v1], W[v2], W[v3], W[4], gamma
        )


def fluxes_from_conservatives(U, F, vels, gamma):
    v1, v2, v3 = int(vels[0]), int(vels[1]), int(vels[2])
    if is_gpu_array(U[0]):
        fluxes_ck(U[0], U[v1], U[v2], U[v3], U[4], gamma, F[0], F[v1], F[v2], F[v3], F[4])
    else:
        m1 = U[v1]
        rho = U[0]
        v1n = m1 / rho
        p = (gamma - 1) * (
            U[4] - 0.5 * (U[v1] ** 2 + U[v2] ** 2 + U[v3] ** 2) / rho
        )
        F[0] = m1
        F[v1] = m1 * v1n + p
        F[v2] = v1n * U[v2]
        F[v3] = v1n * U[v3]
        F[4] = v1n * (U[4] + p)


def sound_speed(P, rho, gamma, min_c2):
    if is_gpu_array(P):
        return cs_k(P, rho, gamma, min_c2)
    return np.sqrt(np.where(gamma * P / rho > min_c2, gamma * P / rho, min_c2))


def conserve(rho, primitive):
    if is_gpu_array(rho):
        return rho * primitive
    return conserve_f(rho, primitive)


def primitive(rho, conserved):
    if is_gpu_array(rho):
        return conserved / rho
    return primitive_f(rho, conserved)


def temperature(rho, p):
    if is_gpu_array(rho):
        return p / rho
    return temperature_f(rho, p)
