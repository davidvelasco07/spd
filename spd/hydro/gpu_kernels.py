"""
CuPy elementwise kernels and numpy fallbacks for 5-variable ideal hydro.

Assumes variable layout: 0=rho, 1-3=velocities, 4=pressure/energy.
Passive scalars and thermal diffusion are handled outside this module.
"""

import numpy as np

from spd.runtime.gpu import CUPY_AVAILABLE, fuse, is_gpu_array

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

    # Fully fused LLF Riemann solver: physical fluxes on both sides, sound
    # speeds, and the dissipative combination in a single kernel (replaces
    # ~15 separate elementwise kernels and their temporaries).
    llf_k = cp.ElementwiseKernel(
        "T rL, T v1L, T v2L, T v3L, T pL, T m1L, T m2L, T m3L, T eL, "
        "T rR, T v1R, T v2R, T v3R, T pR, T m1R, T m2R, T m3R, T eR, "
        "float64 gamma, float64 min_c2",
        "T F0, T F1, T F2, T F3, T F4",
        """
        T fmL = rL * v1L;
        T fmR = rR * v1R;
        T EL = pL / (gamma - 1) + 0.5 * rL * (v1L * v1L + v2L * v2L + v3L * v3L);
        T ER = pR / (gamma - 1) + 0.5 * rR * (v1R * v1R + v2R * v2R + v3R * v3R);
        T c2L = gamma * pL / rL;
        T c2R = gamma * pR / rR;
        T cL = sqrt(max(c2L, (T)min_c2)) + fabs(v1L);
        T cR = sqrt(max(c2R, (T)min_c2)) + fabs(v1R);
        T c = max(cL, cR);
        F0 = 0.5 * (fmL + fmR) - 0.5 * c * (rR - rL);
        F1 = 0.5 * (fmL * v1L + pL + fmR * v1R + pR) - 0.5 * c * (m1R - m1L);
        F2 = 0.5 * (fmL * v2L + fmR * v2R) - 0.5 * c * (m2R - m2L);
        F3 = 0.5 * (fmL * v3L + fmR * v3R) - 0.5 * c * (m3R - m3L);
        F4 = 0.5 * (v1L * (EL + pL) + v1R * (ER + pR)) - 0.5 * c * (eR - eL);
        """,
        "hydro_llf_k",
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


def llf(W_L, W_R, U_L, U_R, F, vels, gamma, min_c2):
    """Fused GPU LLF solver for the 5-variable hydro layout.

    Writes the flux into *F* in place.  Safe when *F* aliases W_R/U_R:
    each thread loads all its inputs before storing any output.
    """
    v1, v2, v3 = int(vels[0]), int(vels[1]), int(vels[2])
    llf_k(
        W_L[0], W_L[v1], W_L[v2], W_L[v3], W_L[4],
        U_L[v1], U_L[v2], U_L[v3], U_L[4],
        W_R[0], W_R[v1], W_R[v2], W_R[v3], W_R[4],
        U_R[v1], U_R[v2], U_R[v3], U_R[4],
        gamma, min_c2,
        F[0], F[v1], F[v2], F[v3], F[4],
    )
    return F


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
