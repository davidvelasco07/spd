"""
Spectral Difference (SD) nested-mesh transfers using **reference-element geometry**
and **tensor-product Lagrange interpolation** (same ``solution_points`` /
``lagrange_matrix`` as ``polynomials.py`` / ``SD_Simulator``).

A coarse reference cell ``[0,1]^d`` is the union of ``2^d`` fine cells in dyadic
refinement. Each coarse solution point is mapped into the corresponding fine child's
local frame ``ξ_f = 2ξ_c − child`` and evaluates the **fine** spectral interpolant
(Lagrange tensor product on fine solution points).

``restrict_sd_w_cv_nested`` composes ``cv_to_sp`` (reference inverse of
``intfromsol_matrix``) → SP restriction → ``sp_to_cv`` on the coarse degree.

**Elliptic / V-cycle (SP):** ``laplacian_sd_sp`` (default ``operator="gram"``) builds
``Dᵀ D`` per direction from ``lagrangeprime_matrix`` (same nodal set as SD); use
``operator="colloc"`` for raw ``lagrange_second_matrix`` (indefinite for ``p>1``).
"""

from __future__ import annotations

import numpy as np

from polynomials import (
    flux_points,
    intfromsol_matrix,
    lagrange_matrix,
    lagrangeprime_matrix,
    lagrange_second_matrix,
    solution_points,
)
from transforms import compute_A_from_B, compute_A_from_B_full


def ref_sd_matrices(p: int):
    """Reference `[0,1]` SD matrices (matches `SD_Simulator` construction)."""
    x_sp = solution_points(0.0, 1.0, p)
    x_fp = flux_points(0.0, 1.0, p)
    sp_to_cv = intfromsol_matrix(x_sp, x_fp)
    cv_to_sp = np.linalg.inv(sp_to_cv)
    return x_sp, sp_to_cv, cv_to_sp


def ref_laplace_gram_matrix(p: int) -> np.ndarray:
    """
    Reference **stiffness** ``H = Dᵀ D`` with ``D = lagrangeprime_matrix(x_sp, x_sp)``.

    This is the Gram matrix of first derivatives of the Lagrange nodal basis on
    ``solution_points(0,1,p)``. It is symmetric positive semidefinite and matches
    the same ``lagrangeprime_matrix`` infrastructure as ``SD_Simulator`` (``dfp_to_sp``
    uses the same ``x_sp``, ``x_fp`` pairing; here we differentiate and contract on
    the solution grid for a **definite** elliptic building block).
    """
    x_sp = solution_points(0.0, 1.0, p)
    d1 = lagrangeprime_matrix(x_sp, x_sp)
    return d1.T @ d1


def ref_laplace_second_deriv_matrix(p: int) -> np.ndarray:
    """
    Collocation matrix ``K2[i,j] = L_j''(ξ_i)`` (``lagrange_second_matrix``).

    Indefinite for ``p > 1``; kept for analysis. Prefer ``ref_laplace_gram_matrix``
    for smoothing / V-cycles.
    """
    x_sp = solution_points(0.0, 1.0, p)
    return lagrange_second_matrix(x_sp, x_sp)


def laplacian_sd_sp(
    u: np.ndarray,
    p: int,
    ndim: int,
    h: dict[str, float],
    *,
    operator: str = "gram",
) -> np.ndarray:
    """
    Tensor-product **elliptic operator** on SD solution points (element-local).

    * ``operator="gram"`` (default): ``Σ_d (1/h[d]²) H_d`` with ``H = Dᵀ D`` from
      ``lagrangeprime_matrix`` on ``solution_points`` — symmetric positive
      semidefinite, suitable for Jacobi / V-cycle demos.

    * ``operator="colloc"``: collocation ``Σ_d (1/h[d]²) K2`` with ``K2 = L_j''(ξ_i)``
      (indefinite for ``p > 1``; not recommended for plain Jacobi).

    No coupling between elements (add face fluxes separately if needed).

    Parameters
    ----------
    u
        Shape ``(nvar, …, p+1, …, p+1)`` with ``ndim`` spatial axes then ``ndim``
        local axes (same layout as ``SD_Simulator`` / ``compute_A_from_B``).
    h
        Keys ``"x"``, ``"y"``, ``"z"`` subset for ``ndim`` (element sizes, e.g.
        ``sim.h["x"]``).
    """
    if ndim not in (1, 2, 3):
        raise ValueError("ndim must be 1, 2, or 3")
    if operator == "gram":
        K = ref_laplace_gram_matrix(p)
    elif operator == "colloc":
        K = ref_laplace_second_deriv_matrix(p)
    else:
        raise ValueError('operator must be "gram" or "colloc"')
    out = np.zeros_like(u, dtype=float)
    out += compute_A_from_B(u, K / float(h["x"]) ** 2, "x", ndim)
    if ndim >= 2:
        out += compute_A_from_B(u, K / float(h["y"]) ** 2, "y", ndim)
    if ndim >= 3:
        out += compute_A_from_B(u, K / float(h["z"]) ** 2, "z", ndim)
    return out


def laplacian_sd_sp_diagonal(
    p: int, ndim: int, h: dict[str, float], *, operator: str = "gram"
) -> np.ndarray:
    """
    Diagonal of ``laplacian_sd_sp`` (same on every element; shape ``(p+1,)*ndim``).
    """
    if operator == "gram":
        K = ref_laplace_gram_matrix(p)
    else:
        K = ref_laplace_second_deriv_matrix(p)
    pp = p + 1
    if ndim == 1:
        d = np.diag(K) / float(h["x"]) ** 2
        return d.reshape(pp)
    if ndim == 2:
        d = np.zeros((pp, pp), dtype=float)
        for ly in range(pp):
            for lx in range(pp):
                d[ly, lx] = K[ly, ly] / float(h["y"]) ** 2 + K[lx, lx] / float(h["x"]) ** 2
        return d
    d = np.zeros((pp, pp, pp), dtype=float)
    for lz in range(pp):
        for ly in range(pp):
            for lx in range(pp):
                d[lz, ly, lx] = (
                    K[lz, lz] / float(h["z"]) ** 2
                    + K[ly, ly] / float(h["y"]) ** 2
                    + K[lx, lx] / float(h["x"]) ** 2
                )
    return d


def defect_sd_sp(
    f: np.ndarray,
    u: np.ndarray,
    p: int,
    ndim: int,
    h: dict[str, float],
    *,
    operator: str = "gram",
) -> np.ndarray:
    """Residual ``r = f - L u`` for ``L = laplacian_sd_sp(..., operator=operator)``."""
    return f - laplacian_sd_sp(u, p, ndim, h, operator=operator)


def smooth_jacobi_sd_sp(
    u: np.ndarray,
    f: np.ndarray,
    p: int,
    ndim: int,
    h: dict[str, float],
    *,
    omega: float = 2.0 / 3.0,
    nu: int = 1,
    operator: str = "gram",
) -> np.ndarray:
    """``nu`` weighted Jacobi steps on ``L u = f`` with ``L = laplacian_sd_sp``."""
    d = laplacian_sd_sp_diagonal(p, ndim, h, operator=operator)
    d_bc = np.broadcast_to(
        d.reshape((1,) + (1,) * ndim + d.shape),
        u.shape,
    )
    d_safe = np.where(np.abs(d_bc) > 1e-30, d_bc, 1.0)
    v = u
    for _ in range(nu):
        r = defect_sd_sp(f, v, p, ndim, h, operator=operator)
        v = v + omega * r / d_safe
    return v


def h_coarser(h: dict[str, float]) -> dict[str, float]:
    """Element sizes double when the mesh is coarsened by a factor of two per axis."""
    return {k: float(v) * 2.0 for k, v in h.items()}


def v_cycle_sd_sp(
    u: np.ndarray,
    f: np.ndarray,
    p: int,
    ndim: int,
    h: dict[str, float],
    *,
    levels: int = 1,
    nu1: int = 2,
    nu2: int = 2,
    nu_coarse: int = 40,
    omega: float = 2.0 / 3.0,
    operator: str = "gram",
) -> np.ndarray:
    """
    One geometric **V-cycle** correction for ``L u = f`` with ``L = laplacian_sd_sp``.

    Uses ``restrict_sd_w_sp_nested`` / ``prolong_sd_w_sp_nested`` on residuals /
    corrections. Coarser operators use ``h_coarser(h)``. Deepest level applies
    ``nu_coarse`` Jacobi sweeps instead of a direct solve.

    Parameters
    ----------
    levels
        Number of coarsenings before the coarse correction (``1`` = one fine + one coarse).
    """
    if levels < 0:
        raise ValueError("levels must be non-negative")
    spatial = list(u.shape[1 : 1 + ndim])
    if levels == 0 or min(spatial) < 2:
        return smooth_jacobi_sd_sp(
            u, f, p, ndim, h, omega=omega, nu=nu_coarse, operator=operator
        )

    if any(s % 2 != 0 for s in spatial):
        raise ValueError(f"V-cycle requires even element counts to coarsen; got {spatial}")

    v = smooth_jacobi_sd_sp(
        u, f, p, ndim, h, omega=omega, nu=nu1, operator=operator
    )

    r = defect_sd_sp(f, v, p, ndim, h, operator=operator)
    rc = restrict_sd_w_sp_nested(r, ndim, p_fine=p, p_coarse=p)
    hc = h_coarser(h)
    ec = np.zeros_like(rc, dtype=float)
    ec = v_cycle_sd_sp(
        ec,
        rc,
        p,
        ndim,
        hc,
        levels=levels - 1,
        nu1=nu1,
        nu2=nu2,
        nu_coarse=nu_coarse,
        omega=omega,
        operator=operator,
    )
    v = v + prolong_sd_w_sp_nested(ec, ndim, p_fine=p, p_coarse=p)
    v = smooth_jacobi_sd_sp(
        v, f, p, ndim, h, omega=omega, nu=nu2, operator=operator
    )
    return v


def mg_sd_sp(
    u0: np.ndarray,
    f: np.ndarray,
    p: int,
    ndim: int,
    h: dict[str, float],
    *,
    cycles: int = 1,
    **v_cycle_kw,
) -> np.ndarray:
    """Apply ``cycles`` V-cycles starting from ``u0``."""
    u = u0.astype(float, copy=True)
    for _ in range(cycles):
        u = v_cycle_sd_sp(u, f, p, ndim, h, **v_cycle_kw)
    return u


def h_dict_from_simulator(sim) -> dict[str, float]:
    """Build ``h`` for ``laplacian_sd_sp`` / V-cycle from an ``SD_Simulator`` instance."""
    return {dim: float(sim.h[dim]) for dim in sim.dims}


def _lagrange_rows_dyadic_child(x_fine: np.ndarray, x_coarse: np.ndarray):
    """
    Rows ``L[k, :]`` that evaluate the fine Lagrange interpolant at coarse ref nodes
    ``x_coarse[k]``, using the dyadic map into the appropriate child:
    ``child = floor(2 ξ_c)``, ``ξ_f = 2 ξ_c − child``.
    """
    pp = x_coarse.shape[0]
    child = np.minimum(np.floor(2.0 * x_coarse).astype(np.int64), 1)
    xi_f = 2.0 * x_coarse - child
    L = np.stack(
        [lagrange_matrix(np.array([xi_f[k]]), x_fine).ravel() for k in range(pp)],
        axis=0,
    )
    return L, child


def restrict_sd_w_sp_nested(
    W_sp_fine: np.ndarray,
    ndim: int,
    *,
    p_fine: int | None = None,
    p_coarse: int | None = None,
) -> np.ndarray:
    """
    Restrict nested fine solution-point values to coarse elements.

    Layout matches ``SD_Simulator``: 2D arrays are ``(nvar, Ny, Nx, p+1, p+1)`` with
    the last two local indices tensor **y** then **x**.
    """
    if ndim not in (1, 2, 3):
        raise ValueError("ndim must be 1, 2, or 3")

    tail = W_sp_fine.shape[-ndim:]
    if len(set(tail)) != 1:
        raise ValueError("Trailing axes must equal (p+1, …, p+1)")
    pp_f = tail[0]
    if p_fine is None:
        p_fine = pp_f - 1
    if p_coarse is None:
        p_coarse = p_fine

    spatial = list(W_sp_fine.shape[1 : 1 + ndim])
    if any(s % 2 != 0 for s in spatial):
        raise ValueError(f"Need even element counts per axis; got {spatial}")

    nv = W_sp_fine.shape[0]
    x_f = solution_points(0.0, 1.0, p_fine)
    x_c = solution_points(0.0, 1.0, p_coarse)
    pp_c = p_coarse + 1

    if ndim == 1:
        Lx, cx = _lagrange_rows_dyadic_child(x_f, x_c)
        Ne = spatial[0]
        nec = Ne // 2
        idx = np.arange(pp_c)
        out = np.empty((nv, nec, pp_c), dtype=W_sp_fine.dtype)
        for ie in range(nec):
            ie_f = 2 * ie + cx
            G = np.stack([W_sp_fine[:, ie_f[k], :] for k in idx], axis=1)
            out[:, ie, :] = np.einsum("vki,ki->vk", G, Lx)
        return out

    if ndim == 2:
        Ly, cy = _lagrange_rows_dyadic_child(x_f, x_c)
        Lx, cx = _lagrange_rows_dyadic_child(x_f, x_c)
        ny, nx = spatial
        ncy, ncx = ny // 2, nx // 2
        KY, KX = np.meshgrid(np.arange(pp_c), np.arange(pp_c), indexing="ij")
        out = np.empty((nv, ncy, ncx, pp_c, pp_c), dtype=W_sp_fine.dtype)
        for ic in range(ncy):
            for jc in range(ncx):
                iy = 2 * ic + cy[KY]
                ix = 2 * jc + cx[KX]
                G = W_sp_fine[:, iy, ix, :, :]
                out[:, ic, jc, :, :] = np.einsum("vklIJ, kI, lJ->vkl", G, Ly, Lx)
        return out

    Lz, cz = _lagrange_rows_dyadic_child(x_f, x_c)
    Ly, cy = _lagrange_rows_dyadic_child(x_f, x_c)
    Lx, cx = _lagrange_rows_dyadic_child(x_f, x_c)
    nz, ny, nx = spatial
    nzc, nyc, ncx = nz // 2, ny // 2, nx // 2
    KZ, KY, KX = np.meshgrid(
        np.arange(pp_c), np.arange(pp_c), np.arange(pp_c), indexing="ij"
    )
    out = np.empty((nv, nzc, nyc, ncx, pp_c, pp_c, pp_c), dtype=W_sp_fine.dtype)
    for kcz in range(nzc):
        for icy in range(nyc):
            for jcx in range(ncx):
                iz = 2 * kcz + cz[KZ]
                iy = 2 * icy + cy[KY]
                ix = 2 * jcx + cx[KX]
                G = W_sp_fine[:, iz, iy, ix, :, :, :]
                out[:, kcz, icy, jcx, :, :, :] = np.einsum(
                    "Vuvwijk, ui, vj, wk->Vuvw", G, Lz, Ly, Lx
                )
    return out


def restrict_sd_w_cv_nested(
    W_cv_fine: np.ndarray,
    ndim: int,
    *,
    p_fine: int | None = None,
    p_coarse: int | None = None,
) -> np.ndarray:
    """CV restriction: ``cv_to_sp`` → Lagrange SP restriction → ``sp_to_cv``."""
    tail = W_cv_fine.shape[-ndim:]
    if len(set(tail)) != 1:
        raise ValueError("Trailing axes must equal (p+1, …, p+1)")
    pp_f = tail[0]
    if p_fine is None:
        p_fine = pp_f - 1
    if p_coarse is None:
        p_coarse = p_fine

    _, _, cv_to_sp_f = ref_sd_matrices(p_fine)
    _, sp_to_cv_c, _ = ref_sd_matrices(p_coarse)

    W_sp_f = compute_A_from_B_full(W_cv_fine, cv_to_sp_f, ndim)
    W_sp_c = restrict_sd_w_sp_nested(
        W_sp_f,
        ndim,
        p_fine=p_fine,
        p_coarse=p_coarse,
    )
    return compute_A_from_B_full(W_sp_c, sp_to_cv_c, ndim)


def prolong_sd_w_sp_nested(
    W_sp_coarse: np.ndarray,
    ndim: int,
    *,
    p_fine: int | None = None,
    p_coarse: int | None = None,
) -> np.ndarray:
    """
    Map coarse spectral values onto nested fine solution points.

    Fine element ``ife`` lives in coarse parent ``ife // 2`` with offset ``ox = ife % 2``.
    Parent ref coordinate ``ξ_c = (ox + ξ_f) / 2``; evaluate coarse polynomial there.
    """
    if ndim not in (1, 2, 3):
        raise ValueError("ndim must be 1, 2, or 3")

    tail = W_sp_coarse.shape[-ndim:]
    if len(set(tail)) != 1:
        raise ValueError("Trailing axes must equal (p+1, …, p+1)")
    pp_c = tail[0]
    if p_coarse is None:
        p_coarse = pp_c - 1
    if p_fine is None:
        p_fine = p_coarse

    spatial_c = list(W_sp_coarse.shape[1 : 1 + ndim])
    spatial_f = [2 * n for n in spatial_c]

    nv = W_sp_coarse.shape[0]
    x_f = solution_points(0.0, 1.0, p_fine)
    x_c = solution_points(0.0, 1.0, p_coarse)
    pp_f = p_fine + 1

    if ndim == 1:
        out = np.empty((nv, spatial_f[0], pp_f), dtype=W_sp_coarse.dtype)
        for ife in range(spatial_f[0]):
            ie_c = ife // 2
            ox = ife % 2
            xi_c = 0.5 * (ox + x_f)
            L_rows = np.stack(
                [lagrange_matrix(np.array([xi_c[k]]), x_c).ravel() for k in range(pp_f)],
                axis=0,
            )
            out[:, ife, :] = np.einsum("vj,kj->vk", W_sp_coarse[:, ie_c, :], L_rows)
        return out

    if ndim == 2:
        out = np.empty((nv, spatial_f[0], spatial_f[1], pp_f, pp_f), dtype=W_sp_coarse.dtype)
        for ify in range(spatial_f[0]):
            oy = ify % 2
            Ly = np.stack(
                [
                    lagrange_matrix(np.array([0.5 * (oy + x_f[k])]), x_c).ravel()
                    for k in range(pp_f)
                ],
                axis=0,
            )
            for ifx in range(spatial_f[1]):
                ox = ifx % 2
                Lx = np.stack(
                    [
                        lagrange_matrix(np.array([0.5 * (ox + x_f[k])]), x_c).ravel()
                        for k in range(pp_f)
                    ],
                    axis=0,
                )
                ic = ify // 2
                jc = ifx // 2
                out[:, ify, ifx, :, :] = np.einsum(
                    "vJK, jJ, kK->vjk",
                    W_sp_coarse[:, ic, jc, :, :],
                    Ly,
                    Lx,
                )
        return out

    out = np.empty(
        (nv, spatial_f[0], spatial_f[1], spatial_f[2], pp_f, pp_f, pp_f),
        dtype=W_sp_coarse.dtype,
    )
    for ifz in range(spatial_f[0]):
        oz = ifz % 2
        Lz = np.stack(
            [
                lagrange_matrix(np.array([0.5 * (oz + x_f[k])]), x_c).ravel()
                for k in range(pp_f)
            ],
            axis=0,
        )
        for ify in range(spatial_f[1]):
            oy = ify % 2
            Ly = np.stack(
                [
                    lagrange_matrix(np.array([0.5 * (oy + x_f[k])]), x_c).ravel()
                    for k in range(pp_f)
                ],
                axis=0,
            )
            for ifx in range(spatial_f[2]):
                ox = ifx % 2
                Lx = np.stack(
                    [
                        lagrange_matrix(np.array([0.5 * (ox + x_f[k])]), x_c).ravel()
                        for k in range(pp_f)
                    ],
                    axis=0,
                )
                kcz = ifz // 2
                icy = ify // 2
                jcx = ifx // 2
                out[:, ifz, ify, ifx, :, :, :] = np.einsum(
                    "vJKL, jJ, kK, lL->vjkl",
                    W_sp_coarse[:, kcz, icy, jcx, :, :, :],
                    Lz,
                    Ly,
                    Lx,
                )
    return out


def prolong_sd_w_cv_nested(
    W_cv_coarse: np.ndarray,
    ndim: int,
    *,
    p_fine: int | None = None,
    p_coarse: int | None = None,
) -> np.ndarray:
    """``cv_to_sp`` on coarse → prolong_sd_w_sp_nested → ``sp_to_cv`` on fine."""
    tail = W_cv_coarse.shape[-ndim:]
    if len(set(tail)) != 1:
        raise ValueError("Trailing axes must equal (p+1, …, p+1)")
    pp_c = tail[0]
    if p_coarse is None:
        p_coarse = pp_c - 1
    if p_fine is None:
        p_fine = p_coarse

    _, _, cv_to_sp_c = ref_sd_matrices(p_coarse)
    _, sp_to_cv_f, _ = ref_sd_matrices(p_fine)

    W_sp_c = compute_A_from_B_full(W_cv_coarse, cv_to_sp_c, ndim)
    W_sp_f = prolong_sd_w_sp_nested(
        W_sp_c,
        ndim,
        p_fine=p_fine,
        p_coarse=p_coarse,
    )
    return compute_A_from_B_full(W_sp_f, sp_to_cv_f, ndim)
