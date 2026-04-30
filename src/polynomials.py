import numpy as np
from typing import Callable
from typing import Tuple

def gauss_legendre_quadrature(
    x0: float,
    x1: float,
    p: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return p points and weights from the p-th degree
    Gauss-Legendre quadrature on the interval [x0,x1].
    """
    if p == 0:
        return np.ndarray(0), np.ndarray(0)
    x, w = np.polynomial.legendre.leggauss(p)
    return x0 + (x + 1.0)*(x1 - x0)/2.0, w*(x1 - x0)/2.0

def solution_points(
    x0: float,
    x1: float,
    p: int,
) -> np.ndarray:
    """
    Return 'p + 1' Chebyshev-like points on the [x0,x1] interval.

    They therefore lie between corresponding points of the p-th degree
    Gauss-Legendre quadrature together with the bounds.
    """
    i = np.arange(p + 1)
    return x0 + (1.0 - np.cos((2*i + 1)/(2*(p + 1))*np.pi))*(x1 - x0)/2.0

def flux_points(
    x0: float,
    x1: float,
    p: int,
) -> np.ndarray:
    """
    Return p points from the p-th degree
    Gauss-Legendre quadrature on the interval [x0,x1]
    plus the 2 extreme points.
    """
    x,_=gauss_legendre_quadrature(x0,x1,p)
    return np.hstack((x0, x, x1))

def lagrange_matrix(
    x_to: np.ndarray,
    x_from: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, n) matrix mapping values defined on n points x_from to values
    defined on m points x_to, using Lagrange interpolation.

    For some elementwise scalar function f, f(x_to) \\approx LM f(x_from).
    """
    assert len(x_to.shape) == 1, "x_to must be 1D."
    assert len(x_from.shape) == 1, "x_from must be 1D."
    num = np.broadcast_to(
        x_to[:, np.newaxis, np.newaxis] - x_from[np.newaxis, np.newaxis, :],
        (x_to.shape[0], x_from.shape[0], x_from.shape[0]),
    ).copy()
    num[:, np.arange(x_from.shape[0]), np.arange(x_from.shape[0])] = 1.0
    den = x_from[:, np.newaxis] - x_from[np.newaxis, :]
    np.fill_diagonal(den, 1.0)
    return num.prod(axis=-1)/den.prod(axis=-1)[np.newaxis, :]

def lagrange_second_matrix(
    x_to: np.ndarray,
    x_from: np.ndarray,
) -> np.ndarray:
    """
    Return an ``(m, n)`` matrix of second derivatives of Lagrange polynomials defined
    on ``n`` nodes ``x_from``, evaluated at ``m`` points ``x_to``.

    Each column ``j`` is built from the degree-``len(x_from)-1`` interpolant that is
    one at ``x_from[j]`` and zero at the other nodes (same nodal basis as
    ``lagrange_matrix`` / ``lagrangeprime_matrix``).
    """
    assert len(x_to.shape) == 1, "x_to must be 1D."
    assert len(x_from.shape) == 1, "x_from must be 1D."
    p = x_from.shape[0] - 1
    m, n = x_to.shape[0], x_from.shape[0]
    out = np.zeros((m, n), dtype=float)
    for j in range(n):
        y = np.zeros(n, dtype=float)
        y[j] = 1.0
        coeffs = np.polyfit(x_from, y, deg=p)
        c2 = np.polyder(coeffs, 2)
        out[:, j] = np.polyval(c2, x_to)
    return out


def lagrangeprime_matrix(
    x_to: np.ndarray,
    x_from: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, n) matrix of first derivatives of lagrange polynomials defined
    on n points x_from evaluated on m points x_to.
    """
    assert len(x_to.shape) == 1, "x_to must be 1D."
    assert len(x_from.shape) == 1, "x_from must be 1D."
    num = np.broadcast_to(
        x_to[:, np.newaxis, np.newaxis, np.newaxis] - x_from[np.newaxis, np.newaxis, np.newaxis, :],
        (x_to.shape[0], x_from.shape[0], x_from.shape[0], x_from.shape[0]),
    ).copy()
    a = np.arange(x_from.shape[0])
    num[:, a, :, a] = 1.0
    num[:, :, a, a] = 1.0
    den = x_from[:, np.newaxis] - x_from[np.newaxis, :]
    np.fill_diagonal(den, 1.0)
    p = num.prod(axis=-1)/den.prod(axis=-1)[np.newaxis, :, np.newaxis]
    p[:, a, a] = 0.0
    return p.sum(axis=-1)

def intfromsol_matrix(
    x_sp: np.ndarray,
    x_fp: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, m) matrix to transform values defined on m points `x_sp` to mean values defined
    on m control volumes bounded by m + 1 points `x_fp`.
    """
    assert len(x_sp.shape) == 1, "x_sp must be 1D."
    assert len(x_fp.shape) == 1, "x_fp must be 1D."
    m = x_fp.shape[0] - x_sp.shape[0]
    n = x_sp.shape[0] - 1
    if n == 0:
        return np.ones((1, 1))
    x, w = gauss_legendre_quadrature(0.0, 1.0, n)
    xi = (x_fp[:-1, np.newaxis] + np.diff(x_fp)[:, np.newaxis]*x[np.newaxis, :]).ravel()
    lm = lagrange_matrix(xi, x_sp).reshape(n + m, n, n + 1)
    return np.einsum("ijk,j->ik", lm, w)

def ader_matrix(
    x_time: np.ndarray,
    w_time: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Return the ADER (n, n) matrix defined using the n x_t points and w_t weigths on the [0, dt] interval.
    """
    assert len(x_time.shape) == 1, "x_time must be 1D."
    assert len(w_time.shape) == 1, "w_time must be 1D."
    assert x_time.shape == w_time.shape, "x_time and w_time shapes must match."
    ltm = lagrange_matrix(np.array([dt]), x_time).ravel()
    lpm = lagrangeprime_matrix(x_time, x_time)
    return ltm[np.newaxis, :]*ltm[:, np.newaxis] - lpm.T*w_time[np.newaxis, :]

def quadrature_mean(
    mesh: np.ndarray,
    fct: Callable[[np.ndarray,int], np.ndarray],
    d: int,
    v: int,
    light: bool = True,
) -> np.ndarray:
    """
    Return an array with the same number of dimensions as `mesh`, the last `d` dimensions reduced by 1,
    containing mean values of `fct` inside mesh control volumes.
    Means are calculated with a Gauss-Legendre quadrature of degree one less than the maximum length of the last `d` dimensions.
    """
    n = 2 if light else max(mesh.shape[-d:]) - 1
    x, w = gauss_legendre_quadrature(0.0, 1.0, n)
    pts = mesh[(Ellipsis,) + (slice(-1),)*d + (np.newaxis,)*d]
    pts = np.broadcast_to(pts, pts.shape[:-d] + (n,)*d).copy()
    for i in range(d):
        pts += (
            np.diff(mesh[(Ellipsis,) + (slice(-1),)*i + (slice(None),) + (slice(-1),)*(d - i - 1)], axis=(i - d))[(Ellipsis,) + (np.newaxis,)*d]
            *x[(np.newaxis,)*len(mesh.shape) + (np.newaxis,)*i + (slice(None),) + (np.newaxis,)*(d - i - 1)]
        )
    args = [fct(pts,v), (Ellipsis,) + tuple(range(d))]
    for i in range(d):
        args.extend([w, (i,)])
    return np.einsum(*args)

