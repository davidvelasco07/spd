"""Prolongation and restriction between adjacent refinement levels.

Block-based: `prolongate_block` takes one coarse block's data and produces
2**ndim fine children; `restrict_blocks` is the adjoint. All operations are
ndim-agnostic.

The Lagrange matrices are the ones from the multigrid notebook:
    x_coarse = solution points on [0, 1]  (size p+1)
    x_fine   = concat(x_coarse/2, x_coarse/2 + 1/2)  (size 2*(p+1))
    LM_prolong  = lagrange_matrix(x_fine,   x_coarse)  # (2n, n)
    LM_restrict = lagrange_matrix(x_coarse, x_fine)    # (n, 2n)

`restrict_blocks o prolongate_block` recovers the original coarse data to
roundoff for any polynomial data of degree <= p (i.e., all SD solutions).
"""
from typing import Tuple
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

from spd.numerics.polynomials import lagrange_matrix
from spd.numerics.polynomials import intfromsol_matrix
from spd.numerics.transforms import compute_A_from_B_full
from spd.numerics.transforms import compute_A_from_B


def build_transfer_matrices(x_sp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (LM_prolong, LM_restrict) given the p+1 coarse solution points
    `x_sp` on [0, 1]."""
    x_coarse = x_sp
    x_fine = np.concatenate((x_coarse / 2.0, x_coarse / 2.0 + 0.5))
    LM_prolong = lagrange_matrix(x_fine, x_coarse)    # (2n, n)
    LM_restrict = lagrange_matrix(x_coarse, x_fine)   # (n, 2n)
    return LM_prolong, LM_restrict


def _segment_to_coarse_rows(x_fp: np.ndarray,
                            side_edges_local: np.ndarray,
                            side: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map child-side segments to coarse-cell indices and lengths.

    `side_edges_local` lives on child local coordinates [0, 1]. The
    corresponding global coordinates are x = 0.5*(x_local + side).
    """
    widths = np.diff(x_fp)
    edges = 0.5 * (side_edges_local + float(side))
    mids = 0.5 * (edges[:-1] + edges[1:])
    rows = np.searchsorted(x_fp, mids, side="right") - 1
    rows = np.clip(rows, 0, widths.size - 1)
    lens = np.diff(edges)
    return rows, lens


def _cv_overlap_matrix(x_fp: np.ndarray, side: int) -> np.ndarray:
    """Return (n, n) child-CV -> coarse-CV overlap restriction matrix.

    Matrix rows are coarse CVs and columns are child-side fine CVs.
    Coefficients are overlap length divided by coarse-cell width.
    """
    n = x_fp.size - 1
    out = np.zeros((n, n))
    coarse_w = np.diff(x_fp)
    fine_edges = 0.5 * (x_fp + float(side))
    for j in range(n):
        a0, a1 = x_fp[j], x_fp[j + 1]
        for i in range(n):
            b0, b1 = fine_edges[i], fine_edges[i + 1]
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            if overlap > 0.0:
                out[j, i] += overlap / coarse_w[j]
    return out


def build_overlap_restrict_matrices(
    x_sp: np.ndarray,
    x_fp: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build side-aware overlap restriction matrices.

    Returns
    -------
    (R_sp_0, R_sp_1), (R_cv_0, R_cv_1)
        - R_sp_* : child-SP -> coarse-CV contribution matrix on side *.
        - R_cv_* : child-CV -> coarse-CV contribution matrix on side *.
    """
    p = x_sp.size - 1
    n = p + 1
    m = (p + 2) // 2
    x_c = 2.0 * x_fp
    # For even p, x=0.5 is not a Gauss flux point; insert the split.
    if p % 2 == 0:
        x_c = np.concatenate((x_c[:m], np.ones(1), x_c[m:]))
    x_1 = x_c[:m + 1]
    x_2 = x_c[m:] - 1.0
    side_edges = (x_1, x_2)

    R_sp = []
    for side in (0, 1):
        L_side = intfromsol_matrix(x_sp, side_edges[side])
        rows, lens = _segment_to_coarse_rows(x_fp, side_edges[side], side)
        R_side = np.zeros((n, n))
        widths = np.diff(x_fp)
        for k, j in enumerate(rows):
            R_side[j] += (lens[k] / widths[j]) * L_side[k]
        R_sp.append(R_side)

    R_cv = [_cv_overlap_matrix(x_fp, side) for side in (0, 1)]
    return (R_sp[0], R_sp[1]), (R_cv[0], R_cv[1])


def _restrict_blocks_overlap(
    W_fine: np.ndarray,
    R_side: Tuple[np.ndarray, np.ndarray],
    ndim: int,
) -> np.ndarray:
    """Side-aware overlap restriction to coarse CV representation."""
    xp = np
    if CUPY_AVAILABLE and isinstance(W_fine, cp.ndarray):
        xp = cp
    lead = W_fine.ndim - 1 - 2 * ndim
    n = R_side[0].shape[0]
    NB = list(W_fine.shape[lead + 1:lead + 1 + ndim])

    # 1) Expand the child axis into ndim binary child-direction axes.
    split = list(W_fine.shape[:lead]) + [2] * ndim + NB + [n] * ndim
    Wf = W_fine.reshape(split)
    # 2) Interleave child-direction and cell axes.
    child_axes = [lead + k for k in range(ndim)]
    cell_axes = [lead + ndim + k for k in range(ndim)]
    pt_axes = [lead + 2 * ndim + k for k in range(ndim)]
    interleaved = []
    for k in range(ndim):
        interleaved += [child_axes[k], cell_axes[k]]
    perm = list(range(lead)) + interleaved + pt_axes
    Wf = Wf.transpose(perm)
    # 3) Merge each (2_child, NB) pair to a single fine-element axis (2*NB).
    merged = list(Wf.shape[:lead]) + [2 * nb for nb in NB] + [n] * ndim
    Wf = Wf.reshape(merged)

    out = xp.zeros(Wf.shape[:lead] + tuple(NB) + (n,) * ndim, dtype=W_fine.dtype)
    dims = ("x", "y", "z")
    # 4) Gather parity subsets (fine element parity per dim), apply side-aware
    #    point restriction, and accumulate contributions into coarse CVs.
    for sub_idx in range(2 ** ndim):
        # Cell axes are ordered (z,y,x); sub_idx bits are (x,y,z).
        cell_sel = tuple(
            slice((sub_idx >> (ndim - 1 - ax)) & 1, None, 2)
            for ax in range(ndim)
        )
        contrib = Wf[(slice(None),) * lead + cell_sel + (Ellipsis,)]
        for k in range(ndim):
            side = (sub_idx >> k) & 1
            Rk = R_side[side]
            if CUPY_AVAILABLE and xp is cp and not isinstance(Rk, cp.ndarray):
                Rk = cp.asarray(Rk)
            contrib = compute_A_from_B(contrib, Rk, dims[k], ndim)
        out += contrib
    return out


def restrict_blocks_overlap_sp(
    W_fine: np.ndarray,
    R_side_sp: Tuple[np.ndarray, np.ndarray],
    cv_to_sp: np.ndarray,
    ndim: int,
) -> np.ndarray:
    """2**ndim fine children -> coarse block using overlap-aware CV transfer.

    Child data is interpreted at solution points. Restriction is:
      child SP -> overlap-weighted coarse CV -> coarse SP.
    """
    W_cv = _restrict_blocks_overlap(W_fine, R_side_sp, ndim)
    return compute_A_from_B_full(W_cv, cv_to_sp, ndim)


def restrict_blocks_overlap_cv(
    W_fine: np.ndarray,
    R_side_cv: Tuple[np.ndarray, np.ndarray],
    ndim: int,
) -> np.ndarray:
    """2**ndim fine children -> coarse CV data with overlap weights."""
    return _restrict_blocks_overlap(W_fine, R_side_cv, ndim)


def prolongate_block(W_coarse: np.ndarray,
                     LM_prolong: np.ndarray,
                     ndim: int) -> np.ndarray:
    """Split one coarse block into 2**ndim fine children.

    A coarse block with ``NB`` elements per dim becomes 2**ndim children,
    each with the same ``NB`` but at half the physical scale. Child index
    ordering is row-major (iz, iy, ix) -> iz*4 + iy*2 + ix (3D).

    Parameters
    ----------
    W_coarse : shape [..., NzB, NyB, NxB, pz, py, px] for ndim=3 (drop axes
               as appropriate for 1D/2D). The ``...`` absorbs nvar + any
               extra batch axes.
    LM_prolong : Lagrange matrix of shape (2(p+1), p+1).
    ndim : spatial dimension.

    Returns
    -------
    W_fine : shape [..., 2**ndim, NzB, NyB, NxB, pz, py, px].
    """
    # 1) Interpolate each element's p+1 solution points -> 2*(p+1) fine points.
    Wf = compute_A_from_B_full(W_coarse, LM_prolong, ndim)
    # Layout now: [..., cells_d..cells_1, 2n_d..2n_1]

    n = LM_prolong.shape[0] // 2   # = p+1
    lead = Wf.ndim - 2 * ndim      # number of leading (batch) axes
    NB = list(Wf.shape[lead:lead + ndim])

    # 2) Split each 2n point axis into (2_sub, n). 2_sub is the left/right
    #    half of each coarse element.
    split = list(Wf.shape[:lead + ndim])
    for _ in range(ndim):
        split += [2, n]
    Wf = Wf.reshape(split)

    # 3) Interleave (coarse, sub) pairs so that flattening them yields the
    #    fine-element index (fine = 2*coarse + sub). Axes layout:
    #        [..., NzB, 2z, NyB, 2y, NxB, 2x, nz, ny, nx]
    coarse_axes = list(range(lead, lead + ndim))                # NzB, NyB, NxB
    sub_axes = [lead + ndim + 2 * k for k in range(ndim)]       # 2z, 2y, 2x
    n_axes = [lead + ndim + 2 * k + 1 for k in range(ndim)]     # nz, ny, nx
    interleaved = []
    for k in range(ndim):
        interleaved += [coarse_axes[k], sub_axes[k]]
    perm = list(range(lead)) + interleaved + n_axes
    Wf = Wf.transpose(perm)

    # 4) Merge each (coarse, sub) pair into a fine axis of size 2*NB.
    merged = list(Wf.shape[:lead]) + [2 * nb for nb in NB] + [n] * ndim
    Wf = Wf.reshape(merged)

    # 5) Split each fine axis 2*NB into (2_child, NB): the first NB fine
    #    elements become child 0, the next NB become child 1.
    split2 = list(Wf.shape[:lead])
    for nb in NB:
        split2 += [2, nb]
    split2 += [n] * ndim
    Wf = Wf.reshape(split2)

    # 6) Transpose so all child axes come first (in z,y,x order), then cells,
    #    then points.
    child_axes = [lead + 2 * k for k in range(ndim)]            # positions of the "2_child" axes
    cell_axes2 = [lead + 2 * k + 1 for k in range(ndim)]        # positions of NB axes
    pt_axes = [lead + 2 * ndim + k for k in range(ndim)]
    perm2 = list(range(lead)) + child_axes + cell_axes2 + pt_axes
    Wf = Wf.transpose(perm2)

    # 7) Merge the `ndim` child axes into a single axis of size 2**ndim.
    out = list(Wf.shape[:lead]) + [2 ** ndim] + NB + [n] * ndim
    return Wf.reshape(out)


def restrict_blocks(W_fine: np.ndarray,
                    LM_restrict: np.ndarray,
                    ndim: int) -> np.ndarray:
    """Adjoint of ``prolongate_block``: 2**ndim fine children -> 1 coarse block.

    Parameters
    ----------
    W_fine : shape [..., 2**ndim, NzB, NyB, NxB, pz, py, px] with children in
             row-major (iz, iy, ix) order.
    LM_restrict : Lagrange matrix of shape (p+1, 2(p+1)).
    ndim : spatial dimension.

    Returns
    -------
    W_coarse : shape [..., NzB, NyB, NxB, pz, py, px].
    """
    lead = W_fine.ndim - 1 - 2 * ndim
    NB = list(W_fine.shape[lead + 1:lead + 1 + ndim])
    n = LM_restrict.shape[0]        # = p+1

    # 1) Split the 2**ndim child axis into ``ndim`` "2_child" axes.
    split = list(W_fine.shape[:lead]) + [2] * ndim + NB + [n] * ndim
    Wf = W_fine.reshape(split)

    # 2) Interleave 2_child with its NB: [..., 2z, NzB, 2y, NyB, 2x, NxB, pz, py, px].
    child_axes = [lead + k for k in range(ndim)]
    cell_axes = [lead + ndim + k for k in range(ndim)]
    pt_axes = [lead + 2 * ndim + k for k in range(ndim)]
    interleaved = []
    for k in range(ndim):
        interleaved += [child_axes[k], cell_axes[k]]
    perm = list(range(lead)) + interleaved + pt_axes
    Wf = Wf.transpose(perm)

    # 3) Merge each (2_child, NB) pair into 2*NB (fine-element index).
    merged = list(Wf.shape[:lead]) + [2 * nb for nb in NB] + [n] * ndim
    Wf = Wf.reshape(merged)

    # 4) Split each 2*NB into (NB, 2_sub). Sub-element index comes second so
    #    the prolongate reshape's ``fine = 2*coarse + sub`` is recovered.
    split2 = list(Wf.shape[:lead])
    for nb in NB:
        split2 += [nb, 2]
    split2 += [n] * ndim
    Wf = Wf.reshape(split2)

    # 5) Transpose: [..., NzB, NyB, NxB, 2z_sub, nz, 2y_sub, ny, 2x_sub, nx].
    coarse_axes = [lead + 2 * k for k in range(ndim)]
    sub_axes = [lead + 2 * k + 1 for k in range(ndim)]
    pt_axes2 = [lead + 2 * ndim + k for k in range(ndim)]
    pt_pairs = []
    for k in range(ndim):
        pt_pairs += [sub_axes[k], pt_axes2[k]]
    perm2 = list(range(lead)) + coarse_axes + pt_pairs
    Wf = Wf.transpose(perm2)

    # 6) Merge each (2_sub, n) pair into 2n.
    merged2 = list(Wf.shape[:lead]) + NB + [2 * n] * ndim
    Wf = Wf.reshape(merged2)

    # 7) Lagrange restriction (n, 2n) along each point axis -> coarse block.
    return compute_A_from_B_full(Wf, LM_restrict, ndim)
