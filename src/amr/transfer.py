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

from polynomials import lagrange_matrix
from transforms import compute_A_from_B_full


def build_transfer_matrices(x_sp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (LM_prolong, LM_restrict) given the p+1 coarse solution points
    `x_sp` on [0, 1]."""
    x_coarse = x_sp
    x_fine = np.concatenate((x_coarse / 2.0, x_coarse / 2.0 + 0.5))
    LM_prolong = lagrange_matrix(x_fine, x_coarse)    # (2n, n)
    LM_restrict = lagrange_matrix(x_coarse, x_fine)   # (n, 2n)
    return LM_prolong, LM_restrict


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
