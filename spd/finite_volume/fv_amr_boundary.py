"""Forest-aware FV ghost exchange and flux corrections for block-based runs.

Ported from the legacy runtime (`amr-smr-sd` branch, ``fv_simulator.py``).
FV arrays carry a meshblock axis right after nvar:

    M      : [nvar, Nb, (cz,) (cy,) cx]            (ghosted cells)
    F      : [nvar, Nb, ..., NB[dim]*n + 1, ...]   (face fluxes along dim)
    BC buf : [2, nvar, Nb, ..., Nghc slab, ...]

Neighbor relations come from ``forest.blocks[ib].neighbors[dim][side]``:

  - SAME:    copy the neighbor block's interior slab (periodic wrap included).
  - COARSER: inject (replicate) the coarse neighbor's slab to fine resolution.
  - FINER:   average the 2^(ndim-1) fine neighbors' slabs down to coarse cells.
  - BC:      apply the domain boundary condition.

``correct_coarse_fine_fv_flux`` and ``symmetrize_same_level_fv_flux`` restore
strict conservation of the blended fluxes across block interfaces.

All functions take the (fallback) scheme as ``self``; overlap-restriction
matrices live on the primary scheme's data manager (``self.primary.dm``).
"""
from __future__ import annotations

import numpy as np

from spd.numerics.slicing import cut, indices
from spd.amr.tree import SAME, FINER, COARSER, BC as BC_TAG
from spd.amr.transfer import restrict_blocks_overlap_cv


def _view(arr: np.ndarray, ib: int) -> np.ndarray:
    """Select block ib along the Nb axis (axis 1, right after nvar)."""
    return arr[:, ib]


def _avg_pairs(arr: np.ndarray, axes) -> np.ndarray:
    """Average adjacent pairs along each axis in ``axes`` (FV restriction)."""
    for ax in sorted(axes, reverse=True):
        s = arr.shape
        assert s[ax] % 2 == 0, (
            f"FV restriction requires even cell count on axis {ax}; got {s}")
        arr = arr.reshape(s[:ax] + (s[ax] // 2, 2) + s[ax + 1:]).mean(axis=ax + 1)
    return arr


def _inject_pairs(arr: np.ndarray, axes) -> np.ndarray:
    """Replicate each cell twice along each axis in ``axes`` (prolongation)."""
    for ax in sorted(axes):
        arr = np.repeat(arr, 2, axis=ax)
    return arr


def store_BC(self, M: np.ndarray, dim: str, all: bool = True,
             scalar: bool = False, BC_buf: np.ndarray = None) -> None:
    """Populate BC_buf[side, :, ib, ...] per block via the forest neighbor
    table.

    ``scalar=True`` fills ghosts for auxiliary fields (trouble indicator,
    cascade index): every physical boundary becomes a zero-gradient copy so
    no solution-dependent boundary state leaks into the flags.
    """
    idim = self.dims[dim]
    ngh = self.Nghc
    BC = self.BC[dim]
    buf = self.BC_fp[dim] if BC_buf is None else BC_buf
    # side=0 (left ghost slab of ib): the neighbor's RIGHT-interior slab.
    # side=1 (right ghost slab of ib): the neighbor's LEFT-interior slab.
    interior_cuts = (cut(-2 * ngh, -ngh, idim),
                     cut(ngh, 2 * ngh, idim))
    xp = self.dm.xp
    for side in (0, 1):
        same_jb = self.forest.same_jb[dim][side]
        if same_jb is not None:
            # Fast path: every block's neighbor on this face is SAME-level.
            # One vectorized gather along Nb replaces the Python loop.
            buf[side][...] = xp.take(
                M[interior_cuts[side]], xp.asarray(same_jb), axis=1)
            continue
        for ib, block in enumerate(self.forest.blocks):
            entries = block.neighbors[dim][side]
            bc_slot = _view(buf[side], ib)
            rel0 = entries[0][1] if entries else None
            if len(entries) == 1 and rel0 == SAME:
                jb, _rel, _sub = entries[0]
                bc_slot[...] = _view(M, jb)[interior_cuts[side]]
            elif len(entries) == 1 and rel0 == BC_TAG:
                bc_type = "gradfree" if scalar else BC[side]
                if bc_type == "reflective":
                    if all:
                        reverse = ((Ellipsis, slice(None, None, -1))
                                   + (slice(None),) * idim)
                        bc_slot[...] = _view(M, ib)[
                            interior_cuts[1 - side]][reverse]
                        bc_slot[self.vels[idim]] *= -1
                elif bc_type == "gradfree":
                    if all:
                        bc_slot[...] = _view(M, ib)[interior_cuts[1 - side]]
                elif bc_type in ("ic", "pressure"):
                    pass    # buffer left as-is / pre-populated by the caller
                elif bc_type == "eq":
                    if all:
                        bc_slot[...] = 0
                else:
                    raise ValueError(f"Undetermined boundary type: {bc_type}")
            elif len(entries) == 1 and rel0 == COARSER:
                _fill_from_coarser(self, M, dim, side, idim, ngh,
                                   bc_slot, entries[0])
            # NB: `all` is the (FV_Scheme-compatible) parameter name here,
            # so the builtin is not available.
            elif False not in [e[1] == FINER for e in entries]:
                _fill_from_finer(self, M, dim, side, idim, ngh,
                                 bc_slot, entries)
            else:
                rels = [e[1] for e in entries]
                raise NotImplementedError(
                    f"Mixed FV neighbor relations {rels} at "
                    f"(ib={ib}, dim={dim}, side={side}).")


def _fill_from_coarser(self, M, dim, side, idim, ngh, bc_slot, entry):
    """Populate a fine block's ghost slab from its coarser neighbor.

    Take the coarse neighbor's inner ngh/2 cells on the appropriate side,
    crop its transverse-active region to the sub-face region this fine block
    covers, then inject (replicate) by 2 along each cell dim so the result
    has fine resolution. Requires ngh even.
    """
    ndim = self.ndim
    jb, _rel, sub = entry
    src = _view(M, jb)
    assert ngh % 2 == 0, "FV coarse-fine ghost fill requires even Nghc"
    half_n = ngh // 2
    coarse_cut = (cut(-ngh - half_n, -ngh, idim) if side == 0
                  else cut(ngh, ngh + half_n, idim))
    coarse_slab = src[coarse_cut]
    # Select the sub-face region in the transverse dims. The coarse
    # neighbor's active transverse length is NB_d*n (= twice my own); bit k
    # of ``sub`` picks the lower/upper half along the k-th non-dim direction
    # in natural self.dims order (same convention as _sub_face_index).
    dim_keys = list(self.dims.keys())
    tv_ranges = {}
    non_dim_k = 0
    for d in dim_keys:
        if self.dims[d] == idim:
            continue
        half_t = self.NB[d] * self.n[d] // 2
        bit = (sub >> non_dim_k) & 1
        tv_ranges[d] = slice(ngh + bit * half_t, ngh + (bit + 1) * half_t)
        non_dim_k += 1
    # Per-block FV arrays have cell axes in (z, y, x) order.
    tv_slice = [slice(None)]    # nvar
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            tv_slice.append(slice(None))
        else:
            tv_slice.append(tv_ranges[d])
    coarse_sub = coarse_slab[tuple(tv_slice)]
    cell_axes = tuple(range(1, ndim + 1))
    fine = _inject_pairs(coarse_sub, cell_axes)
    # Fill the active transverse region only; transverse-ghost corners are
    # overwritten by the subsequent dim passes of Boundaries.
    dest_slice = [slice(None)]
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            dest_slice.append(slice(None))
        else:
            dest_slice.append(slice(ngh, -ngh))
    bc_slot[tuple(dest_slice)] = fine


def _fill_from_finer(self, M, dim, side, idim, ngh, bc_slot, entries):
    """Populate a coarse block's ghost slab from its 2^(ndim-1) fine
    neighbors by stacking their interior 2*ngh slabs and averaging 2^ndim
    fine cells down to one coarse cell."""
    ndim = self.ndim
    fine_interior = (cut(-3 * ngh, -ngh, idim) if side == 0
                     else cut(ngh, 3 * ngh, idim))
    dim_keys = list(self.dims.keys())
    active_tv = [slice(None)]     # nvar
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            active_tv.append(slice(None))
        else:
            active_tv.append(slice(ngh, -ngh))
    active_tv = tuple(active_tv)
    n_sub = 2 ** (ndim - 1)
    fine_slabs = [None] * n_sub
    for (jb, _rel, sub) in entries:
        fine_slabs[sub] = _view(M, jb)[fine_interior][active_tv]

    # bit_axes: which per-block cell axis each sub_idx bit addresses. Bit k
    # iterates the k-th non-dim in natural self.dims order; for a per-block
    # array [nvar, cz, cy, cx] the cell axis of dim d is (ndim - dims[d]).
    bit_axes = [ndim - self.dims[d] for d in dim_keys
                if self.dims[d] != idim]
    if ndim == 1:
        combined = fine_slabs[0]
    elif ndim == 2:
        combined = np.concatenate(fine_slabs, axis=bit_axes[0])
    else:
        inner_lo = np.concatenate([fine_slabs[0], fine_slabs[1]],
                                  axis=bit_axes[0])
        inner_hi = np.concatenate([fine_slabs[2], fine_slabs[3]],
                                  axis=bit_axes[0])
        combined = np.concatenate([inner_lo, inner_hi], axis=bit_axes[1])
    avg_axes = tuple(range(1, ndim + 1))
    coarse_avg = _avg_pairs(combined, avg_axes)
    dest_slice = [slice(None)]
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            dest_slice.append(slice(None))
        else:
            dest_slice.append(slice(ngh, -ngh))
    bc_slot[tuple(dest_slice)] = coarse_avg


def apply_BC(self, M: np.ndarray, dim: str, BC_buf: np.ndarray = None) -> None:
    """Copy BC_buf[side] into the ghost slabs of M."""
    ngh = self.Nghc
    idim = self.dims[dim]
    buf = self.BC_fp[dim] if BC_buf is None else BC_buf
    M[cut(None, ngh, idim)] = buf[0]
    M[cut(-ngh, None, idim)] = buf[1]


def Boundaries(self, M: np.ndarray, all: bool = True) -> None:
    """Fill the ghost cells of the block-based FV array M."""
    for dim in self.dims:
        store_BC(self, M, dim, all=all)
        self.comms.Comms_fv(self.dm, M, self.BC_fp, self.dims[dim],
                            dim, self.Nghc)
        apply_BC(self, M, dim)


def Boundaries_scalar(self, M: np.ndarray) -> None:
    """Ghost fill for auxiliary scalar fields (trouble indicator, cascade
    index): forest exchange as usual, zero-gradient at physical boundaries."""
    for dim in self.dims:
        store_BC(self, M, dim, scalar=True, BC_buf=self.BC_fp_scalar[dim])
        self.comms.Comms_fv(self.dm, M, self.BC_fp_scalar, self.dims[dim],
                            dim, self.Nghc)
        apply_BC(self, M, dim, BC_buf=self.BC_fp_scalar[dim])


def correct_coarse_fine_fv_flux(self, F_faces: np.ndarray, dim: str) -> None:
    """Enforce conservation across coarse-fine FV faces.

    Overwrites the coarse block's face flux with the overlap-aware
    restriction of the fine-side fluxes, so the coarse update matches the
    sum of the fine updates over the shared area. For p>1 the face
    control-volumes (Gauss widths) do not overlap as simple pairwise halves,
    hence the same overlap infrastructure as the SD restriction is used.
    """
    idim = self.dims[dim]
    ndim = self.ndim
    face_ndim = ndim - 1
    R_side_cv = self.primary.dm.RS_cv
    xp = self.dm.xp

    if face_ndim == 0:
        for ib, block in enumerate(self.forest.blocks):
            for side in (0, 1):
                entries = block.neighbors[dim][side]
                if not entries or entries[0][1] != FINER:
                    continue
                my_face_idx = 0 if side == 0 else -1
                fine_face_idx = -1 if side == 0 else 0
                jb, _rel, _sub = entries[0]
                _view(F_faces, ib)[indices(my_face_idx, idim)] = (
                    _view(F_faces, jb)[indices(fine_face_idx, idim)]
                )
        return

    dim_keys = list(self.dims.keys())
    trans_dims = [d for d in dim_keys[::-1] if d != dim]
    n = self.p + 1

    def _face_to_nested(arr_face: np.ndarray) -> np.ndarray:
        # [nvar, ..., NB[d]*n, ...] -> [nvar, NB..., n...]
        shape = [arr_face.shape[0]]
        n_trans = len(trans_dims)
        for d in trans_dims:
            shape += [self.NB[d], n]
        reshaped = arr_face.reshape(shape)
        perm = [0] + [1 + 2 * i for i in range(n_trans)] + [
            2 + 2 * i for i in range(n_trans)
        ]
        return np.transpose(reshaped, perm)

    def _nested_to_face(arr_nested: np.ndarray) -> np.ndarray:
        # [nvar, NB..., n...] -> [nvar, ..., NB[d]*n, ...]
        n_trans = len(trans_dims)
        perm = [0]
        for i in range(n_trans):
            perm += [1 + i, 1 + n_trans + i]
        interleaved = np.transpose(arr_nested, perm)
        out_shape = [arr_nested.shape[0]]
        for d in trans_dims:
            out_shape.append(self.NB[d] * n)
        return interleaved.reshape(out_shape)

    for ib, block in enumerate(self.forest.blocks):
        for side in (0, 1):
            entries = block.neighbors[dim][side]
            if not entries or entries[0][1] != FINER:
                continue
            my_face_idx = 0 if side == 0 else -1
            fine_face_idx = -1 if side == 0 else 0
            n_sub = 2 ** face_ndim
            fine_fluxes = [None] * n_sub
            for (jb, _rel, sub) in entries:
                src = _view(F_faces, jb)
                fine_fluxes[sub] = _face_to_nested(
                    src[indices(fine_face_idx, idim)]
                )
            stack = xp.stack(fine_fluxes, axis=1)
            coarse_nested = restrict_blocks_overlap_cv(
                stack, R_side_cv, face_ndim
            )
            _view(F_faces, ib)[indices(my_face_idx, idim)] = (
                _nested_to_face(coarse_nested)
            )


def symmetrize_same_level_fv_flux(self, F_faces: np.ndarray, dim: str) -> None:
    """Enforce a unique flux at SAME-level block interfaces.

    Each side of a block interface carries its own blended flux (theta can
    differ across the interface), which breaks pairwise cancellation in the
    global divergence sum. Project each interface pair onto the shared
    average to restore strict conservation.
    """
    idim = self.dims[dim]
    seen = set()
    for ib, block in enumerate(self.forest.blocks):
        for side in (0, 1):
            entries = block.neighbors[dim][side]
            if not entries or len(entries) != 1 or entries[0][1] != SAME:
                continue
            jb, _rel, _sub = entries[0]
            # A block pair can share two SAME interfaces under periodic
            # wrapping; distinguish interfaces by side pairing.
            side_j = 1 - side
            key = tuple(sorted(((ib, side), (jb, side_j)))) + (dim,)
            if key in seen:
                continue
            seen.add(key)
            my_face_idx = 0 if side == 0 else -1
            nb_face_idx = 0 if side_j == 0 else -1
            my_face = _view(F_faces, ib)[indices(my_face_idx, idim)]
            nb_face = _view(F_faces, jb)[indices(nb_face_idx, idim)]
            avg = 0.5 * (my_face + nb_face)
            _view(F_faces, ib)[indices(my_face_idx, idim)] = avg
            _view(F_faces, jb)[indices(nb_face_idx, idim)] = avg
