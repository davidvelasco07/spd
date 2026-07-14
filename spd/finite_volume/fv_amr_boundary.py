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

    Blocks are processed in relation groups (one vectorized gather/scatter
    per group): the per-block Python loop makes GPU runs launch-bound.
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
                M[interior_cuts[side]],
                self.forest.dev_same_jb(dim, side, xp), axis=1)
            continue
        groups = self.forest.dev_groups(dim, side, xp)
        if groups is None:
            _store_BC_block_loop(self, M, dim, side, all, scalar, buf,
                                 interior_cuts)
            continue

        interior = M[interior_cuts[side]]   # view [nvar, Nb, ...]
        dst = buf[side]                     # [nvar, Nb, ...]

        ib_s, jb_s = groups["same"]
        if len(ib_s):
            dst[:, ib_s] = interior[:, jb_s]

        for ib in groups["bc"]:
            _fill_domain_bc(self, M, int(ib), side, idim, all, scalar, BC,
                            _view(dst, int(ib)), interior_cuts)

        _fill_from_coarser_fused(self, M, dim, side, idim, ngh, dst,
                                 groups["coarser_by_sub"])

        fi_ib, fi_jb = groups["finer"]
        if len(fi_ib):
            _fill_from_finer_batch(self, M, dim, side, idim, ngh,
                                   dst, fi_ib, fi_jb)


def _fill_domain_bc(self, M, ib, side, idim, all, scalar, BC, bc_slot,
                    interior_cuts):
    """Domain-boundary ghost fill for one block (BC types need per-type
    logic; the number of boundary blocks is small)."""
    bc_type = "gradfree" if scalar else BC[side]
    if bc_type == "reflective":
        if all:
            reverse = ((Ellipsis, slice(None, None, -1))
                       + (slice(None),) * idim)
            bc_slot[...] = _view(M, ib)[interior_cuts[1 - side]][reverse]
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


def _store_BC_block_loop(self, M, dim, side, all, scalar, buf,
                         interior_cuts):
    """Per-block fallback for faces with mixed/unsupported relations."""
    idim = self.dims[dim]
    ngh = self.Nghc
    BC = self.BC[dim]
    for ib, block in enumerate(self.forest.blocks):
        entries = block.neighbors[dim][side]
        bc_slot = _view(buf[side], ib)
        rel0 = entries[0][1] if entries else None
        if len(entries) == 1 and rel0 == SAME:
            jb, _rel, _sub = entries[0]
            bc_slot[...] = _view(M, jb)[interior_cuts[side]]
        elif len(entries) == 1 and rel0 == BC_TAG:
            _fill_domain_bc(self, M, ib, side, idim, all, scalar, BC,
                            bc_slot, interior_cuts)
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


def _inject_coarser_slab(self, coarse_slab, dim, idim, ngh, dst, ib_c, sub):
    """Inject a gathered coarse ghost slab into fine slots for one ``sub``."""
    ndim = self.ndim
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
    tv_slice = [slice(None), slice(None)]   # nvar, K
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            tv_slice.append(slice(None))
        else:
            tv_slice.append(tv_ranges[d])
    fine = _inject_pairs(coarse_slab[tuple(tv_slice)],
                         tuple(range(2, ndim + 2)))
    dest_slice = [slice(None), ib_c]
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            dest_slice.append(slice(None))
        else:
            dest_slice.append(slice(ngh, -ngh))
    dst[tuple(dest_slice)] = fine


def _fill_from_coarser_fused(self, M, dim, side, idim, ngh, dst, by_sub):
    """Fused coarse→fine ghost fill over all ``sub`` buckets.

    Previously each of the ``2**(ndim-1)`` sub-positions issued its own
    gather from ``M``; that was a major launch source with many C-F faces.
    We gather once for the concatenated neighbor list, then inject each
    sub-group (different transverse tile) from the shared gather.
    """
    items = [(int(sub), ib_c, jb_c)
             for sub, (ib_c, jb_c) in by_sub.items() if len(ib_c)]
    if not items:
        return
    assert ngh % 2 == 0, "FV coarse-fine ghost fill requires even Nghc"
    half_n = ngh // 2
    coarse_cut = (cut(-ngh - half_n, -ngh, idim) if side == 0
                  else cut(ngh, ngh + half_n, idim))
    if len(items) == 1:
        sub, ib_c, jb_c = items[0]
        _inject_coarser_slab(self, M[coarse_cut][:, jb_c], dim, idim, ngh,
                             dst, ib_c, sub)
        return

    xp = self.dm.xp
    ib_all = xp.concatenate([ib for _, ib, _ in items])
    jb_all = xp.concatenate([jb for _, _, jb in items])
    coarse_slab_all = M[coarse_cut][:, jb_all]
    offset = 0
    for sub, ib_c, jb_c in items:
        n = int(ib_c.shape[0]) if hasattr(ib_c, "shape") else len(ib_c)
        _inject_coarser_slab(self, coarse_slab_all[:, offset:offset + n],
                             dim, idim, ngh, dst, ib_c, sub)
        offset += n


def _fill_from_coarser_batch(self, M, dim, side, idim, ngh, dst,
                             ib_c, jb_c, sub):
    """Batched ``_fill_from_coarser`` for a single ``sub`` position."""
    _fill_from_coarser_fused(self, M, dim, side, idim, ngh, dst,
                             {sub: (ib_c, jb_c)})


def _fill_from_finer_batch(self, M, dim, side, idim, ngh, dst, fi_ib, fi_jb):
    """Batched ``_fill_from_finer``: gather all fine neighbors of every
    coarse block ([nvar, K, n_sub, ...]), combine sub-faces, average pairs,
    and scatter in one shot."""
    ndim = self.ndim
    fine_interior = (cut(-3 * ngh, -ngh, idim) if side == 0
                     else cut(ngh, 3 * ngh, idim))
    dim_keys = list(self.dims.keys())
    active_tv = [slice(None), slice(None), slice(None)]  # nvar, K, n_sub
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            active_tv.append(slice(None))
        else:
            active_tv.append(slice(ngh, -ngh))
    # [nvar, K, n_sub, cells...] with sub columns ordered by sub_idx.
    slabs = M[fine_interior][:, fi_jb][tuple(active_tv)]

    # bit_axes on the batched layout [nvar, K, cz, cy, cx]: cell axis of
    # dim d is (ndim - dims[d]) + 1.
    bit_axes = [ndim - self.dims[d] + 1 for d in dim_keys
                if self.dims[d] != idim]
    if ndim == 1:
        combined = slabs[:, :, 0]
    elif ndim == 2:
        combined = np.concatenate((slabs[:, :, 0], slabs[:, :, 1]),
                                  axis=bit_axes[0])
    else:
        inner_lo = np.concatenate((slabs[:, :, 0], slabs[:, :, 1]),
                                  axis=bit_axes[0])
        inner_hi = np.concatenate((slabs[:, :, 2], slabs[:, :, 3]),
                                  axis=bit_axes[0])
        combined = np.concatenate((inner_lo, inner_hi), axis=bit_axes[1])
    avg_axes = tuple(range(2, ndim + 2))
    coarse_avg = _avg_pairs(combined, avg_axes)
    dest_slice = [slice(None), fi_ib]
    for d in reversed(dim_keys):
        if self.dims[d] == idim:
            dest_slice.append(slice(None))
        else:
            dest_slice.append(slice(ngh, -ngh))
    dst[tuple(dest_slice)] = coarse_avg


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


def Boundaries_scalar(self, M: np.ndarray, dims=None) -> None:
    """Ghost fill for auxiliary scalar fields (trouble indicator, cascade
    index, packed SED alpha): forest exchange as usual, zero-gradient at
    physical boundaries.

    ``dims`` restricts the exchange to a subset of axes (e.g. a single SED
    coupling direction). Default: all mesh dimensions.
    """
    active = self.dims if dims is None else {
        d: self.dims[d] for d in dims if d in self.dims
    }
    for dim in active:
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
        for side in (0, 1):
            groups = self.forest.dev_groups(dim, side, xp)
            my_face_idx = 0 if side == 0 else -1
            fine_face_idx = -1 if side == 0 else 0
            if groups is not None:
                fi_ib, fi_jb = groups["finer"]
                if len(fi_ib):
                    F_faces[indices(my_face_idx, idim)][:, fi_ib] = (
                        F_faces[indices(fine_face_idx, idim)][:, fi_jb[:, 0]]
                    )
                continue
            for ib, block in enumerate(self.forest.blocks):
                entries = block.neighbors[dim][side]
                if not entries or entries[0][1] != FINER:
                    continue
                jb, _rel, _sub = entries[0]
                _view(F_faces, ib)[indices(my_face_idx, idim)] = (
                    _view(F_faces, jb)[indices(fine_face_idx, idim)]
                )
        return

    dim_keys = list(self.dims.keys())
    trans_dims = [d for d in dim_keys[::-1] if d != dim]
    n = self.p + 1
    n_trans = len(trans_dims)

    def _face_to_nested(arr_face: np.ndarray, lead: int = 1) -> np.ndarray:
        # [lead..., ..., NB[d]*n, ...] -> [lead..., NB..., n...]
        shape = list(arr_face.shape[:lead])
        for d in trans_dims:
            shape += [self.NB[d], n]
        reshaped = arr_face.reshape(shape)
        perm = list(range(lead)) + [
            lead + 2 * i for i in range(n_trans)
        ] + [lead + 1 + 2 * i for i in range(n_trans)]
        return reshaped.transpose(perm)

    def _nested_to_face(arr_nested: np.ndarray, lead: int = 1) -> np.ndarray:
        # [lead..., NB..., n...] -> [lead..., ..., NB[d]*n, ...]
        perm = list(range(lead))
        for i in range(n_trans):
            perm += [lead + i, lead + n_trans + i]
        interleaved = arr_nested.transpose(perm)
        out_shape = list(arr_nested.shape[:lead])
        for d in trans_dims:
            out_shape.append(self.NB[d] * n)
        return interleaved.reshape(out_shape)

    for side in (0, 1):
        groups = self.forest.dev_groups(dim, side, xp)
        my_face_idx = 0 if side == 0 else -1
        fine_face_idx = -1 if side == 0 else 0
        if groups is not None:
            # Batched: one gather + one overlap restriction for all coarse
            # faces on this side.
            fi_ib, fi_jb = groups["finer"]
            if not len(fi_ib):
                continue
            fine = F_faces[indices(fine_face_idx, idim)]    # view
            mine = F_faces[indices(my_face_idx, idim)]      # view
            stack = _face_to_nested(fine[:, fi_jb], lead=3)
            coarse_nested = restrict_blocks_overlap_cv(
                stack, R_side_cv, face_ndim
            )
            mine[:, fi_ib] = _nested_to_face(coarse_nested, lead=2)
            continue
        for ib, block in enumerate(self.forest.blocks):
            entries = block.neighbors[dim][side]
            if not entries or entries[0][1] != FINER:
                continue
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
    xp = self.dm.xp
    groups = self.forest.dev_groups(dim, 0, xp)
    if groups is not None:
        # Every same-level interface appears exactly once as some block's
        # LEFT face (side 0) whose neighbor is SAME (the neighbor holds the
        # matching RIGHT face) -- including periodic wraps. One vectorized
        # gather/scatter per dim replaces the per-block loop + seen-set.
        ib_s, jb_s = groups["same"]
        if not len(ib_s):
            return
        left = F_faces[indices(0, idim)]     # view [nvar, Nb, ...]
        right = F_faces[indices(-1, idim)]   # view [nvar, Nb, ...]
        avg = 0.5 * (left[:, ib_s] + right[:, jb_s])
        left[:, ib_s] = avg
        right[:, jb_s] = avg
        return

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
