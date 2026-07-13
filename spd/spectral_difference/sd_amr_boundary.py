"""Forest-aware SD boundary exchange for block-based (AMR) runs.

Ported from the legacy runtime (`amr-smr-sd` branch). Arrays carry a
meshblock axis right after nvar: [nvar, Nb, cells..., pts...]. Neighbor
relations come from ``forest.blocks[ib].neighbors[dim][side]``:

  - SAME:    pull the opposite face trace of the same-level neighbor.
  - COARSER: prolongate the coarse neighbor's face trace and select our
             sub-face.
  - FINER:   restrict the 2^(ndim-1) fine neighbors' face traces.
  - BC:      apply the domain boundary condition.

``correct_coarse_fine_flux`` restores strict conservation at coarse-fine
faces after the Riemann solve.
"""
from __future__ import annotations

import numpy as np

from spd.numerics.slicing import cut, indices, indices2
from spd.amr.tree import SAME, FINER, COARSER, BC
from spd.amr.transfer import prolongate_block, restrict_blocks_overlap_sp

# Number of leading batch axes before the Nb axis ([nvar] for RK arrays).
N_LEAD = 1


def _block_view(M: np.ndarray, ib: int) -> np.ndarray:
    """Select block ib along the Nb axis (axis N_LEAD)."""
    return M[(slice(None),) * N_LEAD + (ib,)]


def store_interfaces(self, M: np.ndarray, dim: str) -> None:
    """Store flux-point traces at element extremes into MR/ML at every
    interior face. The ellipsis in cut/indices absorbs the Nb axis."""
    shift = self.ndim + self.dims[dim] - 1
    self.MR_fp[dim][cut(None, -1, shift)] = M[indices(0, self.dims[dim])]
    self.ML_fp[dim][cut(1, None, shift)] = M[indices(-1, self.dims[dim])]


def apply_interfaces(self, F: np.ndarray, F_fp: np.ndarray, dim: str) -> None:
    """Write the Riemann-solved interface flux back into F_fp."""
    shift = self.ndim + self.dims[dim] - 1
    F_fp[indices(0, self.dims[dim])] = F[cut(None, -1, shift)]
    F_fp[indices(-1, self.dims[dim])] = F[cut(1, None, shift)]


def store_BC(self, BC_array: np.ndarray, M: np.ndarray, dim: str) -> None:
    """Populate BC_array[side, :, ib, ...] for every block and side from the
    forest neighbor table."""
    idim = self.dims[dim]
    ndim = self.ndim
    LM_prolong = self.dm.LM_prolong
    R_side_sp = self.dm.RS_sp
    xp = self.dm.xp
    for side in (0, 1):
        face_idx = indices2(side - 1, ndim, idim)
        same_jb = self.forest.same_jb[dim][side]
        if same_jb is not None:
            # Fast path: every block's neighbor on this face is SAME-level.
            BC_array[side][...] = xp.take(
                M[face_idx], xp.asarray(same_jb), axis=N_LEAD
            )
            continue
        for ib, block in enumerate(self.forest.blocks):
            entries = block.neighbors[dim][side]
            bc_slot = _block_view(BC_array[side], ib)
            rel0 = entries[0][1]

            if len(entries) == 1 and rel0 == SAME:
                jb, _rel, _sub = entries[0]
                bc_slot[...] = _block_view(M, jb)[face_idx]

            elif len(entries) == 1 and rel0 == BC:
                src = _block_view(M, ib)
                bc_type = self.BC[dim][side]
                if bc_type == "reflective":
                    bc_slot[...] = src[indices2(-side, ndim, idim)]
                    bc_slot[self.vels[idim]] *= -1
                elif bc_type == "gradfree":
                    bc_slot[...] = src[indices2(-side, ndim, idim)]
                elif bc_type in ("ic", "eq"):
                    pass
                elif bc_type == "pressure":
                    src[indices2(-side, ndim, idim)] = bc_slot
                else:
                    raise ValueError(f"Undetermined boundary type: {bc_type}")

            elif len(entries) == 1 and rel0 == COARSER:
                # Fine block facing a coarser neighbor: prolongate the coarse
                # face trace to fine resolution and pick our sub-face.
                jb, _rel, sub = entries[0]
                coarse_trace = _block_view(M, jb)[face_idx]
                if ndim == 1:
                    bc_slot[...] = coarse_trace
                else:
                    prolongated = prolongate_block(
                        coarse_trace, LM_prolong, ndim - 1
                    )
                    bc_slot[...] = prolongated[
                        (slice(None),) * N_LEAD + (sub,)
                    ]

            elif all(e[1] == FINER for e in entries):
                # Coarse block facing finer neighbors: collect the fine face
                # traces (ordered by sub_idx) and restrict.
                if ndim == 1:
                    jb, _rel, _sub = entries[0]
                    bc_slot[...] = _block_view(M, jb)[face_idx]
                else:
                    n_sub = 2 ** (ndim - 1)
                    assert len(entries) == n_sub, (
                        f"expected {n_sub} finer neighbors; got {len(entries)}")
                    traces = [None] * n_sub
                    for (jb, _rel, sub) in entries:
                        traces[sub] = _block_view(M, jb)[face_idx]
                    stack = xp.stack(traces, axis=N_LEAD)
                    bc_slot[...] = restrict_blocks_overlap_sp(
                        stack, R_side_sp, self.dm.cv_to_sp, ndim - 1
                    )

            else:
                rels = [e[1] for e in entries]
                raise NotImplementedError(
                    f"Mixed neighbor relations {rels} not handled "
                    f"(ib={ib}, dim={dim}, side={side}).")


def apply_BC(self, dim: str) -> None:
    """Fill the first column of ML_fp and the last of MR_fp from BC_fp."""
    shift = self.ndim + self.dims[dim] - 1
    self.ML_fp[dim][indices(0, shift)] = self.BC_fp[dim][0]
    self.MR_fp[dim][indices(-1, shift)] = self.BC_fp[dim][1]


def Boundaries(self, M: np.ndarray, dim: str) -> None:
    store_BC(self, self.BC_fp[dim], M, dim)
    self.Comms_fp(M, dim)
    apply_BC(self, dim)


def correct_coarse_fine_flux(self, F_fp: np.ndarray, dim: str) -> None:
    """At coarse-fine faces, overwrite the coarse block's face flux with the
    restriction of the fine-side Riemann fluxes.

    Riemann is nonlinear, so restrict(F_fine) != F(restrict(fine_traces));
    using the fine-side answer on both sides makes the coarse update match
    the sum of the fine updates over the shared area (strict conservation).
    Must be called AFTER apply_interfaces.
    """
    idim = self.dims[dim]
    ndim = self.ndim
    R_side_sp = self.dm.RS_sp
    xp = self.dm.xp
    for ib, block in enumerate(self.forest.blocks):
        for side in (0, 1):
            entries = block.neighbors[dim][side]
            if not entries or entries[0][1] != FINER:
                continue
            fine_face_sel = indices2(side - 1, ndim, idim)
            my_face_sel = indices2(-side, ndim, idim)
            my_view = _block_view(F_fp, ib)

            if ndim == 1:
                (jb, _rel, _sub) = entries[0]
                my_view[my_face_sel] = _block_view(F_fp, jb)[fine_face_sel]
            else:
                n_sub = 2 ** (ndim - 1)
                assert len(entries) == n_sub
                fluxes = [None] * n_sub
                for (jb, _rel, sub) in entries:
                    fluxes[sub] = _block_view(F_fp, jb)[fine_face_sel]
                stack = xp.stack(fluxes, axis=N_LEAD)
                my_view[my_face_sel] = restrict_blocks_overlap_sp(
                    stack, R_side_sp, self.dm.cv_to_sp, ndim - 1
                )
