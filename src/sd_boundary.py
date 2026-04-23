import numpy as np
from sd_simulator import SD_Simulator
from slicing import cut
from slicing import indices
from slicing import indices2
from amr.tree import SAME, FINER, COARSER, BC


def _block_view(M: np.ndarray, ib: int, n_lead: int) -> np.ndarray:
    """Select block ib along the Nb axis.

    Arrays are laid out as [nvar, (nader,) Nb, cells, pts]; ``n_lead`` is the
    number of leading batch axes before Nb (2 for ADER arrays: nvar, nader).
    The returned view matches the old pre-Nb layout, so the existing
    `cut`/`indices`/`indices2` helpers operate on it unchanged.
    """
    return M[(slice(None),) * n_lead + (ib,)]


def store_interfaces(self: SD_Simulator,
                     M: np.ndarray,
                     dim: str) -> None:
    """
    Store flux-point traces at element extremes (0 and -1 along `dim`) into
    MR (from the left-trace) and ML (from the right-trace) at every *interior*
    face of every block. Block-boundary faces (index 0 and -1 of ML/MR) are
    filled separately by apply_BC() once BC_fp has been populated by store_BC().
    """
    shift = self.ndim + self.dims[dim] - 1
    # Ellipsis-based slicing absorbs the Nb axis transparently, so the same
    # assignment fills every block's interior faces in one pass.
    self.MR_fp[dim][cut(None, -1, shift)] = M[indices(0, self.dims[dim])]
    self.ML_fp[dim][cut(1, None, shift)] = M[indices(-1, self.dims[dim])]


def apply_interfaces(self: SD_Simulator,
                     F: np.ndarray,
                     F_fp: np.ndarray,
                     dim: str) -> None:
    """
    Write the Riemann-solved interface flux back into F_fp at element faces.
    Same shape rationale as store_interfaces.
    """
    shift = self.ndim + self.dims[dim] - 1
    F_fp[indices(0, self.dims[dim])] = F[cut(None, -1, shift)]
    F_fp[indices(-1, self.dims[dim])] = F[cut(1, None, shift)]


def store_BC(self: SD_Simulator,
             BC_array: np.ndarray,
             M: np.ndarray,
             dim: str) -> None:
    """
    Populate BC_array[side, ..., ib, ...] for every block ib and side in {0,1}
    by consulting forest.blocks[ib].neighbors[dim][side]:

      - SAME: intra-rank same-level neighbor (including the periodic self-wrap
              when Nb=1 and BC==periodic). Pulls the opposite face of block jb.
      - BC:   domain boundary. Applies the boundary condition (reflective,
              gradfree, ic/eq, pressure).
    """
    idim = self.dims[dim]
    ndim = self.ndim
    # Hydro ADER call sites always pass arrays with leading [nvar, nader]
    # before the Nb axis.
    n_lead = 2
    for ib, block in enumerate(self.forest.blocks):
        for side in (0, 1):
            entries = block.neighbors[dim][side]
            bc_slot = _block_view(BC_array[side], ib, n_lead)
            # For SAME/COARSER/BC there is exactly one entry. For FINER
            # there are 2**(ndim-1) entries; Phase 2c will handle them.
            if len(entries) == 1 and entries[0][1] == SAME:
                jb, _rel, _sub = entries[0]
                src = _block_view(M, jb, n_lead)
                bc_slot[...] = src[indices2(side - 1, ndim, idim)]
            elif len(entries) == 1 and entries[0][1] == BC:
                src = _block_view(M, ib, n_lead)
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
            else:
                # FINER / COARSER: coarse-fine flux-point interface lives
                # in Phase 2c. Static SMR can still build and visualize
                # forests; only time-stepping at coarse-fine faces fails.
                rels = [e[1] for e in entries]
                raise NotImplementedError(
                    f"Coarse-fine face not yet supported "
                    f"(ib={ib}, dim={dim}, side={side}, rels={rels}). "
                    f"Phase 2c.")


def apply_BC(self: SD_Simulator,
             dim: str) -> None:
    """
    Fill the first column of ML_fp and the last column of MR_fp from BC_fp.
    Ellipsis-based slicing absorbs the Nb axis so the same assignment covers
    every block's boundary face.
    """
    shift = self.ndim + self.dims[dim] - 1
    self.ML_fp[dim][indices(0, shift)] = self.BC_fp[dim][0]
    self.MR_fp[dim][indices(-1, shift)] = self.BC_fp[dim][1]


def Boundaries_sd(self: SD_Simulator,
                  M: np.ndarray,
                  dim: str) -> None:
    store_BC(self, self.BC_fp[dim], M, dim)
    store_interfaces(self, M, dim)
    self.Comms_fp(M, dim)
    apply_BC(self, dim)
