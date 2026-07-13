"""Block-based (AMR-capable) Spectral Difference scheme.

Arrays carry a meshblock axis right after nvar:
    [nvar, Nb, (Nz,) (Ny,) Nx, (pz,) (py,) px]
where Nz/Ny/Nx are the per-meshblock element counts (``sim.NB``) and Nb is
``forest.Nblocks``. Blocks may live at different refinement levels; boundary
exchange, flux-divergence metrics, and the CFL all account for per-block h.

Supports dynamic AMR: ``tag_blocks`` + ``adapt`` mutate the forest, then
reallocate every Nb-sized array and remap the solution with prolongation /
overlap-aware restriction.

RK time integrators only (the ADER time-node axis is not carried).
"""

import numpy as np

from spd.spectral_difference.sd_scheme import SD_Scheme
from spd.spectral_difference import sd_amr_boundary as bc
from spd.numerics.polynomials import quadrature_mean
from spd.amr.transfer import (
    build_transfer_matrices,
    build_overlap_restrict_matrices,
    prolongate_block,
    restrict_blocks_overlap_sp,
)


class SD_AMR_Scheme(SD_Scheme):

    def __init__(self, sim, riemann_solver="llf", soe=None):
        super().__init__(sim, riemann_solver=riemann_solver, soe=soe)
        # AMR: coarse <-> fine solution-point transfer operators.
        self.dm.LM_prolong, self.dm.LM_restrict = build_transfer_matrices(
            self.sp["x"]
        )
        # Overlap-aware restriction operators, stacked [2, n, n] so the
        # GPUDataManager migrates them with backend switches (AGENTS.md rule:
        # everything consumed in perform_update paths lives on dm).
        RS_sp, RS_cv = build_overlap_restrict_matrices(self.sp["x"], self.fp["x"])
        self.dm.RS_sp = np.stack(RS_sp, axis=0)
        self.dm.RS_cv = np.stack(RS_cv, axis=0)

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        if self.ader:
            raise ValueError("SD_AMR_Scheme supports RK integrators only.")
        if self.potential or self.WB or self.viscosity or self.thdiffusion:
            raise NotImplementedError(
                "potential/WB/viscosity/thdiffusion not supported on the "
                "block-based AMR scheme yet."
            )
        self.mesh_cv = self.compute_mesh_cv()   # rank-global; kept for compat
        self.compute_positions()
        self._refresh_block_metrics()
        self.post_init()
        self._alloc_dual_layout_buffers()
        self.allocate_arrays(ader=False)
        self.init_Boundaries()
        self.create_dicts()
        self.compute_dt()

    def compute_mesh_cv_block(self, block) -> np.ndarray:
        """Per-meshblock control-volume mesh located within the block's
        physical extent (element size ``block.h``)."""
        Nghe = self.Nghe
        Ns = [self.NB[dim] + 2 * Nghe for dim in self.dims]
        shape = (self.ndim,) + tuple(Ns[::-1]) + (self.p + 2,) * self.ndim
        mesh_cv = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h = block.h[dim]
            lo, hi = block.lim[dim]
            length = (hi - lo) + 2 * Nghe * h
            shape1 = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (self.ndim + idim)
            )
            shape2 = (
                (None,) * (2 * self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (idim)
            )
            mesh_cv[idim] = (
                lo + (np.arange(N)[shape1] + self.fp[dim][shape2]) * length / N
                - h
            )
        return mesh_cv

    def post_init(self) -> None:
        nvar = self.nvar
        ngh = self.Nghe
        # Each meshblock's IC values are computed on that block's own
        # physical mesh, so multi-block / multi-level grids see
        # position-dependent ICs correctly.
        W_gh = self.array_sp(ngh=ngh)
        for ib, block in enumerate(self.forest.blocks):
            mesh_cv_b = self.compute_mesh_cv_block(block)
            for var in range(nvar):
                W_gh[var, ib] = quadrature_mean(
                    mesh_cv_b, self.init_fct, self.ndim, self.p, var
                )
        self.W_gh = W_gh
        self.W_init_cv = self.active_region(W_gh)
        self.dm.W_cv = self.W_init_cv.copy()
        self.dm.W_sp = self.compute_sp_from_cv(self.dm.W_cv)
        self.dm.U_sp = self.compute_conservatives(self.dm.W_sp)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)

    # ----------------------------------------------------------------
    # Per-block metrics
    # ----------------------------------------------------------------

    def _refresh_block_metrics(self) -> None:
        """Cache per-block 1/h[dim] (broadcastable over cells+pts) and the
        forest-wide h_min. Stored on dm for GPU migration; create_dicts
        refreshes the dict entries against dm's current location."""
        self._inv_h_block = {}
        for dim in self.dims:
            arr = np.array([1.0 / b.h[dim] for b in self.forest.blocks])
            shape = (1, -1) + (1,) * (2 * self.ndim)
            self.dm.__setattr__(f"inv_h_block_{dim}", arr.reshape(shape))
            self._inv_h_block[dim] = self.dm.__getattribute__(
                f"inv_h_block_{dim}"
            )
        self.h_min = min(
            b.h[d] for b in self.forest.blocks for d in self.dims
        )

    def create_dicts(self):
        super().create_dicts()
        for dim in self.dims:
            self._inv_h_block[dim] = self.dm.__getattribute__(
                f"inv_h_block_{dim}"
            )

    # ----------------------------------------------------------------
    # Array allocation (block-based layout)
    # ----------------------------------------------------------------

    def array(self, px, py, pz, ngh=0, ader=False, nvar=None) -> np.ndarray:
        assert not ader, "AMR scheme arrays carry no ADER axis (RK only)."
        if nvar is None:
            nvar = self.nvar
        shape = [nvar, self.forest.Nblocks]
        N = [self.NB[dim] + 2 * ngh for dim in self.dims][::-1]
        p = [px, py, pz][: self.ndim][::-1]
        return np.ndarray(shape + N + p)

    def array_RS(self, dim="x", dim2=None, ader=False) -> np.ndarray:
        assert not ader
        shape = [self.nvar, self.forest.Nblocks]
        N = []
        for odim in self.dims:
            N.append(self.NB[odim] + (odim == dim))
        shape += N[::-1]
        if self.ndim > 2:
            if (dim2 == "x") or (dim2 == "y" and dim == "x"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        if self.ndim > 1:
            if (dim2 == "z") or (dim2 == "y" and dim == "z"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        return np.ndarray(shape)

    def array_BC(self, dim="x", dim2=None, ader=False) -> np.ndarray:
        assert not ader
        shape = [2, self.nvar, self.forest.Nblocks]
        if self.Z:
            if dim == "x" or dim == "y":
                shape += [self.NB["z"]]
        if self.Y:
            if dim == "x" or dim == "z":
                shape += [self.NB["y"]]
        if dim == "y" or dim == "z":
            shape += [self.NB["x"]]
        if self.Z:
            if dim2 == "x" or (dim2 == "y" and dim == "x"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        if self.Y:
            if dim2 == "z" or (dim2 == "y" and dim == "z"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        return np.ndarray(shape)

    # ----------------------------------------------------------------
    # Layout helpers (block-preserving FV transposes, for I/O and vis)
    # ----------------------------------------------------------------

    def block_to_fv(self, M_block):
        """Per-block SD layout -> per-block flat FV layout.

        Input  (ndim=2): [..., NyB, NxB, py, px]
        Output (ndim=2): [..., NyB*py, NxB*px]
        Leading axes are absorbed.
        """
        s = M_block.shape
        if self.ndim == 1:
            return M_block.reshape(s[:-2] + (s[-2] * s[-1],))
        elif self.ndim == 2:
            nd = M_block.ndim
            perm = tuple(range(nd - 4)) + (nd - 4, nd - 2, nd - 3, nd - 1)
            return np.transpose(M_block, perm).reshape(
                s[:-4] + (s[-4] * s[-2], s[-3] * s[-1])
            )
        else:
            lead = M_block.ndim - 6
            perm = tuple(range(lead)) + (lead, lead + 3,
                                         lead + 1, lead + 4,
                                         lead + 2, lead + 5)
            return np.transpose(M_block, perm).reshape(
                s[:-6] + (s[-6] * s[-3], s[-5] * s[-2], s[-4] * s[-1])
            )

    def transpose_to_fv(self, M):
        """[nvar, Nb, cells..., pts...] -> [nvar, Nb, flat FV cells...]."""
        return self.block_to_fv(M)

    def transpose_to_sd(self, M):
        """Inverse of transpose_to_fv; keeps the Nb axis intact."""
        nvar, Nb = M.shape[0], M.shape[1]
        shape_ordered = [nvar, Nb]
        for dim in list(self.dims.keys())[::-1]:   # z, y, x outer->inner
            shape_ordered += [self.NB[dim], self.n[dim]]
        reshaped = M.reshape(shape_ordered)
        if self.ndim == 1:
            return reshaped
        elif self.ndim == 2:
            return np.transpose(reshaped, (0, 1, 2, 4, 3, 5))
        else:
            return np.transpose(reshaped, (0, 1, 2, 4, 6, 3, 5, 7))

    def _alloc_dual_layout_buffers(self) -> None:
        """Persistent SD- and FV-layout cell-average buffers.

        The SD<->FV layout switches (used by the MUSCL fallback) swap
        ``dm.U_cv``/``dm.W_cv`` between these fixed arrays instead of
        allocating transposed copies each time (same contract as the
        single-grid SD_Scheme). Must be re-run after every forest change.
        """
        self.dm.U_cv_sd = self.dm.U_cv
        self.dm.W_cv_sd = self.dm.W_cv
        # transpose->reshape copies for ndim>1; in 1D the layouts coincide
        # and the "copy" is a view, which is exactly right there.
        self.dm.U_cv_fv = self.transpose_to_fv(self.dm.U_cv)
        self.dm.W_cv_fv = self.transpose_to_fv(self.dm.W_cv)

    def compute_cv_from_sp_fv(self, M_sp, out=None) -> np.ndarray:
        """sp -> cv projection emitted in the block-preserving FV layout.

        The single-grid SD_Scheme fuses projection and transpose into one
        einsum; the block layout keeps them separate (projection over the
        trailing point axes, then a strided per-block transpose).
        """
        res = self.transpose_to_fv(self.compute_cv_from_sp(M_sp))
        if out is None:
            return res
        out[...] = res
        return out

    def compute_sp_from_cv_fv(self, M_fv, out=None) -> np.ndarray:
        """Block-preserving FV layout -> cv -> sp projection."""
        res = self.compute_sp_from_cv(self.transpose_to_sd(M_fv))
        if out is None:
            return res
        out[...] = res
        return out

    # ----------------------------------------------------------------
    # Positions / geometry
    # ----------------------------------------------------------------

    def compute_positions(self):
        na = np.newaxis
        ngh = self.Nghc
        for dim in self.dims:
            idim = self.dims[dim]
            # Rank-global solution points (level-0 spacing; kept for BCs
            # that sample positions, e.g. doublemach).
            sp = self.lim[dim][0] + (
                np.arange(self.N[dim])[:, na] + self.sp[dim][na, :]
            ) * self.h[dim]
            self.dm.__setattr__(
                f"{dim.upper()}_sp", sp.reshape(self.N[dim], self.n[dim])
            )
            # Rank-global flux points.
            fp = np.ndarray((self.N[dim] * self.n[dim] + ngh * 2 + 1))
            fp[ngh:-ngh] = self.h[dim] * np.hstack(
                (
                    np.arange(self.N[dim]).repeat(self.n[dim])
                    + np.tile(self.fp[dim][:-1], self.N[dim]),
                    self.N[dim],
                )
            )
            fp[:ngh] = -fp[(ngh + 1):(2 * ngh + 1)][::-1]
            fp[-ngh:] = fp[-(ngh + 1)] + fp[ngh + 1:2 * ngh + 1]
            self.dm.__setattr__(f"{dim.upper()}_fp", fp)
            self.faces[dim] = fp
            cv = 0.5 * (fp[1:] + fp[:-1])
            self.dm.__setattr__(f"{dim.upper()}_cv", cv)
            self.centers[dim] = cv
            # Per-block face/center spacings at level 0 (per-level scaling
            # enters through _inv_h_block / h_fp_nb when FV lands here).
            fp_b = np.ndarray((self.NB[dim] * self.n[dim] + ngh * 2 + 1))
            fp_b[ngh:-ngh] = self.h[dim] * np.hstack((
                np.arange(self.NB[dim]).repeat(self.n[dim])
                + np.tile(self.fp[dim][:-1], self.NB[dim]),
                self.NB[dim],
            ))
            fp_b[:ngh] = -fp_b[(ngh + 1):(2 * ngh + 1)][::-1]
            fp_b[-ngh:] = fp_b[-(ngh + 1)] + fp_b[ngh + 1:2 * ngh + 1]
            cv_b = 0.5 * (fp_b[1:] + fp_b[:-1])
            h_fp = (fp_b[1:] - fp_b[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_fp", h_fp)
            self.h_fp[dim] = h_fp
            h_cv = (cv_b[1:] - cv_b[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_cv", h_cv)
            self.h_cv[dim] = h_cv

    # ----------------------------------------------------------------
    # Flux divergence with per-block metric
    # ----------------------------------------------------------------

    def compute_sp_from_dfp_x(self, Fx, ader=False):
        return (self.compute_sp_from_dfp(Fx, "x", ader=ader)
                * self._inv_h_block["x"])

    def compute_sp_from_dfp_y(self, Fy, ader=False):
        return (self.compute_sp_from_dfp(Fy, "y", ader=ader)
                * self._inv_h_block["y"])

    def compute_sp_from_dfp_z(self, Fz, ader=False):
        return (self.compute_sp_from_dfp(Fz, "z", ader=ader)
                * self._inv_h_block["z"])

    # ----------------------------------------------------------------
    # Spatial operator with forest-aware boundaries
    # ----------------------------------------------------------------

    def solve_faces(self, M, prims=False, ader=False) -> None:
        assert not ader
        amr = self.forest.max_level > 0
        for key in self.idims:
            dim = self.idims[key]
            vels = np.roll(self.vels, -key)
            self.M_fp[dim][...] = self.compute_fp_from_sp(M, dim, ader=False)
            bc.Boundaries(self, self.M_fp[dim], dim)
            self.compute_fluxes(self.F_fp[dim], self.M_fp[dim], vels, prims)
            bc.store_interfaces(self, self.M_fp[dim], dim)
            F = self.riemann_solver(
                self.ML_fp[dim],
                self.MR_fp[dim],
                self.MR_fp[dim],
                vels,
                self._p_,
                self.gamma,
                self.min_c2,
                prims,
                npassive=self.npassive,
                thdiffusion=self.thdiffusion,
                _t_=self._t_,
                min_rho=self.min_rho,
            )
            bc.apply_interfaces(self, F, self.F_fp[dim], dim)
            # AMR conservation: at coarse-fine faces the two sides must see
            # one shared (fine-restricted) flux.
            if amr:
                bc.correct_coarse_fine_flux(self, self.F_fp[dim], dim)

    # ----------------------------------------------------------------
    # Dynamic AMR: tagging, adaptation, state remap
    # ----------------------------------------------------------------

    def tag_blocks(self, refine_fn=None, derefine_fn=None, max_level=None):
        """Tag blocks for refinement / derefinement via user predicates.

        refine_fn(block, W_block) -> bool : True to refine this block.
                 Blocks already at max_level are skipped.
        derefine_fn(parent_logical, sibling_blocks, sibling_W) -> bool :
                 True to merge this sibling group. Only called for groups
                 where all 2**ndim siblings exist at the same level.
        """
        W_sp = self.compute_primitives(self.dm.U_sp)
        to_refine = []
        if refine_fn is not None:
            for ib, block in enumerate(self.forest.blocks):
                if max_level is not None and block.level >= max_level:
                    continue
                if refine_fn(block, W_sp[:, ib]):
                    to_refine.append(ib)

        to_derefine = []
        if derefine_fn is not None:
            groups = {}
            for ib, b in enumerate(self.forest.blocks):
                if b.level == 0:
                    continue
                pl_key = (b.level, tuple(c // 2 for c in b.logical))
                groups.setdefault(pl_key, []).append(ib)
            n_sib = 2 ** self.ndim
            for (lvl, parent_logical), ibs in groups.items():
                if len(ibs) != n_sib:
                    continue
                sibs = [self.forest.blocks[i] for i in ibs]
                sib_W = [W_sp[:, i] for i in ibs]
                if derefine_fn(parent_logical, sibs, sib_W):
                    to_derefine.append(ibs)
        return to_refine, to_derefine

    def adapt(self, to_refine=None, to_derefine=None) -> None:
        """Apply refinement / derefinement and remap the solution onto the
        new block layout (2:1 balance enforced automatically)."""
        to_refine = list(to_refine or [])
        to_derefine = list(to_derefine or [])
        if not to_refine and not to_derefine:
            return
        snapshot = self._snapshot_U_sp()
        # Resolve ibs to block objects so mutation order is irrelevant.
        refine_refs = [self.forest.blocks[i] for i in to_refine]
        derefine_refs = [[self.forest.blocks[i] for i in sibs]
                         for sibs in to_derefine]
        for block in refine_refs:
            self.forest.refine_block(self.forest.blocks.index(block))
        for sibs in derefine_refs:
            sib_ibs = [self.forest.blocks.index(b) for b in sibs]
            self.forest.derefine_block(sib_ibs)
        self.forest.enforce_2to1_balance()
        self._arrays_realloc()
        self._transfer_from_snapshot(snapshot)
        self.init_Boundaries()
        # dt may shrink: new fine blocks have smaller h.
        self.compute_dt()

    def _arrays_realloc(self) -> None:
        """Re-allocate every Nb-sized array after a forest change."""
        self.dm.W_cv = self.array_sp()
        self.dm.W_sp = self.array_sp()
        self.dm.U_sp = self.array_sp()
        self.dm.U_cv = self.array_sp()
        self._alloc_dual_layout_buffers()
        # Flux/RS/BC buffers + RK stage arrays (via integrator).
        self.allocate_arrays(ader=False)
        self._refresh_block_metrics()
        self.create_dicts()

    def _snapshot_U_sp(self) -> dict:
        """Snapshot U_sp keyed by (level, logical)."""
        return {
            (b.level, b.logical): self.dm.U_sp[:, ib].copy()
            for ib, b in enumerate(self.forest.blocks)
        }

    def _transfer_from_snapshot(self, snapshot: dict) -> None:
        """Populate dm.U_sp for every block in the new forest via direct
        copy / prolongation / restriction; refresh derived arrays."""
        ndim = self.ndim
        dim_keys = list(self.dims.keys())
        xp = self.dm.xp
        LM_p = self.dm.LM_prolong
        R_side_sp = self.dm.RS_sp
        for ib, block in enumerate(self.forest.blocks):
            key = (block.level, block.logical)
            if key in snapshot:
                self.dm.U_sp[:, ib] = snapshot[key]
                continue
            # Prolongation from parent.
            parent_logical = tuple(c // 2 for c in block.logical)
            parent_key = (block.level - 1, parent_logical)
            if parent_key in snapshot:
                children_U = prolongate_block(snapshot[parent_key], LM_p, ndim)
                sub_idx = 0
                for k, d in enumerate(dim_keys):
                    offset = block.logical[k] - 2 * parent_logical[k]
                    sub_idx += (1 << k) * offset
                self.dm.U_sp[:, ib] = children_U[:, sub_idx]
                continue
            # Restriction from children.
            child_keys = []
            for sub_idx in range(2 ** ndim):
                child_logical = tuple(
                    2 * block.logical[k] + ((sub_idx >> k) & 1)
                    for k in range(ndim)
                )
                child_keys.append((block.level + 1, child_logical))
            if all(k in snapshot for k in child_keys):
                stack = xp.stack([snapshot[k] for k in child_keys], axis=1)
                self.dm.U_sp[:, ib] = restrict_blocks_overlap_sp(
                    stack, R_side_sp, self.dm.cv_to_sp, ndim
                )
                continue
            raise RuntimeError(
                f"No source data for block {key} (ib={ib}): neither the "
                f"block itself, its parent {parent_key}, nor its full set "
                f"of children was in the snapshot."
            )
        self.dm.W_sp[...] = self.compute_primitives(self.dm.U_sp)
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)
