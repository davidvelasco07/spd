"""
Finite-volume constrained transport for the induction equation (ideal MHD).

1. **IC:** vector potential on edges → :math:`B` at faces (:meth:`_init_B_from_vector_potential_fv`).
2. **Solve edges** (:meth:`solve_edges`): halo **face** :math:`B` via ``Comms_fv`` (same pattern as
   hydro on ghosted data); interpolate :math:`B` (and velocities from ``W``) to the edge lattice and
   :meth:`fill_E_array`; then a 1D upwind along each transverse direction updates the interior of
   ``E`` from neighbouring edge nodes (no SD-style ``store_edges`` / element interfaces).
3. **dBdt:** :meth:`ader_dBdt` (curl of :math:`E_0` onto ``B`` faces).
4. **Update ``B``:** RK / ADER integrator.

Velocities are not advanced inside the induction substeps; they are read from
the existing ghosted primitive state ``dm.W_gh``. Face-centred :math:`B` halos
are updated separately via :meth:`B_Boundaries_fp` before edge interpolation.

Resistive ``nu`` is not implemented; use ``nu=0``.
"""

from __future__ import annotations

import numpy as np

from spd.finite_volume.fv_scheme import FV_Scheme
from spd.numerics.slicing import cut


class InductionFV_Scheme(FV_Scheme):
    """
    FV mesh + induction CT. Primitives live on ``dm.W_cv``; velocities for
    :math:`\\mathbf{E}=-\\mathbf{v}\\times\\mathbf{B}` use ghosted ``dm.W_gh``
    when present, interpolated to each edge tensor layout. :math:`B` uses
    ``dm.B{dim}_fp`` with the same staggering as the SD induction driver.
    """

    _E_NVAR = 5

    # ------------------------------------------------------------------
    # Life cycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Induction-FV startup order.

        We must allocate induction arrays and initialize/sync face-centered ``B``
        before CFL, and ensure ``W_cp`` exists before ``compute_dt``.
        """
        self.compute_positions()
        self.mesh_cv = self.compute_mesh_cv()
        self.post_init()
        self.allocate_arrays(ader=self.ader)
        self._init_B_from_vector_potential_fv()
        self.init_B_fp_boundaries()
        self._sync_b_fp_halos()
        self._sync_W_cp_from_cv()
        self.init_Boundaries()
        self.create_dicts()
        self.compute_dt()

    def post_init(self) -> None:
        """Cell-centered hydro primitives only; :math:`B` is filled after allocation."""
        FV_Scheme.post_init(self)

    def allocate_arrays(self, ader=False):
        super().allocate_arrays(ader=ader)
        self._allocate_induction_ader_arrays(ader=ader)

    def _allocate_induction_ader_arrays(self, ader: bool) -> None:
        ngh = self.Nghc
        dims_map = ["yz", "zx", "xy"]
        for dim, idim in zip(self.Edims, self.Eidims):
            E_ep = self.array(
                self._E_NVAR, dim=dims_map[idim], ader=ader, ngh=ngh
            )
            self.dm.__setattr__(f"E{dim}_ader_ep", E_ep)
        for dim in self.dims:
            self.dm.__setattr__(
                f"B{dim}_fp", self.array(1, dim=dim, ader=False, ngh=ngh)[0]
            )
            if ader:
                self.dm.__setattr__(
                    f"B{dim}_ader_fp",
                    self.array(1, dim=dim, ader=True, ngh=ngh)[0],
                )
            else:
                self.dm.__setattr__(
                    f"B{dim}_ader_fp", self.dm.__getattribute__(f"B{dim}_fp")
                )
        # FV MPI / periodic halos for face-centred B. Slab shapes follow ``cuts`` on
        # ``B{b}_fp`` (transverse extent can differ from cell-centred ``array_BC``).
        ngh = self.Nghc
        for b_dim in self.dims:
            B = self.dm.__getattribute__(f"B{b_dim}_fp")
            for c_dim in self.dims:
                idim = self.dims[c_dim]
                cuts = (cut(-2 * ngh, -ngh, idim), cut(ngh, 2 * ngh, idim))
                sh0 = B[cuts[0]].shape
                sh1 = B[cuts[1]].shape
                if sh0 != sh1:
                    raise RuntimeError(
                        f"BC slab shape mismatch for B{b_dim} comm {c_dim}: {sh0} vs {sh1}"
                    )
                buf = np.ndarray((2,) + sh0)
                self.dm.__setattr__(f"BC_B{b_dim}_fp_{c_dim}", buf)

    def init_B_fp_boundaries(self) -> None:
        """
        Seed ``BC_B{b}_fp_{c}`` from face-centred ``B`` the same way hydro seeds
        ``BC_fp_{dim}`` from ``W_gh`` in :meth:`FV_Scheme.init_Boundaries`.
        """
        ngh = self.Nghc
        for b_dim in self.dims:
            B = self.dm.__getattribute__(f"B{b_dim}_fp")
            for comm_dim in self.dims:
                idim = self.dims[comm_dim]
                BC_buf = self.dm.__getattribute__(f"BC_B{b_dim}_fp_{comm_dim}")
                BC_buf[0][...] = B[cut(None, ngh, idim)]
                BC_buf[1][...] = B[cut(-ngh, None, idim)]

    def B_fp_store_BC(
        self,
        B: np.ndarray,
        b_dim: str,
        comm_dim: str,
        BC_buf: np.ndarray,
        all: bool = True,
    ) -> None:
        """
        Same packing logic as :meth:`FV_Scheme.store_BC` for a single scalar face
        field.  Reflective walls flip the **normal** ``B`` component (when
        ``b_dim == comm_dim``), analogous to flipping normal velocity in hydro.
        """
        idim = self.dims[comm_dim]
        ngh = self.Nghc
        BC_cfg = self.BC[comm_dim]
        cuts = (
            cut(-2 * ngh, -ngh, idim),
            cut(ngh, 2 * ngh, idim),
        )
        for side in [0, 1]:
            if BC_cfg[side] == "periodic":
                BC_buf[side] = B[cuts[side]]
            elif BC_cfg[side] == "reflective":
                if all:
                    BC_buf[side] = B[cuts[1 - side]]
                    if b_dim == comm_dim:
                        BC_buf[side] = -BC_buf[side]
            elif BC_cfg[side] == "gradfree":
                if all:
                    BC_buf[side] = B[cuts[1 - side]]
            elif BC_cfg[side] == "ic":
                pass
            elif BC_cfg[side] == "pressure":
                pass
            elif BC_cfg[side] == "eq":
                if all:
                    BC_buf[side][...] = 0
            else:
                raise RuntimeError("Undetermined boundary type")

    def B_fp_Comms(self, B: np.ndarray, b_dim: str, comm_dim: str) -> None:
        """
        Same call pattern as :meth:`FV_Scheme.Comms`: pass a **full** BC dict
        (all ``comm`` keys for this ``B`` field), matching ``Comms_fv`` usage in
        hydro where ``self.BC_fp`` contains every dimension's slabs.
        """
        BC_all = {
            cd: self.dm.__getattribute__(f"BC_B{b_dim}_fp_{cd}") for cd in self.dims
        }
        self.comms.Comms_fv(
            self.dm,
            B,
            BC_all,
            self.dims[comm_dim],
            comm_dim,
            self.Nghc,
        )

    def B_fp_apply_BC(self, B: np.ndarray, comm_dim: str, BC_buf: np.ndarray) -> None:
        """Same ghost write as :meth:`FV_Scheme.apply_BC`, targeting ``B``."""
        ngh = self.Nghc
        idim = self.dims[comm_dim]
        B[cut(None, ngh, idim)] = BC_buf[0]
        B[cut(-ngh, None, idim)] = BC_buf[1]

    def B_Boundaries_fp(self, b_dim: str, all: bool = True) -> None:
        """
        Hydro-equivalent halo update for ``B{b_dim}_fp``: for each ``comm_dim``,
        ``B_fp_store_BC`` → ``B_fp_Comms`` → ``B_fp_apply_BC`` (same sequence as
        :meth:`FV_Scheme.Boundaries` for ``M``, with per-field ``BC_B{b}_fp_{c}``
        slabs sized from ``cuts`` on that ``B`` layout).
        """
        B = self.dm.__getattribute__(f"B{b_dim}_fp")
        for comm_dim in self.dims:
            BC_buf = self.dm.__getattribute__(f"BC_B{b_dim}_fp_{comm_dim}")
            self.B_fp_store_BC(B, b_dim, comm_dim, BC_buf, all=all)
            self.B_fp_Comms(B, b_dim, comm_dim)
            self.B_fp_apply_BC(B, comm_dim, BC_buf)

    def _sync_b_fp_halos(self, all: bool = True) -> None:
        """Refresh face-centered magnetic ghosts via ``B_Boundaries_fp`` only."""
        for d in self.dims:
            self.B_Boundaries_fp(d, all=all)

    def induction_b_staging_array(self, dim: str):
        """Scratch buffer matching ``B{dim}_fp`` (RK stage / residual storage)."""
        return self.array(1, dim=dim, ngh=self.Nghc)[0]

    def create_dicts(self):
        super().create_dicts()
        self.create_induction_dicts()

    def create_induction_dicts(self):
        self.E_ader_ep = {}
        self.B_ader_fp = {}
        self.B_fp = {}
        for dim in self.dims:
            self.B_ader_fp[dim] = self.dm.__getattribute__(f"B{dim}_ader_fp")
            self.B_fp[dim] = self.dm.__getattribute__(f"B{dim}_fp")
        for dim in self.Edims:
            self.E_ader_ep[dim] = self.dm.__getattribute__(f"E{dim}_ader_ep")

    def other_dims(self, dim: str):
        dims = ["yz", "zx", "xy"]
        if dim in self.dims:
            idim = self.dims[dim]
        else:
            idim = 2
        dim1 = dims[idim][0]
        dim2 = dims[idim][1]
        return dim1, dim2

    def _sync_W_cp_from_cv(self) -> None:
        self.dm.W_cp = self.W_cv.copy()

    # ------------------------------------------------------------------
    # Spatial axes (last axis = x, then y, then z for dims x,y,z)
    # ------------------------------------------------------------------

    def _axis_for_dim(self, odim: str) -> int:
        return self._spatial_axis_for_dim(odim)

    def _len_fp(self, odim: str) -> int:
        return self.N[odim] * self.n[odim] + 1 + 2 * self.Nghc

    def _len_cv(self, odim: str) -> int:
        return self.N[odim] * self.n[odim] + 2 * self.Nghc

    def _avg_to_fp_along(self, a: np.ndarray, axis: int) -> np.ndarray:
        """
        BC-agnostic CV -> FP interpolation on ghosted data.

        Interior faces are cell averages; boundary-most ghost faces use the
        nearest ghost-cell value (one-sided from available ghost layers).
        """
        xp = self.dm.xp
        out_shape = list(a.shape)
        out_shape[axis] += 1
        out = xp.empty(out_shape, dtype=a.dtype)

        sl_out_mid = [slice(None)] * out.ndim
        sl_out_mid[axis] = slice(1, -1)
        sl_l = [slice(None)] * a.ndim
        sl_r = [slice(None)] * a.ndim
        sl_l[axis] = slice(None, -1)
        sl_r[axis] = slice(1, None)
        out[tuple(sl_out_mid)] = 0.5 * (a[tuple(sl_l)] + a[tuple(sl_r)])

        sl_out_l = [slice(None)] * out.ndim
        sl_out_l[axis] = 0
        sl_a_l = [slice(None)] * a.ndim
        sl_a_l[axis] = 0
        out[tuple(sl_out_l)] = a[tuple(sl_a_l)]

        sl_out_r = [slice(None)] * out.ndim
        sl_out_r[axis] = -1
        sl_a_r = [slice(None)] * a.ndim
        sl_a_r[axis] = -1
        out[tuple(sl_out_r)] = a[tuple(sl_a_r)]
        return out

    def _interp_along_if_needed(self, B_arr: np.ndarray, enrich_dim: str) -> np.ndarray:
        ax = self._axis_for_dim(enrich_dim)
        L = B_arr.shape[ax]
        Lfp = self._len_fp(enrich_dim)
        Lcv = self._len_cv(enrich_dim)
        if L == Lfp:
            return B_arr
        if L == Lcv:
            return self._avg_to_fp_along(B_arr, ax)
        raise ValueError(
            f"Unexpected length {L} along {enrich_dim} (expect cv {Lcv} or fp {Lfp})"
        )

    # ------------------------------------------------------------------
    # B from vector potential (edge A, same curl structure as SD induction)
    # ------------------------------------------------------------------

    def _stacked_edge_mesh(self, edge_dir: str) -> np.ndarray:
        """
        Stacked physical coordinates for edges **parallel** to ``edge_dir``,
        matching :meth:`InductionSD_Scheme._init_B_from_vector_potential`: along
        the edge use flux coordinates, orthogonal active directions use cell
        centres. When ``edge_dir`` is not a physical axis (reduced 1D/2D model
        with pseudo-``z`` in ``Edims``), every active axis uses flux coordinates
        (the in-plane edge lattice), as in the SD driver.
        """
        shape = []
        for d in reversed(self.dims.keys()):
            if edge_dir not in self.dims:
                shape.append(self._len_fp(d))
            elif d == edge_dir:
                shape.append(self._len_fp(d))
            else:
                shape.append(self._len_cv(d))
        xp = self.dm.xp
        mesh = xp.zeros((self.ndim,) + tuple(shape), dtype=float)
        for d in self.dims:
            idim = self.dims[d]
            bcast = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * idim
            )
            if edge_dir not in self.dims or d == edge_dir:
                coord = getattr(self.dm, f"{d.upper()}_fp")
            else:
                coord = getattr(self.dm, f"{d.upper()}_cv")
            mesh[idim] = coord[bcast]
        return mesh

    def _forward_diff_along(self, a: np.ndarray, odim: str) -> np.ndarray:
        """First difference along ``odim`` mapped from fp->cv length."""
        ax = self._axis_for_dim(odim)
        sl_hi = [slice(None)] * a.ndim
        sl_lo = [slice(None)] * a.ndim
        sl_hi[ax] = slice(1, None)
        sl_lo[ax] = slice(None, -1)
        return (a[tuple(sl_hi)] - a[tuple(sl_lo)]) / self.h[odim]

    def _field_to_bfp_layout(self, F: np.ndarray, b_dim: str) -> np.ndarray:
        """
        Remap a discrete field to ``B{b_dim}_fp`` staggering using local
        neighbour averaging on ghosted arrays (CV<->FP per axis).
        """
        xp = self.dm.xp
        tgt = self.dm.__getattribute__(f"B{b_dim}_fp")
        cur = xp.asarray(F)
        if cur.shape != tgt.shape and self.ndim == 2 and b_dim in self.dims:
            if cur.shape == (tgt.shape[1], tgt.shape[0]):
                cur = cur.T

        def fp_to_cv(a: np.ndarray, axis: int) -> np.ndarray:
            sl0 = [slice(None)] * a.ndim
            sl1 = [slice(None)] * a.ndim
            sl0[axis] = slice(None, -1)
            sl1[axis] = slice(1, None)
            return 0.5 * (a[tuple(sl0)] + a[tuple(sl1)])

        for odim in self.dims:
            ax = self._axis_for_dim(odim)
            want = int(tgt.shape[ax])
            while cur.shape[ax] < want:
                cur = self._avg_to_fp_along(cur, ax)
            while cur.shape[ax] > want:
                cur = fp_to_cv(cur, ax)
            if cur.shape[ax] != want:
                raise RuntimeError(
                    f"Shape mismatch projecting field to B{b_dim}: axis {odim} "
                    f"got {cur.shape[ax]}, expected {want}"
                )

        if cur.shape != tgt.shape:
            raise RuntimeError(
                f"Shape mismatch projecting field to B{b_dim}: got {cur.shape}, "
                f"expected {tgt.shape}"
            )
        return cur

    def _init_B_from_vector_potential_fv(self) -> None:
        """
        Discrete curl in the same algebraic form as
        :meth:`InductionSD_Scheme._init_B_from_vector_potential`, with ``A``
        sampled on edge-centred FV lattices (``fp`` along the edge direction,
        ``cv`` transverse in 3D; all-``fp`` in the reduced ``Edims`` topology).
        """
        if self.vectorpot_fct is None:
            raise ValueError("vectorpot_fct is required for induction FV init")

        A_ep: dict[str, np.ndarray] = {}
        for ed, jcomp in zip(self.Edims, self.Eidims):
            mesh = self._stacked_edge_mesh(ed)
            raw = self.vectorpot_fct(mesh, jcomp)
            A = self.dm.xp.asarray(raw, dtype=float)
            if A.ndim > self.ndim and A.shape[0] == 1:
                A = A[0]
            A_ep[ed] = A

        xp = self.dm.xp
        for dim in self.dims:
            dim1, dim2 = self.other_dims(dim)
            B_tgt = self.dm.__getattribute__(f"B{dim}_fp")
            acc = xp.zeros_like(B_tgt, dtype=float)
            if dim1 in self.dims and dim2 in A_ep:
                g = self._forward_diff_along(A_ep[dim2], dim1)
                if g.shape != acc.shape:
                    g = self._field_to_bfp_layout(g, dim)
                acc += g
            if dim2 in self.dims and dim1 in A_ep:
                g = self._forward_diff_along(A_ep[dim1], dim2)
                if g.shape != acc.shape:
                    g = self._field_to_bfp_layout(g, dim)
                acc -= g
            B_tgt[...] = acc
            self.dm.__setattr__(f"B{dim}_init_fp", acc.copy())

    # ------------------------------------------------------------------
    # CT: interpolate B and W to edge layout, E = v×B, Riemann, curl
    # ------------------------------------------------------------------

    def _b_face(self, dim: str, ad: bool):
        name = f"B{dim}_ader_fp" if ad else f"B{dim}_fp"
        src = self.dm.__getattribute__(name)
        if ad:
            return src[0, ...]
        return src

    def _interp_b_pair_for_edge(self, dim: str, ad: bool):
        dim1, dim2 = self.other_dims(dim)
        B1 = (
            self._interp_along_if_needed(self._b_face(dim1, ad), dim2)
            if dim1 in self.dims
            else 0
        )
        B2 = (
            self._interp_along_if_needed(self._b_face(dim2, ad), dim1)
            if dim2 in self.dims
            else 0
        )
        return B1, B2

    def _edge_plane_letters(self, Edim: str) -> str:
        m = {"x": "yz", "y": "zx", "z": "xy"}
        return m[Edim]

    def _interp_cv_prims_to_E_plane(self, W_cv: np.ndarray, Edim: str) -> np.ndarray:
        """Cell-centered primitives (nvar, …) to the same spatial layout as E[Edim]."""
        plane = self._edge_plane_letters(Edim)
        out = W_cv
        for odim in self.dims:
            if odim not in plane:
                continue
            ax = self._axis_for_dim(odim)
            if out.shape[ax] == self._len_cv(odim):
                out = self._avg_to_fp_along(out, ax)
        return out

    def compute_vels(self, dim, dim1, dim2, Wc=None):
        """
        Velocities on the same lattice as ``B`` after :meth:`_interp_b_pair_for_edge`:
        ghosted ``dm.W_gh`` and :meth:`_interp_cv_prims_to_E_plane` (FV cell centres → edge fp).
        """
        if Wc is None:
            Wc = self._interp_cv_prims_to_E_plane(self.dm.W_gh, dim)
        v1 = Wc[self.vels[self.dims[dim1]]]
        v2 = Wc[self.vels[self.dims[dim2]]]
        return v1, v2

    def fill_E_array(self, E_ep, B1, B2, dim, ader=False, v1=None, v2=None):
        dim1, dim2 = self.other_dims(dim)
        xp = self.dm.xp
        if isinstance(B1, int) and B1 == 0:
            B1 = xp.zeros_like(E_ep[0])
        if isinstance(B2, int) and B2 == 0:
            B2 = xp.zeros_like(E_ep[0])
        if v1 is None or v2 is None:
            v1, v2 = self.compute_vels(dim, dim1, dim2)
        E_ep[0] = v1 * B2 - v2 * B1
        E_ep[1] = B1
        E_ep[2] = B2
        na = xp.newaxis
        if ader:
            E_ep[3] = v1[na]
            E_ep[4] = v2[na]
        else:
            E_ep[3] = v1
            E_ep[4] = v2

    def E_riemann_solver(self, EL, ER, _v1_):
        xp = self.dm.xp
        v = xp.where(
            xp.abs(EL[_v1_]) > xp.abs(ER[_v1_]), EL[_v1_], ER[_v1_]
        )
        return xp.where(v >= 0, EL, ER)

    def _edges_use_ader(self) -> bool:
        return bool(self._sim.ader)

    def interpolate_L_edge(self, a: np.ndarray, axis: int) -> np.ndarray:
        """
        Left-state interpolation from CV to FP along ``axis``.

        Uses already-ghosted CV values; this is the edge-lattice analogue of
        hydro ``interpolate_L`` side selection.
        """
        xp = self.dm.xp
        out_shape = list(a.shape)
        out_shape[axis] += 1
        out = xp.empty(out_shape, dtype=a.dtype)
        sl0 = [slice(None)] * out.ndim
        sl0[axis] = 0
        sa0 = [slice(None)] * a.ndim
        sa0[axis] = 0
        out[tuple(sl0)] = a[tuple(sa0)]
        slr = [slice(None)] * out.ndim
        slr[axis] = slice(1, None)
        sar = [slice(None)] * a.ndim
        sar[axis] = slice(None, None)
        out[tuple(slr)] = a[tuple(sar)]
        return out

    def interpolate_R_edge(self, a: np.ndarray, axis: int) -> np.ndarray:
        """
        Right-state interpolation from CV to FP along ``axis``.

        Uses already-ghosted CV values; this is the edge-lattice analogue of
        hydro ``interpolate_R`` side selection.
        """
        xp = self.dm.xp
        out_shape = list(a.shape)
        out_shape[axis] += 1
        out = xp.empty(out_shape, dtype=a.dtype)
        sll = [slice(None)] * out.ndim
        sll[axis] = slice(None, -1)
        sal = [slice(None)] * a.ndim
        sal[axis] = slice(None, None)
        out[tuple(sll)] = a[tuple(sal)]
        sl1 = [slice(None)] * out.ndim
        sl1[axis] = -1
        sa1 = [slice(None)] * a.ndim
        sa1[axis] = -1
        out[tuple(sl1)] = a[tuple(sa1)]
        return out

    def _four_state_edge_upwind(self, E_ep: np.ndarray, Edim: str) -> bool:
        """
        Four-state edge upwinding with sequential transverse Riemann solves.

        For edge direction ``Edim`` with transverse directions ``(dim1, dim2)``:
        1) Build four edge states (LL, RL, LR, RR) from native staggering:
           ``B1`` side states along ``dim2``, ``B2`` side states along ``dim1``,
           and velocity side states from ``W_gh``.
        2) Solve Riemann along ``dim1``:
              (LL, RL) -> L ,  (LR, RR) -> R
        3) Solve Riemann along ``dim2``:
              (L, R) -> *
        The full upwinded vector ``[E, B1, B2, v1, v2]`` is retained.
        """
        dim1, dim2 = self.other_dims(Edim)
        if dim1 not in self.dims or dim2 not in self.dims:
            return False

        ax1 = self._axis_for_dim(dim1)
        ax2 = self._axis_for_dim(dim2)

        ad = self._edges_use_ader()
        B1_fp = self._b_face(dim1, ad)  # fp in dim1, cv in dim2
        B2_fp = self._b_face(dim2, ad)  # cv in dim1, fp in dim2

        W = self.dm.W_gh
        v1_cv = W[self.vels[self.dims[dim1]]]
        v2_cv = W[self.vels[self.dims[dim2]]]

        xp = self.dm.xp
        interp = {"L": self.interpolate_L_edge, "R": self.interpolate_R_edge}
        sides = ("L", "R")

        # Respect native staggering:
        # B1 -> edge via dim2 only, B2 -> edge via dim1 only.
        B1_side_dim2 = {side2: interp[side2](B1_fp, ax2) for side2 in sides}
        B2_side_dim1 = {side1: interp[side1](B2_fp, ax1) for side1 in sides}

        U_pair_dim1 = {side1: xp.zeros_like(E_ep) for side1 in sides}
        U_after_dim1 = {}
        for side2 in sides:
            for side1 in sides:
                v1 = interp[side2](interp[side1](v1_cv, ax1), ax2)
                v2 = interp[side2](interp[side1](v2_cv, ax1), ax2)
                self.fill_E_array(
                    U_pair_dim1[side1],
                    B1_side_dim2[side2],
                    B2_side_dim1[side1],
                    Edim,
                    ader=ad,
                    v1=v1,
                    v2=v2,
                )
            # First transverse solve (dim1) for fixed dim2 side.
            U_after_dim1[side2] = self.E_riemann_solver(
                U_pair_dim1["L"], U_pair_dim1["R"], 3
            )

        # Second transverse solve (dim2) combines the two intermediates.
        U_star = self.E_riemann_solver(U_after_dim1["L"], U_after_dim1["R"], 4)

        n = self.Nghc
        sl_out = [slice(None)] * E_ep.ndim
        for odim in self.dims:
            aod = self._axis_for_dim(odim)
            nmax = int(E_ep.shape[aod])
            sl_out[aod] = slice(n, nmax - n)
        E_ep[tuple(sl_out)] = U_star[tuple(sl_out)]
        return True

    def solve_edges(self, ader_iter):
        """
        Induction edge solve for FV:

        1. Face-centred ``B`` halos via :meth:`B_Boundaries_fp` (``Comms_fv`` on ``B``).
        2. Interpolate ``B`` and read ``v`` from ``W``, then :meth:`fill_E_array` on ``E``.
        3. Four-state edge upwinding with sequential transverse Riemann solves
           on the full edge vector ``[E, B1, B2, v1, v2]``.

        ``E`` is local working state only (no E halo communications); interpolation
        operates directly on already-ghosted ``B`` and ``W_gh``.

        ``ader_iter`` is accepted for API parity with the SD driver and is unused here.
        """
        _ = ader_iter
        ad = self._edges_use_ader()
        self._sync_b_fp_halos()
        for dim in self.Edims:
            B1, B2 = self._interp_b_pair_for_edge(dim, ad)
            self.fill_E_array(self.E_ader_ep[dim], B1, B2, dim, ader=ad)
            ok = self._four_state_edge_upwind(self.E_ader_ep[dim], dim)
            if not ok:
                raise RuntimeError(
                    f"Edge upwind directions for E{dim} are not active spatial dims: "
                    f"{self.other_dims(dim)} vs active {tuple(self.dims.keys())}"
                )

    def ader_string(self) -> str:
        # FV arrays are flat in space: (nader, Nz, Ny, Nx) in 3D.
        if self.ndim == 3:
            return "zyx"
        if self.ndim == 2:
            return "yx"
        return "x"

    def _E0(self, E_ep: np.ndarray, ad: bool) -> np.ndarray:
        Ez = E_ep[0]
        # In ADER mode keep the leading nader axis.
        if ad:
            return Ez
        return Ez

    def _diff_E0_to_bface(self, E_ep, E_edim: str, deriv_dim: str, b_dim: str, ad: bool):
        """
        Donor-cell discrete derivative from edge ``E0`` to face ``B`` layout.

        FV CT needs the same topological operator used by the constrained-
        transport update: first difference along the derivative direction
        (``fp -> cv`` on that axis), leaving the other axes untouched.
        """
        _ = E_edim
        E0 = self._E0(E_ep, ad)
        ax = self._axis_for_dim(deriv_dim)
        sl_hi = [slice(None)] * E0.ndim
        sl_lo = [slice(None)] * E0.ndim
        sl_hi[ax] = slice(1, None)
        sl_lo[ax] = slice(None, -1)
        dE = (E0[tuple(sl_hi)] - E0[tuple(sl_lo)]) / self.h[deriv_dim]
        tgt = self.B_ader_fp[b_dim] if ad else self.B_fp[b_dim]
        if dE.shape != tgt.shape:
            raise RuntimeError(
                f"Discrete curl shape mismatch for B{b_dim}: got {dE.shape}, "
                f"expected {tgt.shape} from E derivative along {deriv_dim}"
            )
        return dE

    def ader_dBdt(self, dim):
        ad = self._edges_use_ader()
        dim1, dim2 = self.other_dims(dim)
        Btgt = self.B_ader_fp[dim] if ad else self.B_fp[dim]
        dBdt = self.dm.xp.zeros_like(Btgt)
        if dim1 in self.Edims:
            dBdt += self._diff_E0_to_bface(
                self.E_ader_ep[dim1], dim1, dim2, dim, ad
            )
        if dim2 in self.Edims:
            dBdt -= self._diff_E0_to_bface(
                self.E_ader_ep[dim2], dim2, dim1, dim, ad
            )
        return dBdt * self.dt

    def dBdt_dim(self, dim):
        dt0 = self.dt
        if dt0 == 0:
            dt0 = 1.0
        return self.ader_dBdt(dim) / dt0

    def ader_predictor(self, prims: bool = False) -> None:
        if not self._sim.ader:
            return
        xp = self.dm.xp
        _b0 = self.dm.__getattribute__(f"B{list(self.dims.keys())[0]}_fp")
        na = (xp.newaxis,) + (slice(None),) * _b0.ndim
        for dim in self.dims:
            Bf = self.dm.__getattribute__(f"B{dim}_fp")
            Bad = self.dm.__getattribute__(f"B{dim}_ader_fp")
            Bad[...] = Bf[na]

        for ader_iter in range(self.m + 1):
            self.solve_edges(ader_iter)
            if self.nu > 0:
                self.add_nabla_terms()
            if ader_iter < self.m:
                for dim in self.dims:
                    s = self.ader_string()
                    Bad = self.dm.__getattribute__(f"B{dim}_ader_fp")
                    Bf = self.B_fp[dim]
                    inv = xp.asarray(self.dm.invader)
                    Bad[...] = xp.einsum(
                        f"np,p{s}->n{s}", inv, self.ader_dBdt(dim)
                    )
                    Bad[...] = Bf[na] - Bad

    def ader_update(self):
        if not self._sim.ader:
            return
        xp = self.dm.xp
        wtp = xp.asarray(self.dm.w_tp)
        for dim in self.dims:
            s = self.ader_string()
            dBdt = xp.einsum(f"t,t{s}->{s}", wtp, self.ader_dBdt(dim))
            self.B_fp[dim] -= dBdt

    def add_nabla_terms(self):
        if self.nu <= 0:
            return
        raise NotImplementedError(
            "Magnetic resistivity (nu) for InductionFV_Scheme is not implemented yet."
        )

    def compute_B2(self):
        xp = self.dm.xp
        na = xp.newaxis
        acc = None
        for d in self.dims:
            Bd = self.dm.__getattribute__(f"B{d}_fp")
            bc = self.compute_sp_from_fp(Bd[na], d)[0]
            acc = bc * bc if acc is None else acc + bc * bc
        return acc

    def compute_dt(self) -> None:
        """Match :meth:`induction_sd_scheme.InductionSD_Scheme.compute_dt` line-for-line (xp-aware)."""
        xp = self.dm.xp
        W = self.dm.W_cp
        vel = xp.abs(W[0]).copy()
        for idim in range(1, self.ndim):
            vel = vel + xp.abs(W[idim])
        c_max = xp.max(vel)
        c_max = xp.maximum(c_max, 1e-30)
        cmax_f = c_max.item() if hasattr(c_max, "item") else float(c_max)
        h = self.h_min / (self.p + 1)
        dt = h / cmax_f
        if self.nu > 0:
            dt_nu = (0.25 * self.h_min / (self.p + 1)) ** 2 / self.nu
            dt = min(dt, float(dt_nu))
        dt = self.comms.reduce_min(dt)
        dt_s = dt.item() if hasattr(dt, "item") else float(dt)
        self.dt = self.cfl_coeff * dt_s
        self._sim.dt = self.dt

    def post_update(self):
        """Hydro post-update plus primitive ghost/state alignment for next edge solve."""
        FV_Scheme.post_update(self)
        sl = (slice(None),) + tuple(
            slice(self.Nghc, -self.Nghc) for _ in range(self.ndim)
        )
        # ``FV_Scheme.post_update`` sets ``self.W_cv`` from ``U_cv``; keep ghosted
        # ``dm.W_gh`` and ``dm`` cell storage aligned (``dm.W_cv`` may not alias
        # ``W_gh`` after host/device conversion).
        self.dm.W_gh[sl] = self.W_cv
        self.dm.W_cv[...] = self.W_cv
        self._sync_W_cp_from_cv()
