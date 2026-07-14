"""
Standalone Fallback scheme using MUSCL or MUSCL-Hancock reconstruction.

Can operate in two modes:
  1. Standalone:  acts as the sole semi-discrete scheme for a Simulator,
     providing a robust low-order FV method.
  2. Fallback:    paired with a primary high-order scheme (SD or FV).
     After the primary scheme produces high-order fluxes, the fallback
     detects troubled cells and blends in MUSCL fluxes to maintain
     stability near discontinuities.
"""

import numpy as np

from spd.schemes.scheme import SemiDiscreteScheme
from spd.finite_volume.fv_scheme import FV_Scheme
from .trouble_detection import detect_troubles
from spd.numerics.slicing import cut, indices, indices2, crop_fv
from spd.numerics.polynomials import quadrature_mean
from spd.runtime.gpu import CUPY_AVAILABLE, is_gpu_array

if CUPY_AVAILABLE:
    import cupy as cp

    # Fused convex flux blend: F = theta*FB + (1-theta)*F in one kernel
    # (theta broadcasts over the variable axis).  The endpoints select
    # instead of blending so a non-finite flux on the discarded side (e.g.
    # HLLD fed inadmissible predictor states) cannot poison the result.
    blend_k = cp.ElementwiseKernel(
        "T f, T fb, T theta",
        "T out",
        "out = (theta >= 1) ? fb : ((theta <= 0) ? f : f + theta * (fb - f));",
        "fb_blend_k",
    )


def blend_fluxes(F, F_FB, theta):
    """Convex blend theta*F_FB + (1-theta)*F (dispatcher).

    Out of place, since F may be a view of the primary scheme's fluxes.
    Scalar theta (godunov mode) takes the generic path.  The endpoints
    (theta = 0, 1) select the corresponding flux exactly, so a non-finite
    value on the fully discarded side does not propagate.
    """
    if is_gpu_array(theta):
        return blend_k(F, F_FB, theta)
    if np.isscalar(theta):
        if theta >= 1:
            return F_FB.copy()
        if theta <= 0:
            return F.copy()
        return theta * F_FB + (1 - theta) * F
    return np.where(
        theta >= 1,
        F_FB,
        np.where(theta <= 0, F, theta * F_FB + (1 - theta) * F),
    )


class FallbackScheme(FV_Scheme):
    """
    MUSCL / MUSCL-Hancock fallback scheme.

    Inherits the full FV spatial operator from FV_Scheme and adds
    trouble detection and flux blending when used with a primary
    high-order scheme.

    Parameters
    ----------
    sim : Simulator
        Parent simulator providing shared state.
    primary : SemiDiscreteScheme or None
        The primary high-order scheme (e.g., SD_Scheme). If None,
        the fallback operates in standalone mode.
    riemann_solver : str
        Riemann solver for the FV fluxes.
    slope_limiter : str
        Slope limiter name ('minmod', 'moncen').
    predictor : bool
        If True use MUSCL-Hancock, else plain MUSCL.
    FB : bool
        Enable flux blending (trouble detection + blending).
    tolerance : float
        Tolerance for the NAD check.
    SED : bool
        Enable smooth extrema detection.
    NAD : str
        NAD tolerance mode ('' for relative, 'delta' for range-scaled).
    NAD_neighbors : str
        DMP stencil for the NAD bounds: '1st' (von Neumann / face neighbors)
        or '2nd' (Moore / box neighborhood, includes diagonal neighbors).
    PAD : bool
        Enable physically admissible detection.
    blending : bool
        Spread trouble indicators to neighbors.
    min_rho, max_rho, min_P : float
        Bounds for PAD checks.
    godunov : bool
        If True, use pure Godunov (no blending, theta=1 everywhere).
    limiting_variables : list
        Variable indices used for NAD check (default: density and pressure).
    """

    def __init__(
        self,
        sim,
        primary=None,
        riemann_solver="llf",
        slope_limiter="minmod",
        scheme="MUSCL",
        FB=False,
        tolerance=1e-5,
        SED=True,
        NAD="",
        NAD_neighbors="2nd",
        PAD=True,
        blending=True,
        min_rho=1e-10,
        max_rho=1e10,
        min_P=1e-10,
        godunov=False,
        limiting_variables=None,
        max_revs=5,
    ):
        super().__init__(
            sim,
            riemann_solver=riemann_solver,
            slope_limiter=slope_limiter,
            scheme=scheme,
        )
        self.primary = primary

        # Trouble detection parameters
        self.FB = FB
        self.tolerance = tolerance
        self.SED = SED
        self.NAD = NAD
        self.NAD_neighbors = NAD_neighbors
        self.PAD = PAD
        self.blending = blending
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.min_P = min_P
        self.godunov = godunov
        # MOOD cascade (used when blending is off): number of fallback levels
        # beyond the base scheme -- hardwired to [MUSCL(-Hancock), 1st order]
        # -- and the maximum number of detection/revision sweeps per update.
        self.n_cascade = 2
        self.max_revs = max_revs
        # Default: check density and pressure.
        self.limiting_variables = (
            limiting_variables
            if limiting_variables is not None
            else [self._d_, self._p_]
        )
        # True only while dm.M is known to hold the current ghosted W_cv
        # (set around detect_troubles by compute_corrected_fluxes).
        self._W_ghosts_current = False
        if self.primary is not None:
            self.ader = False

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        """Initialize FV arrays. If primary exists, also allocate FB arrays."""
        # If standalone, do a full FV init
        if self.primary is None:
            super().initialize()
        else:
            # When used as fallback, the primary scheme handles mesh/init.
            # We just need FV working arrays.
            self.allocate_arrays(ader=False)
            self.fb_arrays()
            # The primary's potential is set up in its own initialize; the FV
            # fallback operates in control-volume layout and needs its own
            # FV-layout gradient (and equilibrium, when well-balanced).
            if self.potential:
                self.init_potential()
            if self.WB:
                self.init_equilibrium_state()

    def init_potential(self) -> None:
        """
        Build the FV-layout potential gradient ``dm.grad_phi``.

        In standalone mode this is computed directly on the FV mesh.  As a
        fallback for a high-order primary, the FV operator runs in
        control-volume layout, so the primary's solution-point gradient is
        projected to the FV layout (an FV primary already stores it there).
        """
        if self.primary is None:
            super().init_potential()
            return
        pdm = self.primary.dm
        if getattr(pdm, "grad_phi_sp", None) is not None:
            # High-order (SD) primary: project the SP gradient to FV layout.
            grad_phi_cv = self.primary.compute_cv_from_sp(pdm.grad_phi_sp)
            self.dm.grad_phi = self.primary.transpose_to_fv(grad_phi_cv)
        else:
            # FV primary already stores the FV-layout gradient.
            self.dm.grad_phi = pdm.grad_phi

    def init_equilibrium_state(self) -> None:
        """
        Build the FV-layout equilibrium state for the well-balanced fallback.

        Standalone mode delegates to the FV-native implementation.  As a
        fallback for a high-order primary, the MUSCL operator runs in
        control-volume layout while the primary holds the equilibrium in its
        own (SD) layout.  Here the equilibrium primitives are projected onto
        the FV mesh (``dm.M_eq``, ghosted) and the single-valued face
        equilibrium (``M_eq_fp_{dim}`` / ``F_eq_fp_{dim}``) is rebuilt with the
        same MUSCL reconstruction used for the solution, so the perturbation
        flux vanishes exactly at equilibrium.  The perturbation shift of the
        evolved solution is owned by the primary scheme and is not repeated.
        """
        if self.primary is None:
            super().init_equilibrium_state()
            return
        primary = self.primary
        nvar = self.nvar
        ngh_c = self.Nghc
        n = primary.n["x"]  # subcells per element (same for all dims)
        # Sample the equilibrium primitives on the primary (SD) mesh, ghosted.
        W_gh = primary.array_sp(ngh=primary.Nghe)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(
                primary.mesh_cv, self.eq_fct, self.ndim, self.p, var
            )
        # Project to the FV cell layout and trim the element ghosts
        # (Nghe * n cell layers) down to the Nghc cell ghosts of the FV operator.
        M_eq_fv = primary.transpose_to_fv(W_gh)
        crop = primary.Nghe * n - ngh_c
        if crop > 0:
            M_eq_fv = M_eq_fv[crop_fv(crop, -crop, 0, self.ndim, crop)]
        M_eq = FV_Scheme.array(self, nvar, ngh=ngh_c)
        M_eq[...] = M_eq_fv
        self.dm.M_eq = M_eq
        # Populate the FV cell/face spacings (h_cv, h_fp) from the primary mesh
        # so the MUSCL reconstruction operators below are available at init.
        self.working_arrays()
        # Single-valued equilibrium faces, reconstructed exactly like the
        # solution so the perturbation MUSCL flux vanishes at equilibrium.
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels, -idim)
            S = self.compute_slopes(M_eq, idim)
            ML = self.interpolate_L(M_eq, S, idim)
            MR = self.interpolate_R(M_eq, S, idim)
            M_eq_fp = 0.5 * (ML + MR)
            self.dm.__setattr__(f"M_eq_fp_{dim}", M_eq_fp)
            F = M_eq_fp.copy()
            self.compute_physical_fluxes(F, M_eq_fp, vels)
            self.dm.__setattr__(f"F_eq_fp_{dim}", F)

    def fb_arrays(self):
        """Allocate arrays used in trouble detection and flux blending."""
        self.dm.troubles = self.array(1)
        self.dm.theta = self.array(1, ngh=self.Nghc)
        for dim in self.dims:
            self.dm.__setattr__(
                f"affected_faces_{dim}",
                self.array(1, dim=dim),
            )
            self.dm.__setattr__(
                f"F_fp_FB_{dim}",
                self.array(self.nvar, dim=dim),
            )
            # Boundary buffer for the trouble field (see Boundaries_scalar).
            # Allocated here so dm.switch_to() moves it to the active backend.
            self.dm.__setattr__(
                f"BC_fp_scalar_{dim}",
                self.array_BC(dim=dim),
            )
        if self.use_mood:
            # MOOD cascade: ghosted per-cell cascade index and the per-level
            # flux sets (level 0 = base/high order, 1 = MUSCL(-Hancock),
            # 2 = first order).
            self.dm.cascade = self.array(1, ngh=self.Nghc)
            for dim in self.dims:
                self.dm.__setattr__(
                    f"F_fp_HO_{dim}", self.array(self.nvar, dim=dim)
                )
                self.dm.__setattr__(
                    f"F_fp_FB2_{dim}", self.array(self.nvar, dim=dim)
                )

    @property
    def use_mood(self):
        """The MOOD cascade replaces the single-pass theta blend whenever the
        neighbor-spreading blend is off (superfv-style: blending and cascade
        revisions are mutually exclusive)."""
        return not self.blending and not self.godunov

    def allocate_arrays(self, ader=False):
        """Allocate arrays.  When a primary exists, super() will allocate the
        integrator stage arrays using array_sp() (→ primary/SD layout), which
        is correct for the time integrator.  The FV working arrays (M, U_new,
        dtM) allocated by FV_Scheme.allocate_arrays must stay in FV layout, so
        we overwrite them afterwards using FV_Scheme.array directly."""
        super().allocate_arrays(ader)
        if self.primary is not None:
            self.dm.M = FV_Scheme.array(self, self.nvar, ngh=self.Nghc)
            self.dm.U_new = FV_Scheme.array(self, self.nvar)
            if self.scheme == "MUSCL-Hancock":
                self.dm.dtM = FV_Scheme.array(self, self.nvar, ngh=self.Nghc - 1)

    def create_dicts(self):
        """Create dictionaries for the working arrays."""
        super().create_dicts()
        self.F_fp_FB = {dim: self.dm.__getattribute__(f"F_fp_FB_{dim}") for dim in self.dims}
        self.BC_fp_scalar = {dim: self.dm.__getattribute__(f"BC_fp_scalar_{dim}") for dim in self.dims}
        if self.use_mood and getattr(self.dm, "cascade", None) is not None:
            self.F_fp_HO = {
                dim: self.dm.__getattribute__(f"F_fp_HO_{dim}") for dim in self.dims
            }
            self.F_fp_FB2 = {
                dim: self.dm.__getattribute__(f"F_fp_FB2_{dim}") for dim in self.dims
            }

    @property
    def state_dm(self):
        """The solution state lives on the primary's data manager (when one
        exists); W_cv/U_cv/U_eq_cv property access resolves through it, so no
        per-switch pointer refresh is needed."""
        if self.primary is None:
            return self.dm
        return self.primary.dm

    def working_arrays(self) -> None:
        """Refresh the mesh-coordinate dictionaries from the owning data
        manager (needed after host/device switches, via create_dicts)."""
        dm = self.state_dm
        for dim in self.dims:
            self.faces[dim] = dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = dm.__getattribute__(f"d{dim}_cv")

    # ----------------------------------------------------------------
    # Trouble detection
    # ----------------------------------------------------------------

    def detect_troubles(self):
        """
        Detect troubled cells using NAD/PAD criteria.
        Delegates to the trouble_detection module.
        """
        detect_troubles(self)

    def refresh_theta_ghosts(self, theta) -> None:
        """Hook called after apply_blending. On a single global grid the
        ghost cells written by the blending pass are already consistent;
        the block-based AMR fallback overrides this to re-exchange theta
        across block interfaces."""

    def minimize_alpha_across_blocks(self, alpha, dims=None) -> None:
        """Hook: see FallbackAMRScheme. No-op on a single global grid."""

    # ----------------------------------------------------------------
    # Flux blending
    # ----------------------------------------------------------------

    def _split_sd_axes(self, M):
        """View an FV cell-based (possibly strided) array with the element and
        point axes split, in SD element-based order (u, Nz, Ny, Nx, nz, ny, nx).

        Pure stride manipulation (no copy), so SD-layout data can be copied
        into FV-layout storage in a single strided pass.
        """
        Ns = [self.N[dim] for dim in self.dims][::-1]
        ns = [self.n[dim] for dim in self.dims][::-1]
        shape = (M.shape[0],) + tuple(Ns) + tuple(ns)
        s = [M.strides[1 + ax] for ax in range(self.ndim)]
        strides = (
            (M.strides[0],)
            + tuple(ns[ax] * s[ax] for ax in range(self.ndim))
            + tuple(s)
        )
        if CUPY_AVAILABLE and isinstance(M, cp.ndarray):
            from cupy.lib.stride_tricks import as_strided
        else:
            from numpy.lib.stride_tricks import as_strided
        return as_strided(M, shape=shape, strides=strides)

    def store_high_order_fluxes(self, i_ader, ader=True):
        """
        Store the primary scheme's high-order fluxes into the
        FV face-flux layout for comparison/blending.
        """
        if self.primary is None:
            return
        if "FE" in self.primary.scheme:
            # For FE schemes the fluxes live in the SD element-based layout.
            # Copy them straight into a split-axis view of the FV flux array
            # (single strided copy; no transposed temporary).
            ndim = self.ndim
            dims2 = [(0), (0, 1, 2), (0, 1, 3, 2, 4)]
            Nn = [self.N[dim] * self.n[dim] for dim in self.dims][::-1]
            for dim in self.dims:
                shift = self.dims[dim]
                F = self.get_high_order_fluxes(dim, i_ader, ader)
                # Interior faces: n faces per element (last one dropped).
                self._split_sd_axes(self.F_fp[dim][cut(None, -1, shift)])[
                    ...
                ] = F[cut(None, -1, shift)]
                # Domain-boundary face (last face of the last element).
                shape = [self.nvar] + Nn[: ndim - 1 - shift] + Nn[ndim - shift :]
                self.F_fp[dim][indices(-1, shift)] = np.transpose(
                    F[indices2(-1, ndim, shift)], dims2[ndim - 1]
                ).reshape(shape)
        else:
            """ For FV schemes we can just store the high-order fluxes in the FV face-flux layout. """
            for dim in self.dims:
                self.F_fp[dim] = self.get_high_order_fluxes(dim, i_ader, ader)

    def get_high_order_fluxes(self, dim, i_ader, ader=True):
        return self.primary.F_fp[dim][:, i_ader] if ader else self.primary.F_fp[dim]
        
        
    def correct_fluxes(self):
        """Blend high-order (primary) and low-order (MUSCL) fluxes."""
        for dim in self.dims:
            if self.godunov:
                theta = 1
            else:
                theta = self.dm.__getattribute__(f"affected_faces_{dim}")
            self.F_fp[dim] = blend_fluxes(
                self.F_fp[dim], self.F_fp_FB[dim], theta
            )

    def compute_corrected_fluxes(self, dt):
        """
        Compute the corrected fluxes with trouble detection.

        MOOD path (blending off): iterative cascade of detection/revision
        sweeps over the flux levels [high order, MUSCL(-Hancock), 1st order].

        Blending path: single sweep --
        1. Tentatively apply high-order fluxes (already in F_fp).
        2. Compute MUSCL fluxes into F_fp_FB (not overwriting F_fp/HO).
        3. Detect troubled cells.
        4. Blend F_fp (HO) and F_fp_FB (MUSCL) based on trouble indicators.

        The MUSCL fluxes are computed *before* the detection: both start from
        the same ghosted primitive field in ``dm.M``, so when the flux
        pipeline leaves it untouched (plain MUSCL, no nabla terms) the
        detection reuses it and skips a redundant ghost fill + boundary/halo
        exchange.
        """
        self.W_cv[...] = self.primary.compute_primitives_cv(self.U_cv)
        if self.use_mood:
            return self.mood_loop(dt)
        # Tentative HO update for trouble detection; F_fp still holds HO fluxes
        self.apply_fluxes(dt)
        # Redirect compute_fluxes output to F_fp_FB so HO fluxes in F_fp survive
        self.compute_fluxes(self.F_fp_FB, dt)
        self._W_ghosts_current = self.scheme == "MUSCL" and not (
            self.viscosity or self.thdiffusion
        )
        self.detect_troubles()
        self._W_ghosts_current = False
        # Blend: F_fp = HO, F_fp_FB = MUSCL
        self.correct_fluxes()

    # ----------------------------------------------------------------
    # MOOD cascade
    # ----------------------------------------------------------------

    def compute_first_order_fluxes(self, F):
        """First-order Godunov fluxes: face states are the adjacent cell
        values of the start-of-node primitives (no reconstruction, no time
        prediction) -- the terminal, unconditionally robust cascade level."""
        ngh = self.Nghc
        self.dm.M[...] = 0
        self.fill_active_region(self.W_cv)
        self.Boundaries(self.dm.M)
        M = self.dm.M
        for dim in self.dims:
            idim = self.dims[dim]
            crop = lambda start, end: crop_fv(start, end, idim, self.ndim, ngh)
            self.ML_fp[dim][...] = M[crop(1, -2)]
            self.MR_fp[dim][...] = M[crop(2, -1)]
            self.solve_riemann_problem(dim, F[dim], prims=True)

    def cascade_face_mask(self, dim):
        """Per-face cascade level: max of the two adjacent cells' levels
        (from the ghosted ``dm.cascade``)."""
        ngh = self.Nghc
        idim = self.dims[dim]
        crop = lambda start, end: crop_fv(start, end, idim, self.ndim, ngh)
        c = self.dm.cascade[0]
        return np.maximum(c[crop(ngh - 1, -ngh)], c[crop(ngh, -(ngh - 1))])

    def assign_cascade_fluxes(self, i_max_computed):
        """Assemble ``F_fp`` per face from the per-level flux sets according
        to the face cascade mask (level of a face = max of its two cells)."""
        levels = [self.F_fp_HO, self.F_fp_FB, self.F_fp_FB2]
        for dim in self.dims:
            mask = self.cascade_face_mask(dim)
            F = self.F_fp[dim]
            F[...] = levels[0][dim]
            for i in range(1, i_max_computed + 1):
                F[...] = np.where(mask == i, levels[i][dim], F)

    def mood_hook_start(self, dt):
        """Hook for coupled schemes (MHD): the level-1 (MUSCL) fluxes have
        just been computed and ``dm.M`` holds the ghosted (possibly
        Hancock-predicted) primitives; prepare matching level-1 data."""

    def mood_hook_candidate(self, dt):
        """Hook for coupled schemes (MHD): update auxiliary candidate data
        for the current flux/E assembly before detection."""

    def mood_hook_level(self, level):
        """Hook for coupled schemes (MHD): a new cascade level's fluxes have
        just been computed (``dm.M`` holds the ghosted start-of-node
        primitives); prepare the matching auxiliary level (edge E)."""

    def mood_loop(self, dt):
        """superfv-style MOOD loop with the cascade hardwired to
        [base/high-order, MUSCL(-Hancock), first order].

        Each sweep builds the candidate update from the current per-face flux
        assembly, runs NAD/PAD detection on it, and demotes still-troubled
        cells one cascade level.  Because a revision changes the fluxes of
        neighboring cells too, the loop continues until no *revisable*
        troubled cell remains (or ``max_revs`` sweeps were done).  Cells at
        the terminal (first-order) level are no longer revisable.
        """
        dm = self.dm
        # Level 0: high-order node fluxes (currently in F_fp).
        for dim in self.dims:
            self.F_fp_HO[dim][...] = self.F_fp[dim]
        # Level 1: MUSCL(-Hancock) fluxes.
        self.compute_fluxes(self.F_fp_FB, dt)
        self.mood_hook_start(dt)
        i_max_computed = 1
        # Ghosted per-cell cascade index (float array; levels are small ints).
        dm.cascade[...] = 0
        interior = (Ellipsis,) + (slice(self.Nghc, -self.Nghc),) * self.ndim
        cascade_in = dm.cascade[0][interior]

        for _ in range(self.max_revs):
            self.apply_fluxes(dt)
            self.mood_hook_candidate(dt)
            self.detect_troubles()
            troubles = dm.troubles[0]
            revisable = troubles * (cascade_in < self.n_cascade)
            if float(revisable.sum()) == 0:
                break
            cascade_in += revisable
            # Refresh the ghost layers of the cascade index (face masks read
            # neighbor levels across element/domain boundaries).
            dm.M[...] = 0
            self.fill_active_region(cascade_in)
            self.Boundaries_scalar(dm.M)
            dm.cascade[0][...] = dm.M[0]
            if int(cascade_in.max()) > i_max_computed:
                self.compute_first_order_fluxes(self.F_fp_FB2)
                self.mood_hook_level(2)
                i_max_computed = 2
            self.assign_cascade_fluxes(i_max_computed)

    # ----------------------------------------------------------------
    # Solution state delegation to primary (for RK integrator)
    # ----------------------------------------------------------------

    def array_sp(self, **kwargs):
        if self.primary is not None:
            return self.primary.array_sp(**kwargs)
        return super().array_sp(**kwargs)

    def compute_primitives_cv(self, U):
        # The well-balanced equilibrium conservatives live on the primary's
        # data manager (and follow its SD<->FV layout toggling), so delegate
        # the perturbation->primitive conversion to the primary.
        if self.primary is not None:
            return self.primary.compute_primitives_cv(U)
        return super().compute_primitives_cv(U)

    def convert_solution(self, W=False):
        # With a primary, the solution state (and its well-balanced
        # perturbation/full bookkeeping) is owned by the primary scheme, so
        # delegate.  The inherited FV conversion assumes ``U_cv`` is the
        # perturbation, which is not the case after the SDFB update restores
        # the full state, and would otherwise add the equilibrium twice.
        if self.primary is not None:
            return self.primary.convert_solution(W=W)
        return super().convert_solution(W=W)

    def get_solution(self, ader=False):
        if self.primary is not None:
            return self.primary.get_solution(ader=ader)
        return super().get_solution(ader=ader)

    def set_solution(self, U):
        if self.primary is not None:
            return self.primary.set_solution(U)
        return super().set_solution(U)

    def update_solution(self, dU):
        if self.primary is not None:
            return self.primary.update_solution(dU)
        return super().update_solution(dU)

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Evaluate the spatial RHS for a given state U."""
        if self.primary is None:
            return super().compute_update(U, ader=ader, prims=prims, **kwargs)

        self.primary.solve_faces(U, ader=ader, prims=prims)
        self.primary.switch_to_finite_volume(U_sp=U)
        self.store_high_order_fluxes(0, ader=ader)
        self.compute_corrected_fluxes(self.dt)
        # Project dU/dt from the FV layout straight to solution points
        # (layout change and projection fused into one contraction).
        dUdt_fv = self.compute_dudt(self.U_cv)
        dUdt_sp = self.primary.compute_sp_from_cv_fv(dUdt_fv)
        self.primary.switch_to_high_order(update_solution_points=False)
        return dUdt_sp

    def ader_predictor(self, prims: bool = False) -> None:
        """Perform the ADER predictor step.

        When a primary scheme exists, delegate the Picard iteration to it
        (the predictor operates entirely in the primary scheme's data layout).
        """
        if self.primary is None:
            return super().ader_predictor(prims=prims)
        self.primary.ader_predictor(prims=prims)

    def ader_update(self):
        """Perform the ADER corrector/update step."""
        if self.primary is None:
            return super().ader_update()
        # Predictor has filled primary.F_fp with fluxes at each ADER time point.
        # Switch layout, then for each quadrature point: detect troubles, blend,
        # and accumulate the weighted flux divergence.
        self.switch_to_finite_volume(ader=True)
        for i_ader in range(self.nader):
            dt_i = self.dt * self.dm.w_tp[i_ader]
            self.store_high_order_fluxes(i_ader, ader=True)
            # apply_fluxes inside compute_corrected_fluxes sets dm.U_new for
            # trouble detection; blended F_fp is ready afterwards
            self.compute_corrected_fluxes(dt_i)
            self.U_cv -= self.compute_dudt(self.U_cv) * dt_i
        self.switch_to_high_order()

    def switch_to_finite_volume(self, ader=False):
        """Convert SD element-based arrays to FV cell-based layout."""
        self.primary.switch_to_finite_volume(ader=ader)

    def switch_to_high_order(self):
        """Convert FV cell-based arrays back to SD element-based layout."""
        self.primary.switch_to_high_order()
