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

from .scheme import SemiDiscreteScheme
from spd.finite_volume.fv_scheme import FV_Scheme
from spd.trouble_detection import detect_troubles
from spd.numerics.slicing import cut, indices, indices2, crop_fv
from spd.numerics.polynomials import quadrature_mean


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
        NAD mode ('' or 'delta').
    PAD : bool
        Enable physically admissible detection.
    blending : bool
        Spread trouble indicators to neighbors.
    min_rho, max_rho, min_P : float
        Bounds for PAD checks.
    godunov : bool
        If True, use pure Godunov (no blending, theta=1 everywhere).
    limiting_variables : list
        Variable indices used for NAD check.
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
        PAD=True,
        blending=True,
        min_rho=1e-10,
        max_rho=1e10,
        min_P=1e-10,
        godunov=False,
        limiting_variables=None,
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
        self.PAD = PAD
        self.blending = blending
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.min_P = min_P
        self.godunov = godunov
        self.limiting_variables = (
            limiting_variables if limiting_variables is not None else [0]
        )
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

    def working_arrays(self) -> None:
        """ Create pointers to the working arrays of the primary scheme. """
        if self.primary is None:
            dm = self.dm
        else:
            dm = self.primary.dm

        for dim in self.dims:
            self.faces[dim] = dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = dm.__getattribute__(f"d{dim}_cv")

        #Pointer to the working array
        self.W_cv = dm.W_cv
        self.U_cv = dm.U_cv

        # The primary re-creates its equilibrium conservatives on every
        # SD<->FV layout switch, so mirror the current (FV-layout) array onto
        # the fallback data manager for the trouble-detection routine, which
        # reads ``self.dm.U_eq_cv`` directly.
        if self.WB and self.primary is not None:
            self.dm.U_eq_cv = dm.U_eq_cv

    # ----------------------------------------------------------------
    # Trouble detection
    # ----------------------------------------------------------------

    def detect_troubles(self):
        """
        Detect troubled cells using NAD/PAD criteria.
        Delegates to the trouble_detection module.
        """
        detect_troubles(self)

    # ----------------------------------------------------------------
    # Flux blending
    # ----------------------------------------------------------------

    def store_high_order_fluxes(self, i_ader, ader=True):
        """
        Store the primary scheme's high-order fluxes into the
        FV face-flux layout for comparison/blending.
        """
        if self.primary is None:
            return
        if "FE" in self.primary.scheme:
            """ For FE schemes, we need to store the high-order fluxes in the FV face-flux layout. """
            ndim = self.ndim
            dims = [(0, 1, 2), (0, 1, 3, 2, 4), (0, 1, 4, 2, 5, 3, 6)]
            dims2 = [(0), (0, 1, 2), (0, 1, 3, 2, 4)]
            Nn = [self.N[dim] * self.n[dim] for dim in self.dims][::-1]
            for dim in self.dims:
                shift = self.dims[dim]
                shape = [self.nvar] + Nn
                F = self.get_high_order_fluxes(dim, i_ader, ader)
                self.F_fp[dim][cut(None, -1, shift)] = np.transpose(
                    F[cut(None, -1, shift)], dims[ndim - 1]
                ).reshape(shape)
                shape.pop(ndim - shift)
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
            self.F_fp[dim] = (
                theta * self.F_fp_FB[dim]
                + (1 - theta) * self.F_fp[dim]
            )

    def compute_corrected_fluxes(self, dt):
        """
        Compute the corrected solution with trouble detection and blending.

        1. Tentatively apply high-order fluxes (already in F_fp).
        2. Detect troubled cells.
        3. Compute MUSCL fluxes into F_fp_FB (not overwriting F_fp/HO).
        4. Blend F_fp (HO) and F_fp_FB (MUSCL) based on trouble indicators.
        """
        self.W_cv[...] = self.primary.compute_primitives_cv(self.U_cv)
        # Tentative HO update for trouble detection; F_fp still holds HO fluxes
        self.apply_fluxes(dt)
        self.detect_troubles()
        # Redirect compute_fluxes output to F_fp_FB so HO fluxes in F_fp survive
        self.compute_fluxes(self.F_fp_FB, dt)
        # Blend: F_fp = HO, F_fp_FB = MUSCL
        self.correct_fluxes()

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
            self.primary.update_solution(dU)
            # Refresh pointer after primary may have reallocated U_cv
            self.working_arrays()
            return
        return super().update_solution(dU)

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Evaluate the spatial RHS for a given state U."""
        if self.primary is None:
            return super().compute_update(U, ader=ader, prims=prims, **kwargs)

        self.primary.solve_faces(U, ader=ader, prims=prims)
        self.primary.switch_to_finite_volume(U_sp=U)
        self.working_arrays()
        self.store_high_order_fluxes(0, ader=ader)
        self.compute_corrected_fluxes(self.dt)
        # Compute dU/dt in FV layout, then reshape to primary (SD) layout
        dUdt_fv = self.compute_dudt(self.U_cv)
        dUdt_sd = self.primary.transpose_to_sd(dUdt_fv)
        dUdt_sd = self.primary.compute_sp_from_cv(dUdt_sd)
        self.primary.switch_to_high_order(update_solution_points=False)
        self.working_arrays()
        return dUdt_sd

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
        #Update pointers to the working arrays
        self.working_arrays()
    
    def switch_to_high_order(self):
        """Convert FV cell-based arrays back to SD element-based layout."""
        self.primary.switch_to_high_order()
        #Update pointers to the working arrays
        self.working_arrays()
