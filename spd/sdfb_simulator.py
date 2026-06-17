"""
Spectral Difference with Fallback Blending Simulator.

Uses SD_Scheme as the primary high-order spatial scheme and
FallbackScheme for robust shock capturing via MUSCL/MUSCL-Hancock
with trouble detection and flux blending.
"""

import numpy as np

from .simulator import Simulator
from .runtime.data_management import CupyLocation
from .spectral_difference.sd_scheme import SD_Scheme
from .finite_volume.fv_scheme import FV_Scheme
from .fallback import FallbackScheme
from .schemes.scheme import SemiDiscreteScheme


class SPD_Simulator(Simulator):
    """
    Simulator combining high-order SD with MUSCL fallback.

    The SD_Scheme computes high-order fluxes.  The FallbackScheme
    detects troubled cells and blends in low-order MUSCL fluxes
    to maintain stability near discontinuities.

    Parameters
    ----------
    FB : bool
        Enable flux blending.
    tolerance : float
        Tolerance for NAD check.
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
        Spread trouble to neighbors.
    min_rho, max_rho, min_P : float
        Bounds for PAD.
    godunov : bool
        Use pure Godunov (no blending).
    limiting_variables : list
        Variable indices for NAD.
    predictor : bool
        Use MUSCL-Hancock for fallback.
    riemann_solver_sd, riemann_solver_fv : str
        Riemann solver names.
    slope_limiter : str
        Slope limiter name.
    """

    def __init__(
        self,
        scheme: str = "SDFB",
        fallback: str = None,
        FB: bool = None,
        tolerance: float = 1e-5,
        SED: bool = True,
        NAD: str = "",
        NAD_neighbors: str = "2nd",
        PAD: bool = True,
        blending: bool = True,
        min_rho: float = 1e-10,
        max_rho: float = 1e10,
        min_P: float = 1e-10,
        godunov: bool = False,
        limiting_variables: list = [0],
        predictor: bool = False,
        riemann_solver_sd: str = "llf",
        riemann_solver_fv: str = "llf",
        slope_limiter: str = "minmod",
        ho_scheme_cls=None,
        *args,
        **kwargs,
    ):
        # Handle the FB convenience flag:
        # FB=True  -> scheme with "FB" (e.g. "SDFB")
        # FB=False -> scheme without "FB" (e.g. "SD")
        if FB is not None:
            if FB and "FB" not in scheme:
                scheme = scheme + "FB"
            elif not FB and "FB" in scheme:
                scheme = scheme.replace("FB", "")

        # Store FB params before super().__init__
        self._fb_params = dict(
            tolerance=tolerance,
            SED=SED,
            NAD=NAD,
            NAD_neighbors=NAD_neighbors,
            PAD=PAD,
            blending=blending,
            min_rho=min_rho,
            max_rho=max_rho,
            min_P=min_P,
            limiting_variables=limiting_variables,
        )

        init = kwargs.pop("init", True)
        Simulator.__init__(self, init=False, *args, **kwargs)
        self.init = init

        # Default fallback reconstruction depends on the time integrator.
        # MUSCL-Hancock adds a half-step (dt-dependent) time prediction that
        # suits the ADER update but is not a pure spatial RHS, so under a
        # Runge-Kutta (method-of-lines) integrator we default to plain MUSCL.
        if fallback is None:
            fallback = "MUSCL-Hancock" if self.ader else "MUSCL"

        # The simulator's main scheme is the SD scheme;
        # the fallback is used during ader_update / fv_update.

        # Create SD scheme (primary high-order scheme)
        if ho_scheme_cls is None:
            ho_scheme_cls = SD_Scheme
        if "SD" in scheme:
            self.ho_scheme = ho_scheme_cls(self, riemann_solver=riemann_solver_sd)
        elif "FV" in scheme:
            for dim in self.dims:
                self.n[dim] = 1
                setattr(self, f"n{dim}", 1)
            self.ho_scheme = FV_Scheme(self, riemann_solver=riemann_solver_fv)
            self.cfl_coeff /= self.p + 1
        else:
            raise ValueError(f"Invalid scheme: {scheme}")

        # Create Fallback scheme (MUSCL/MUSCL-Hancock)
        if "FB" in scheme:
            self.lo_scheme = FallbackScheme(
                self,
                primary=self.ho_scheme,
                riemann_solver=riemann_solver_fv,
                slope_limiter=slope_limiter,
                scheme=fallback,
                **self._fb_params,
            )
            self.ader_update = self.lo_scheme.ader_update
            self.scheme = self.lo_scheme
        else:
            # Use a generic semi-discrete scheme for fallback
            self.lo_scheme = SemiDiscreteScheme(self)
            self.ader_update = self.ho_scheme.ader_update
            self.scheme = self.ho_scheme
            
        if self.init:
            self._initialize()
            # Sync dt from scheme to simulator
            self.dt = self.ho_scheme.dt


    @property
    def dm(self):
        """
        The GPUDataManager lives on the scheme.  When no scheme has
        been attached yet (e.g. bare Simulator in tests), a private
        fallback dm is lazily created.
        """
        scheme = self.__dict__.get('ho_scheme')
        return scheme.dm

    def _initialize(self):
        """Initialize both SD and Fallback schemes."""
        # Initialize the SD scheme (primary)
        self.ho_scheme.initialize()
        # Initialize the Fallback (FV arrays + FB arrays)
        self.lo_scheme.initialize()

        self.create_dicts()

    # ------------------------------------------------------------------
    # Dict creation
    # ------------------------------------------------------------------

    def create_dicts(self):
        self.ho_scheme.create_dicts()
        self.lo_scheme.create_dicts()

    # ------------------------------------------------------------------
    # Boundary initialization
    # ------------------------------------------------------------------

    def init_Boundaries(self):
        self.ho_scheme.init_Boundaries()
        self.lo_scheme.init_Boundaries()

    # ------------------------------------------------------------------
    # perform_update
    # ------------------------------------------------------------------

    def perform_update(self) -> bool:
        self.n_step += 1
        # The SD perturbation shift only applies when the high-order scheme
        # owns an element-based solution (U_sp / U_eq_sp).  The FV scheme keeps
        # U_cv as the perturbation directly and handles the equilibrium inside
        # its own reconstruction, so it must skip this shift.
        wb_sd = self.WB and getattr(self.dm, "U_eq_sp", None) is not None
        if wb_sd:
            # U -> U'
            self.dm.U_sp -= self.dm.U_eq_sp
        self.integrator.update(self.scheme)
        if wb_sd:
            # U' -> U
            self.dm.U_sp[...] += self.dm.U_eq_sp
            self.dm.U_cv[...] += self.dm.U_eq_cv
        U_full = self.dm.U_cv
        if self.WB and not wb_sd:
            # FV well-balanced: U_cv stores the perturbation; report full state.
            U_full = self.dm.U_cv + self.dm.U_eq_cv
        self.compute_primitives(U_full, W=self.dm.W_cv)
        self.time += self.dt
        return True

    # ------------------------------------------------------------------
    # Simulation lifecycle (switch both scheme dms)
    # ------------------------------------------------------------------

    def switch_to_device(self):
        self.ho_scheme.dm.switch_to(CupyLocation.device)
        self.lo_scheme.dm.switch_to(CupyLocation.device)

    def switch_to_host(self):
        self.ho_scheme.dm.switch_to(CupyLocation.host)
        self.lo_scheme.dm.switch_to(CupyLocation.host)

    # Potential and well-balanced equilibrium are initialized by each scheme
    # in its own ``initialize`` (SD_Scheme / FV_Scheme), so there is no
    # simulator-level init here.

    @property
    def domain_size(self):
        return self.ho_scheme.domain_size

    def regular_mesh(self, W):
        return self.ho_scheme.interpolate_to_regular_mesh(W)

    def transpose_to_fv(self, M):
        return self.ho_scheme.transpose_to_fv(M)

    def transpose_to_sd(self, M):
        return self.ho_scheme.transpose_to_sd(M)

    # Delegate to sd_scheme or fallback for backward compat
    def __getattr__(self, name):
        if name.startswith('_') or name in ('scheme', 'ho_scheme', 'lo_scheme'):
            raise AttributeError(name)
        d = object.__getattribute__(self, '__dict__')
        ho = d.get('ho_scheme')
        lo = d.get('lo_scheme') # Fallback scheme
        # Avoid recursive getattr loops between scheme<->sim proxying:
        # inspect concrete attributes on the scheme objects without invoking
        # their own __getattr__ fallbacks.
        if ho is not None:
            try:
                return object.__getattribute__(ho, name)
            except AttributeError:
                pass
        if lo is not None:
            try:
                return object.__getattribute__(lo, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Backward-compatible names used by tests and legacy scripts.
SDFB_Simulator = SPD_Simulator
SDADER_Simulator = SPD_Simulator
