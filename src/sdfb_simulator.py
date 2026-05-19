"""
Spectral Difference with Fallback Blending Simulator.

Uses SD_Scheme as the primary high-order spatial scheme and
FallbackScheme for robust shock capturing via MUSCL/MUSCL-Hancock
with trouble detection and flux blending.
"""

import numpy as np

from simulator import Simulator
from runtime.data_management import CupyLocation
from spectral_difference.sd_scheme import SD_Scheme
from finite_volume.fv_scheme import FV_Scheme
from schemes.fallback import FallbackScheme
from schemes.scheme import SemiDiscreteScheme
from numerics.polynomials import quadrature_mean
from numerics.slicing import cut, indices, indices2, crop_fv


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
        NAD mode ('' or 'delta').
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
        fallback: str = "MUSCL-Hancock",
        FB: bool = None,
        tolerance: float = 1e-5,
        SED: bool = True,
        NAD: str = "",
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
        if self.WB:
            # U -> U'
            self.dm.U_sp -= self.dm.U_eq_sp
        self.integrator.update(self.scheme)
        if self.WB:
            # U' -> U
            self.dm.U_sp[...] += self.dm.U_eq_sp
            self.dm.U_cv[...] += self.dm.U_eq_cv
        self.compute_primitives(self.dm.U_cv, W=self.dm.W_cv)
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

    def init_sim(self):
        self.checkpoint = False
        self.switch_to_device()
        self.create_dicts()
        from timeit import default_timer as timer
        self.execution_time = -timer()

    def end_sim(self):
        self.switch_to_host()
        from timeit import default_timer as timer
        self.execution_time += timer()
        self.create_dicts()
        self.convert_solution()
        if self.rank == 0:
            print(
                f"t={self.time}, steps taken {self.n_step}, "
                f"time taken {self.execution_time}"
            )

    # ------------------------------------------------------------------
    # Potential and equilibrium
    # ------------------------------------------------------------------

    def init_potential(self) -> None:
        self.ho_scheme.init_potential()
        self.lo_scheme.init_potential()

    def init_equilibrium_state(self) -> None:
        crop = lambda start, end, idim, ngh: crop_fv(
            start, end, idim, self.ndim, ngh
        )
        p = self.p
        n = p + 1
        nvar = self.nvar
        ngh = self.Nghe
        W_gh = self.sd_scheme.array_sp(ngh=ngh)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(
                self.sd_scheme.mesh_cv, self.eq_fct, self.ndim, self.p, var
            )

        W_sp = self.sd_scheme.compute_sp_from_cv(W_gh)
        U_sp = self.compute_conservatives(W_sp)
        self.dm.U_eq_sp = self.crop(U_sp)
        self.dm.U_eq_cv = self.sd_scheme.compute_cv_from_sp(self.dm.U_eq_sp)
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels, -idim)
            U = self.sd_scheme.compute_fp_from_sp(U_sp, dim)
            self.dm.__setattr__(f"M_eq_fp_{dim}", self.crop(U))
            M_fp = self.dm.__getattribute__(f"M_eq_fp_{dim}")
            # Force equilibrium values at flux points to match between elements
            M_fp[cut(1, None, idim + self.ndim)][indices(0, idim)] = M_fp[
                cut(None, -1, idim + self.ndim)
            ][indices(-1, idim)]
            F = U.copy()
            W = self.compute_primitives(U)
            self.compute_fluxes(F, W, vels, prims=True)
            self.dm.__setattr__(f"F_eq_fp_{dim}", self.crop(F))

            update_mode = getattr(self, '_update_mode', 'SD')
            if update_mode == "FV":
                W_faces = self.sd_scheme.integrate_faces(
                    W, dim, ader=False
                )[cut(None, -1, idim)]
                W_faces = self.sd_scheme.transpose_to_fv(W_faces)
                W_faces = W_faces[crop(p + 1, -p, idim, p + 1)]
                self.dm.__setattr__(f"M_eq_faces_{dim}", W_faces)
                F = W_faces.copy()
                self.compute_fluxes(F, W_faces, vels, prims=True)
                self.dm.__setattr__(f"F_eq_faces_{dim}", F)
        ngh = self.Nghc
        if update_mode == "FV":
            if n > ngh:
                self.dm.M_eq_fv = self.sd_scheme.transpose_to_fv(W_gh)[
                    crop(n - ngh, -(n - ngh), 0, n - ngh)
                ]
            else:
                self.dm.M_eq_fv = self.sd_scheme.transpose_to_fv(W_gh)

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
        if ho is not None and hasattr(ho, name):
            return getattr(ho, name)
        if lo is not None and hasattr(lo, name):
            return getattr(lo, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Backward-compatible names used by tests and legacy scripts.
SDFB_Simulator = SPD_Simulator
SDADER_Simulator = SPD_Simulator
