"""
MHD simulator: coupled fluid + constrained transport, with an MHD-aware
fallback.

With ``scheme="SD"``/``"SDFB"`` the high-order scheme is
:class:`MHD_SD_Scheme` (fluid SD fluxes + edge CT with a fused ADER
predictor/corrector); with ``FB`` enabled the low-order scheme is
:class:`MHDFallbackScheme`, which limits both the conserved state (MUSCL
flux blending / MOOD cascade) and the face-staggered B field (edge E-field
levels), keeping div(B) = 0 to machine precision.

With ``scheme="FV"`` the primary scheme is :class:`MHD_FV_Scheme`: pure
MUSCL / MUSCL-Hancock fluxes with face-staggered CT on the FV mesh (the
same corner E-field construction the fallback borrows), under a
Runge-Kutta integrator (``time_integrator='rk1'`` with
``fallback='MUSCL-Hancock'`` gives the classic second-order scheme).
"""

from spd.hydro.hydro_simulator import HydroSimulator
from .mhd_sd_scheme import MHD_SD_Scheme
from .mhd_fv_scheme import MHD_FV_Scheme
from .mhd_fallback import MHDFallbackScheme
from . import mhd as mhd_eq


class MHDSimulator(HydroSimulator):
    """
    Same constructor as :class:`HydroSimulator` plus MHD induction on the SD
    mesh (``vectorpot_fct`` provides the vector potential for a
    divergence-free initial B).
    """

    def __init__(
        self,
        equations=mhd_eq,
        riemann_solver_sd: str = "llf",
        *args,
        **kwargs,
    ):
        kwargs.setdefault("soe", "mhd")
        scheme = kwargs.get("scheme", "SDFB")
        if "FV" in scheme and "FB" in scheme:
            raise NotImplementedError(
                "scheme='FVFB' is not supported for MHD yet; use 'SDFB' "
                "(subcell fallback) or 'FV' (pure MUSCL/MUSCL-Hancock CT)."
            )
        # No neighbor-spreading of the trouble indicator for MHD: detection
        # stays on cell-centered quantities (rho, p and the active B rows).
        kwargs.setdefault("blending", False)
        self.equations = equations
        super().__init__(
            *args,
            ho_scheme_cls=MHD_SD_Scheme,
            fv_scheme_cls=MHD_FV_Scheme,
            fb_scheme_cls=MHDFallbackScheme,
            riemann_solver_sd=riemann_solver_sd,
            **kwargs,
        )

    def B_to_U(self):
        self.ho_scheme.B_to_U()
