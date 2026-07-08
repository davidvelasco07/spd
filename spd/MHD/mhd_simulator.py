"""
MHD simulator: coupled SD + constrained transport, with an MHD-aware fallback.

The high-order scheme is :class:`MHD_SD_Scheme` (fluid SD fluxes + edge CT
with a fused ADER predictor/corrector).  With ``FB`` enabled the low-order
scheme is :class:`MHDFallbackScheme`, which limits both the conserved state
(MUSCL flux blending) and the face-staggered B field (theta-blended edge
E-field), keeping div(B) = 0 to machine precision.
"""

from spd.hydro.hydro_simulator import HydroSimulator
from .mhd_sd_scheme import MHD_SD_Scheme
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
        # No neighbor-spreading of the trouble indicator for MHD: detection
        # stays on cell-centered quantities (rho, p and the active B rows).
        kwargs.setdefault("blending", False)
        self.equations = equations
        super().__init__(
            *args,
            ho_scheme_cls=MHD_SD_Scheme,
            fb_scheme_cls=MHDFallbackScheme,
            riemann_solver_sd=riemann_solver_sd,
            **kwargs,
        )
        if not self.ader:
            raise NotImplementedError(
                "MHDSimulator only supports the ADER time integrator; "
                "a coupled U+B Runge-Kutta path is not implemented."
            )

    def B_to_U(self):
        self.ho_scheme.B_to_U()
