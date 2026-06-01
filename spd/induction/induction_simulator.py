"""
Pure induction driver: SD constrained transport + ADER or induction RK.
"""

import os
import numpy as np

from simulator import Simulator
from induction.induction_sd_scheme import InductionSD_Scheme
from induction.induction_fv_scheme import InductionFV_Scheme
from induction.induction_fallback import InductionFallbackScheme


class InductionSimulator(Simulator):
    """
    Parameters
    ----------
    scheme_fb : str
        ``"SD"``, ``"SDFB"``, ``"FV"``, or ``"FVFB"``. ``"FV*"`` uses
        :class:`induction.induction_fv_scheme.InductionFV_Scheme` (cell-centered
        primitives + face CT); ``"SD*"`` uses spectral-difference induction.
    Same hydro-style kwargs as :class:`Simulator` (``init_fct`` should
    return rho, v, p components for quadrature; ``vectorpot_fct`` for A).
    """

    def __init__(
        self,
        scheme_fb: str = "SD",
        FB: bool = None,
        tolerance: float = 0.05,
        blending: bool = True,
        godunov: bool = False,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("soe", "induction")
        if FB is not None:
            if FB and "FB" not in scheme_fb:
                scheme_fb = scheme_fb + "FB"
            elif not FB and "FB" in scheme_fb:
                scheme_fb = scheme_fb.replace("FB", "")

        init = kwargs.pop("init", True)
        # Pure induction tests default to inviscid B; FV CT has no nu-closure yet.
        kwargs.setdefault("nu", 0.0)
        kwargs.setdefault("chi", 0.0)
        Simulator.__init__(self, init=False, *args, **kwargs)
        self.init = init

        if scheme_fb.upper().startswith("FV"):
            # Match :class:`finite_volume.fv_simulator.FV_Simulator`: the FV mesh has
            # one control volume per ``N[dim]``, not ``N[dim] * (p+1)`` SD subcells.
            for dim in self.dims:
                self.n[dim] = 1
                setattr(self, f"n{dim}", 1)
            self.ho_scheme = InductionFV_Scheme(
                self, riemann_solver="llf", equations="hydro"
            )
        else:
            self.ho_scheme = InductionSD_Scheme(self, riemann_solver="llf", soe="hydro")

        if "FB" in scheme_fb:
            self.scheme = InductionFallbackScheme(
                self,
                self.ho_scheme,
                FB=True,
                tolerance=tolerance,
                blending=blending,
                godunov=godunov,
            )
        else:
            self.scheme = self.ho_scheme

        if self.init:
            self.scheme.initialize()
            self.dt = self.scheme.dt
            self.create_dicts()

    def regular_mesh(self, W):
        return self.ho_scheme.interpolate_to_regular_mesh(W)

    def compute_dt(self) -> None:
        """CFL timestep from primitive velocities; mirrors :meth:`InductionSD_Scheme.compute_dt`."""
        self.ho_scheme.compute_dt()

    def output(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank == 0:
            os.makedirs(folder)
        self.comms.barrier()
        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size > 1:
            file += f"_{self.comms.rank}"
        B2 = self.ho_scheme.compute_B2()
        np.save(file, B2)
        self.outputs.append([self.time, self.noutput])
        if self.rank == 0:
            np.savetxt(folder + "/outputs.out", self.outputs)
        self.noutput += 1
