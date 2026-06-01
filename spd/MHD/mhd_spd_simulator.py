"""
MHD on SPD (SD + fallback) with coupled ADER predictor for fluid + induction.
"""

import numpy as np

from sdfb_simulator import SPD_Simulator
from induction.induction_sd_scheme import InductionSD_Scheme, MHDInductionSD_Scheme
from riemann_solvers.riemann_solver_1D import Riemann_solver_1D as rs1d
import mhd as mhd_eq


class MHDCoupledScheme:
    """
    Wraps the hydro time-step object (``FallbackScheme`` or bare ``SD_Scheme``)
    together with the shared :class:`MHDInductionSD_Scheme` ``ho`` for fused
    ``ader_predictor`` / ``ader_update`` + ``B_to_U``.
    """

    def __init__(self, sim, hydro_scheme, ho: MHDInductionSD_Scheme):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_hydro", hydro_scheme)
        object.__setattr__(self, "_ho", ho)

    @property
    def dm(self):
        return self._hydro.dm

    def __getattr__(self, name):
        if name in ("_sim", "_hydro", "_ho"):
            raise AttributeError(name)
        return getattr(self._hydro, name)

    def ader_predictor(self, prims: bool = False):
        sim = self._sim
        ho = self._ho
        na = sim.dm.xp.newaxis
        sim.dm.U_ader[...] = sim.dm.U_sp[:, na, ...]
        for dim in sim.dims:
            B = sim.dm.__getattribute__(f"B{dim}_fp")
            ho.B_ader_fp[dim][...] = B[na]
            sim.dm.U_ader[sim.b[dim]][...] = ho.compute_sp_from_fp(B[na], dim=dim)

        for ader_iter in range(sim.m + 1):
            ho.solve_faces(sim.dm.U_ader, prims=prims, ader=True)
            ho.solve_edges(ader_iter)
            if sim.nu > 0:
                InductionSD_Scheme.add_nabla_terms(ho)
            if ader_iter < sim.m:
                s = ho.ader_string()
                dUdt = ho.compute_dudt(sim.dm.U_ader, ader=True)
                sim.dm.U_ader[...] = (
                    np.einsum(f"np,up{s}->un{s}", sim.dm.invader, dUdt) * sim.dt
                )
                sim.dm.U_ader[...] = sim.dm.U_sp[:, na] - sim.dm.U_ader
                for dim in sim.dims:
                    ho.B_ader_fp[dim] = np.einsum(
                        f"np,p{s}->n{s}", sim.dm.invader, ho.ader_dBdt(dim)
                    )
                    ho.B_ader_fp[dim][...] = ho.B_fp[dim][na] - ho.B_ader_fp[dim]

    def ader_update(self):
        self._hydro.ader_update()
        self._ho.ader_update()
        self._sim.B_to_U()

    def post_update(self):
        if hasattr(self._hydro, "post_update"):
            self._hydro.post_update()


class MHD_SPD_Simulator(SPD_Simulator):
    """
    Same constructor as :class:`SPD_Simulator` plus MHD induction on the SD mesh.
    """

    def __init__(
        self,
        equations=mhd_eq,
        riemann_solver_sd: str = "llf_mhd",
        *args,
        **kwargs,
    ):
        kwargs.setdefault("soe", "mhd")
        self.equations = equations
        self.godunov = kwargs.pop("godunov", False)
        super().__init__(
            *args,
            ho_scheme_cls=MHDInductionSD_Scheme,
            riemann_solver_sd=riemann_solver_sd,
            **kwargs,
        )
        self._hydro_scheme = self.scheme
        self.scheme = MHDCoupledScheme(self, self._hydro_scheme, self.ho_scheme)

    def B_to_U(self):
        ho = self.ho_scheme
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            self.dm.U_sp[self.b[dim]] = ho.compute_sp_from_fp(B[np.newaxis], dim=dim)[0]

    def compute_dt(self) -> None:
        W = self.dm.W_cv
        c_max = 0
        for dim in self.dims:
            dim1, dim2 = self.ho_scheme.other_dims(dim)
            c_max = c_max + self.equations.compute_fast_vel(
                W[self._p_],
                W[self._d_],
                W[self.b[dim]],
                W[self.b[dim1]] if dim1 in self.dims else 0,
                W[self.b[dim2]] if dim2 in self.dims else 0,
                self.gamma,
                self.min_c2,
            )
        c_max = np.max(c_max)
        h = self.h_min / (self.p + 1)
        dt = h / c_max
        dt = self.comms.reduce_min(dt).item()
        if self.viscosity and self.nu > 0:
            nu = max(self.nu, self.chi)
            dt_nu = (0.25 * self.h_min / (self.p + 1)) ** 2 / nu
            dt = min(dt, dt_nu)
        self.dt = self.cfl_coeff * dt


SDFB_MHD_Simulator = MHD_SPD_Simulator
SDADER_MHD_Simulator = MHD_SPD_Simulator
