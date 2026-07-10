"""
Coupled fluid + constrained-transport spectral-difference scheme for ideal MHD.

``MHD_SD_Scheme`` is the single high-order scheme of an MHD run: it owns both
the hydro state (``U_sp`` with cell-centered B rows) and the face-staggered
CT magnetic field of :class:`MHDInductionSD_Scheme`, and fuses their ADER
predictor / corrector so the Picard iteration solves fluid and induction
simultaneously.
"""

import numpy as np

from spd.schemes.scheme import SemiDiscreteScheme
from spd.induction.induction_sd_scheme import (
    InductionSD_Scheme,
    MHDInductionSD_Scheme,
)


class MHD_SD_Scheme(MHDInductionSD_Scheme):
    """
    SD scheme for ideal MHD: hydro SD fluxes + edge-based constrained
    transport, with a coupled ADER predictor/corrector and a fast
    magnetosonic CFL condition.
    """

    # The scheme-level dt is an alias of the simulator's: ader_dBdt and the
    # coupled predictor scale by dt, and a private copy would go stale when
    # the simulator recomputes or clamps the step.
    @property
    def dt(self):
        return self._sim.dt

    @dt.setter
    def dt(self, value):
        self._sim.dt = value

    def compute_dt(self) -> None:
        """CFL condition on the fast magnetosonic speed (summed over dims)."""
        sim = self._sim
        W = self.dm.W_cv
        c_max = 0
        for dim in self.dims:
            dim1, dim2 = self.other_dims(dim)
            c_max = c_max + self.equations.compute_fast_vel(
                W[sim._p_],
                W[sim._d_],
                W[sim.b[dim]],
                W[sim.b[dim1]] if dim1 in self.dims else 0,
                W[sim.b[dim2]] if dim2 in self.dims else 0,
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

    def B_to_U(self):
        """Overwrite the cell-centered B rows of ``U_sp`` with the projection
        of the (divergence-free) face-staggered CT field, and refresh the
        corresponding cell averages of ``U_cv`` (they drive the next step's
        CFL condition and trouble detection)."""
        sim = self._sim
        b_rows = [sim.b[dim] for dim in self.dims]
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            self.dm.U_sp[sim.b[dim]] = self.compute_sp_from_fp(
                B[np.newaxis], dim=dim
            )[0]
        self.dm.U_cv[b_rows] = self.compute_cv_from_sp(self.dm.U_sp[b_rows])

    def ader_predictor(self, prims: bool = False) -> None:
        """Coupled Picard iteration for the fluid state and the face B field."""
        sim = self._sim
        na = self.dm.xp.newaxis
        self.dm.U_ader[...] = self.dm.U_sp[:, na, ...]
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            self.B_ader_fp[dim][...] = B[na]
            self.dm.U_ader[sim.b[dim]][...] = self.compute_sp_from_fp(B[na], dim=dim)

        for ader_iter in range(self.m + 1):
            self.solve_faces(self.dm.U_ader, prims=prims, ader=True)
            self.solve_edges(ader_iter)
            if self.viscosity and self.nu > 0:
                # Resistive correction to the edge E-field (the hydro viscous
                # fluxes were already added inside solve_faces).
                InductionSD_Scheme.add_nabla_terms(self)
            if ader_iter < self.m:
                s = self.ader_string()
                dUdt = self.compute_dudt(self.dm.U_ader, ader=True)
                self.dm.U_ader[...] = (
                    np.einsum(f"np,up{s}->un{s}", self.dm.invader, dUdt) * self.dt
                )
                self.dm.U_ader[...] = self.dm.U_sp[:, na] - self.dm.U_ader
                for dim in self.dims:
                    self.B_ader_fp[dim] = np.einsum(
                        f"np,p{s}->n{s}", self.dm.invader, self.ader_dBdt(dim)
                    )
                    self.B_ader_fp[dim][...] = (
                        self.B_fp[dim][na] - self.B_ader_fp[dim]
                    )

    def ader_update(self):
        """Coupled corrector: hydro state update, CT update of the face B
        field, then re-projection of B onto the cell-centered rows of U."""
        # The induction parent overrides ader_update with the CT-only update,
        # so invoke the generic semi-discrete corrector explicitly for U.
        SemiDiscreteScheme.ader_update(self)
        InductionSD_Scheme.ader_update(self)
        self.B_to_U()

    # ----------------------------------------------------------------
    # Runge-Kutta (method-of-lines) interface
    # ----------------------------------------------------------------

    def set_stage_state(self, U_stage):
        """Synchronize the internal state with an RK stage: the B rows of
        ``U_stage`` are re-projected from the (divergence-free) stage face
        field already written into ``B{dim}_fp``, and ``W_sp`` is refreshed
        (the edge solver interpolates primitives from it)."""
        sim = self._sim
        na = np.newaxis
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            U_stage[sim.b[dim]] = self.compute_sp_from_fp(B[na], dim=dim)[0]
        self.dm.W_sp[...] = self.compute_primitives(U_stage)

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Stage RHS for U; under RK (ader=False) it also solves the edges
        of the same stage and stores the face-B RHS (edge-E curl) for
        ``compute_B_update``."""
        dUdt = super().compute_update(U, ader=ader, prims=prims, **kwargs)
        if not ader:
            self.solve_edges(0)
            if self.viscosity and self.nu > 0:
                InductionSD_Scheme.add_nabla_terms(self)
            self._K_B = {dim: self.dBdt_dim(dim) for dim in self.dims}
        return dUdt

    def compute_B_update(self):
        """Face-B RHS of the stage prepared by ``compute_update``."""
        return self._K_B
