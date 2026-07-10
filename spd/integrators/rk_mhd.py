"""
Explicit Runge-Kutta for coupled MHD: the conserved state (solution points)
and the face-staggered CT magnetic field advance together with the same
Butcher table.

Per stage:
  1. build the stage state  U_s = U0 - dt sum_j A[s,j] K_j  and the stage
     face field  B_s = B0 - dt sum_j A[s,j] K_B_j;
  2. the scheme synchronizes its internal state (B rows of U_s from the
     divergence-free stage B, stage primitives for the edge solver);
  3. the scheme returns the (possibly MOOD-corrected) spatial RHS for U and
     for B (edge-E curl) of the same stage.

The final combination applies the b-weights to both families, then the CT
field is re-projected onto the cell-centered B rows (B_to_U).
"""

from .rk import RK_Integrator, stage_state, weighted_sum


class MHDRKIntegrator(RK_Integrator):
    """RK stages for U (``U_stage``/``K_s``) plus the face B field
    (``B_stage_{dim}``/``K_B_{s}_{dim}``, allocated on the primary's dm)."""

    @staticmethod
    def _primary(target):
        prim = getattr(target, "primary", None)
        return prim if prim is not None else target

    def allocate_arrays(self, target) -> None:
        super().allocate_arrays(target)
        prim = self._primary(target)
        for dim in prim.dims:
            prim.dm.__setattr__(
                f"B_stage_{dim}", prim.induction_b_staging_array(dim)
            )
            for stage in range(self.nstages):
                prim.dm.__setattr__(
                    f"K_B_{stage}_{dim}", prim.induction_b_staging_array(dim)
                )

    def update(self, target) -> None:
        dt = target.dt
        prim = self._primary(target)
        pdm = prim.dm
        dims = prim.dims
        B0 = {dim: pdm.__getattribute__(f"B{dim}_fp").copy() for dim in dims}
        U0 = target.get_solution()

        for stage in range(self.nstages):
            terms = [
                (target.dm.__getattribute__(f"K_{j}"), dt * self.A[stage, j])
                for j in range(stage)
                if self.A[stage, j] != 0.0
            ]
            stage_state(target.dm.U_stage, U0, terms)
            for dim in dims:
                Bs = pdm.__getattribute__(f"B_stage_{dim}")
                Bs[...] = B0[dim]
                for j in range(stage):
                    if self.A[stage, j] != 0.0:
                        Bs -= dt * self.A[stage, j] * pdm.__getattribute__(
                            f"K_B_{j}_{dim}"
                        )
                # In place: the scheme's B_fp dict aliases these buffers.
                pdm.__getattribute__(f"B{dim}_fp")[...] = Bs
            prim.set_stage_state(target.dm.U_stage)
            Ks = target.dm.__getattribute__(f"K_{stage}")
            Ks[...] = target.compute_update(
                target.dm.U_stage, ader=False, c_l=self.c[stage], dt=dt
            )
            K_B = target.compute_B_update()
            for dim in dims:
                pdm.__getattribute__(f"K_B_{stage}_{dim}")[...] = K_B.get(
                    dim, 0.0
                )

        terms = [
            (target.dm.__getattribute__(f"K_{stage}"), dt * self.b[stage])
            for stage in range(self.nstages)
        ]
        weighted_sum(target.dm.U_stage, terms)
        for dim in dims:
            Bfp = pdm.__getattribute__(f"B{dim}_fp")
            Bfp[...] = B0[dim]
            for stage in range(self.nstages):
                Bfp -= dt * self.b[stage] * pdm.__getattribute__(
                    f"K_B_{stage}_{dim}"
                )
        target.update_solution(target.dm.U_stage)
        # Re-project the (divergence-free) CT field onto the cell-centered
        # B rows of the conserved state (scheme-specific: SD solution points
        # + cell averages, or the FV cell averages directly).
        prim.B_to_U()
