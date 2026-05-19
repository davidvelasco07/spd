"""
Explicit Runge–Kutta for induction (face-centered B), staggered storage per dim.
"""

import numpy as np

from .rk import RK_Integrator


class InductionRKIntegrator(RK_Integrator):
    """RK stages use ``B_stage_{dim}`` and ``K_B_{stage}_{dim}`` on ``dm``."""

    def allocate_arrays(self, target) -> None:
        buf_fn = getattr(target, "induction_b_staging_array", None)
        for dim in target.dims:
            buf = buf_fn(dim) if buf_fn is not None else target.array_fp(dims=dim)[0]
            target.dm.__setattr__(f"B_stage_{dim}", buf)
        for stage in range(self.nstages):
            for dim in target.dims:
                buf = (
                    buf_fn(dim) if buf_fn is not None else target.array_fp(dims=dim)[0]
                )
                target.dm.__setattr__(f"K_B_{stage}_{dim}", buf)

    def update(self, target) -> None:
        dt = target.dt
        dims = target.dims
        U0 = {dim: target.dm.__getattribute__(f"B{dim}_fp").copy() for dim in dims}

        for stage in range(self.nstages):
            for dim in dims:
                Bs = target.dm.__getattribute__(f"B_stage_{dim}")
                Bs[...] = U0[dim]
                for j in range(stage):
                    Kj = target.dm.__getattribute__(f"K_B_{j}_{dim}")
                    Bs[...] -= dt * self.A[stage, j] * Kj
                target.dm.__getattribute__(f"B{dim}_fp")[...] = Bs
            target.solve_edges(0)
            if target.nu > 0:
                target.add_nabla_terms()
            for dim in dims:
                Ks = target.dm.__getattribute__(f"K_B_{stage}_{dim}")
                Ks[...] = target.dBdt_dim(dim)

        for dim in dims:
            Bfp = target.dm.__getattribute__(f"B{dim}_fp")
            Bfp[...] = U0[dim]
            inc = sum(
                self.b[s]
                * target.dm.__getattribute__(f"K_B_{s}_{dim}")
                for s in range(self.nstages)
            )
            Bfp[...] -= dt * inc
