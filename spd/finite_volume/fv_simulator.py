"""
Finite Volume Simulator.

Thin wrapper around FV_Scheme + Simulator that provides backward
compatibility.  Creates an FV_Scheme and connects it to the Simulator.
"""

import numpy as np

from spd.simulator import Simulator
from .fv_scheme import FV_Scheme
from spd.numerics.polynomials import solution_points, flux_points


class FV_Simulator(Simulator):
    """
    Simulator using Finite Volume spatial discretization with
    MUSCL / MUSCL-Hancock reconstruction.

    Creates an FV_Scheme internally and delegates all spatial
    operations to it while the Simulator handles time integration,
    I/O, and the simulation lifecycle.

    Parameters
    ----------
    riemann_solver : str
        Riemann solver name ('llf', 'hllc', 'lhllc').
    slope_limiter : str
        Slope limiter name ('minmod', 'moncen').
    predictor : bool
        If True use MUSCL-Hancock, else plain MUSCL.
    *args, **kwargs
        Forwarded to Simulator.__init__().
    """

    def __init__(
        self,
        riemann_solver: str = "llf",
        slope_limiter: str = "minmod",
        scheme: str = "MUSCL-Hancock",
        time_integrator: str = "rk",
        *args,
        **kwargs,
    ):
        Simulator.__init__(
            self, time_integrator=time_integrator, *args, **kwargs
        )

        # FV mesh has no subcells: override n=1 and use p=0 reference points
        for dim in self.dims:
            self.n[dim] = 1
        self.sp = {dim: solution_points(0.0, 1.0, 0) for dim in self.dims}
        self.fp = {dim: flux_points(0.0, 1.0, 0) for dim in self.dims}

        # Create the semi-discrete FV scheme
        self.scheme = FV_Scheme(
            self,
            riemann_solver=riemann_solver,
            slope_limiter=slope_limiter,
            scheme=scheme,
        )

        self.scheme.initialize()
        # Sync dt from scheme to simulator
        self.dt = self.scheme.dt

    def regular_mesh(self, W):
        return W

    def transpose_to_fv(self, W):
        return W

    @property
    def domain_size(self):
        return self.scheme.domain_size

    # Delegate FV-specific methods to the scheme for backward compat
    def __getattr__(self, name):
        # Called only when normal lookup fails
        if name.startswith('_') or name == 'scheme':
            raise AttributeError(name)
        scheme = object.__getattribute__(self, '__dict__').get('scheme')
        if scheme is not None:
            # Probe the scheme's real attributes only. Using hasattr/getattr
            # here would trigger the scheme's own __getattr__, which proxies
            # back to this simulator and causes infinite recursion for names
            # that exist on neither object.
            try:
                return object.__getattribute__(scheme, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
