"""
Spectral Difference Simulator.

Thin wrapper around SD_Scheme + Simulator that provides backward
compatibility.  Creates an SD_Scheme and connects it to the Simulator.
"""

import numpy as np

from spd.simulator import Simulator
from .sd_scheme import SD_Scheme


class SD_Simulator(Simulator):
    """
    Simulator using Spectral Difference spatial discretization.

    Creates an SD_Scheme internally and delegates all spatial
    operations to it while the Simulator handles time integration,
    I/O, and the simulation lifecycle.

    Parameters
    ----------
    riemann_solver : str
        Riemann solver name for the SD faces ('llf', 'hllc', 'lhllc').
    *args, **kwargs
        Forwarded to Simulator.__init__().
    """

    def __init__(
        self,
        riemann_solver: str = "llf",
        soe: str = "hydro",
        *args,
        **kwargs,
    ):
        Simulator.__init__(self, *args, **kwargs)

        # Create the semi-discrete SD scheme
        self.scheme = SD_Scheme(self, riemann_solver=riemann_solver, soe=self.soe)

        if self.init:
            self.scheme.initialize()
            # Sync dt from scheme to simulator
            self.dt = self.scheme.dt

    # ------------------------------------------------------------------
    # Convenience accessors that forward to the scheme
    # ------------------------------------------------------------------

    @property
    def domain_size(self):
        return self.scheme.domain_size

    def regular_mesh(self, W):
        return self.scheme.interpolate_to_regular_mesh(W)

    def transpose_to_fv(self, M):
        return self.scheme.transpose_to_fv(M)

    def transpose_to_sd(self, M):
        return self.scheme.transpose_to_sd(M)

    # Expose scheme dicts at simulator level for backward compat
    @property
    def M_fp(self):
        return self.scheme.M_fp

    @property
    def F_fp(self):
        return self.scheme.F_fp

    @property
    def MR_fp(self):
        return self.scheme.MR_fp

    @property
    def ML_fp(self):
        return self.scheme.ML_fp

    @property
    def BC_fp(self):
        return self.scheme.BC_fp

    # Delegate SD-specific methods to the scheme for backward compat
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
