"""
Base class for semi-discrete spatial discretization schemes.

A semi-discrete scheme handles the spatial discretization and flux
computation, producing dU/dt = L(U) where L is the spatial operator.
The scheme is coupled with a time integrator by the Simulator.

The scheme **owns** the GPUDataManager (``dm``) that stores all
spatial arrays (solution, fluxes, mesh coordinates, etc.).  The
parent Simulator accesses the data manager through the scheme.

Attribute Proxy Pattern:
    The scheme stores a reference to the parent simulator (_sim).
    Any attribute not found on the scheme itself is automatically
    looked up on the simulator via __getattr__. This allows the
    scheme methods to access shared state (mesh, equations, etc.)
    transparently while keeping scheme-specific state local.
"""

import numpy as np
from runtime.data_management import GPUDataManager


class SemiDiscreteScheme:
    """
    Base class for semi-discrete spatial discretization schemes.
    
    Subclasses implement the spatial operator L(U) for specific
    discretization methods (Spectral Difference, Finite Volume, etc.).
    
    Parameters
    ----------
    sim : Simulator
        Parent simulator providing shared state (mesh, equations,
        physics parameters, etc.)
    """

    def __init__(self, sim):
        object.__setattr__(self, '_sim', sim)
        object.__setattr__(self, 'dm', GPUDataManager(sim.use_cupy))

    def __getattr__(self, name):
        """Proxy attribute access to the parent simulator."""
        return getattr(self._sim, name)

    # ----------------------------------------------------------------
    # Core spatial interface
    # ----------------------------------------------------------------

    def initialize(self):
        """Set up scheme-specific arrays, initial conditions, meshes."""
        pass

    def init_Boundaries(self):
        """Initialize the boundaries of the scheme."""
        pass

    def compute_update(self, U, ader=False, prims=False, c_l=0.0, dt=None):
        """
        Compute the spatial right-hand side L(U).

        Parameters
        ----------
        U : np.ndarray
            Current solution state.
        ader : bool
            Whether operating in ADER mode (multi-time-point arrays).
        prims : bool
            Whether U contains primitive variables.
        c_l : float
            Butcher tableau c coefficient for the current RK stage.
            Represents the fraction of dt at which this stage is evaluated.
            Useful for time-dependent source terms or boundary conditions.
        dt : float or None
            Full time step size. None when not applicable (e.g. ADER).

        Returns
        -------
        dUdt : np.ndarray
            Spatial operator applied to U.
        """
        raise NotImplementedError

    def compute_dt(self):
        """Compute and store the CFL-limited time step in self.dt."""
        raise NotImplementedError

    def get_solution(self, ader=False):
        """Return the current solution array."""
        raise NotImplementedError

    def set_solution(self, U):
        """Set the solution array."""
        raise NotImplementedError

    def update_solution(self, dU):
        """Apply an increment dU to the solution."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # ADER interface
    # ----------------------------------------------------------------

    def ader_string(self) -> str:
        """Return the string to be used in the einsum performed to compute the ADER update."""
        raise NotImplementedError

    def ader_predictor(self, prims: bool = False) -> None:
        """Perform the ADER predictor step (Picard iteration)."""
        self.dm.U_ader[...] = self.get_solution(ader=True)
        for ader_iter in range(self.integrator.nader):
            dUdt = self.compute_update(
                self.dm.U_ader, ader=True, prims=prims
            )
            if ader_iter < self.m:
                s = self.ader_string()
                self.dm.U_ader[...] = (
                    np.einsum(
                        f"np,up{s}->un{s}", self.dm.invader, dUdt
                    )
                    * self.dt
                )
                self.dm.U_ader[...] = (
                    self.get_solution(ader=True) - self.dm.U_ader
                )

    def ader_update(self):
        dUdt = self.compute_dudt(self.dm.U_ader, ader=True)
        s = self.ader_string()
        dU = np.einsum(f"t,ut{s}->u{s}", self.dm.w_tp, dUdt) * self.dt
        self.update_solution(dU)

    # ----------------------------------------------------------------
    # Integrator support
    # ----------------------------------------------------------------

    def allocate_arrays(self, ader=False):
        """
        Allocate arrays required by the time integrator.
        """ 
        self.integrator.allocate_arrays(self)
        """Allocate scheme working arrays for flux computations."""
        for dim in self.dims:
            # Conservative/Primitive variables at flux points
            self.dm.__setattr__(f"M_fp_{dim}", self.array_fp(dims=dim, ader=ader))
            # Conservative fluxes at flux points
            self.dm.__setattr__(f"F_fp_{dim}", self.array_fp(dims=dim, ader=ader))
            # Arrays to solve Riemann problem at the interface between elements
            self.dm.__setattr__(f"ML_fp_{dim}", self.array_RS(dim=dim, ader=ader))
            self.dm.__setattr__(f"MR_fp_{dim}", self.array_RS(dim=dim, ader=ader))
            # Arrays to communicate boundary values
            self.dm.__setattr__(f"BC_fp_{dim}", self.array_BC(dim=dim, ader=ader))

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def create_dicts(self):
        """Create dimension-keyed dictionaries for convenient array access."""
        pass

    def convert_solution(self, W=False):
        """Convert between conservative/primitive or point representations."""
        pass

    def compute_primitives_cv(self, U):
        """
        Compute primitive variables from conservatives, handling
        the well-balanced case when WB is enabled.
        """
        if self.WB:
            return (
                self.compute_primitives(U + self.dm.U_eq_cv)
                - self.compute_primitives(self.dm.U_eq_cv)
            )
        else:
            return self.compute_primitives(U)

    def post_update(self):
        """Called after the time integrator completes a step."""
        pass
