from spd.torlo.ADER import ADER
from .integrator import Integrator
import numpy as np
from spd.numerics.polynomials import gauss_legendre_quadrature
from spd.numerics.polynomials import ader_matrix

class ADER_Integrator(Integrator):
    def __init__(
        self,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.ader = True

    def ader_string(self)->str:
        """
        Returns a string to be used in the
        einsum performed to compute the ADER update.
        The string length depends on the dimensions
        """
        if self.ndim==3:
            return "zyxkji"
        elif self.ndim==2:
            return "yxji"
        else:
            return "xi"

    def allocate_arrays(self, target) -> None:
        """
        Allocate ADER matrices and arrays on the target object.
        
        Parameters
        ----------
        target : SemiDiscreteScheme or Simulator
            The object that holds dm and time_integrator_arrays().
        """
        ## ADER matrix
        self.ader  = ADER(-1,self.m+1,'gaussLegendre')
        self.nader = self.ader.M_sub+1
        target.dm.invader = self.ader.evolMat
        target.dm.w_tp = self.ader.bADER.flatten()
        target.nader = self.nader
        #Allocate arrays for the time integrator
        target.dm.U_ader = target.array_sp(ader=True)
        fn = getattr(target, "allocate_ader_extras", None)
        if callable(fn):
            fn()


    def update(self, target) -> None:
        """
        Perform one ADER time step: predictor then update.
        
        Parameters
        ----------
        target : Simulator or SemiDiscreteScheme
            Object with ader_predictor() and ader_update() methods.
        """
        target.ader_predictor()
        """Perform the ADER corrector/update step."""
        target.ader_update()
