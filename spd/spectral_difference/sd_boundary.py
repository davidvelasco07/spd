from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sd_simulator import SD_Simulator
from spd.numerics.slicing import cut
from spd.numerics.slicing import indices
from spd.numerics.slicing import indices2
   
def store_interfaces(self: SD_Simulator,
                     M: np.ndarray,
                     dim: str) -> None:
    """
    Stores the values of flux points at the extremes of elements(0,-1)
    These arrays are then used to solve the Riemann problem
    """
    shift=self.ndim+self.dims[dim]-1
    axis = -(self.dims[dim]+1)
    self.MR_fp[dim][cut(None,-1,shift)] = M[indices( 0,self.dims[dim])]
    self.ML_fp[dim][cut(1 ,None,shift)] = M[indices(-1,self.dims[dim])]

def apply_interfaces(self: SD_Simulator,
                     F: np.ndarray,
                     F_fp: np.ndarray,
                     dim: str):
    """
    Applies the values of flux points at the extremes of elements(0,-1)
    This is done after the Riemann problem at element interfaces has been
    solved. 
    """
    shift=self.ndim+self.dims[dim]-1
    F_fp[indices( 0,self.dims[dim])] = F[cut(None,-1,shift)]
    F_fp[indices(-1,self.dims[dim])] = F[cut(1, None,shift)]

def store_BC(self: SD_Simulator,
             BC_array: np.ndarray,
             M: np.ndarray,
             dim: str) -> None:
    """
    Stores the solution at flux points for the extremes of the domain
    These boundary arrays can then be communicated between domains
    """    
    idim = self.dims[dim]
    BC = self.BC[dim]
    for side in [0,1]:
        if  BC[side] == "periodic":
            BC_array[side] = M[indices2(side-1,self.ndim,idim)]
        elif BC[side] == "reflective":
            BC_array[side] = M[indices2(-side,self.ndim,idim)]
            BC_array[side,self.vels[idim]] *= -1
        elif BC[side] == "gradfree":
            BC_array[side] = M[indices2(-side,self.ndim,idim)]
        elif BC[side] == "ic":
            next
        elif BC[side] == "eq":
            next
        elif BC[side] == "pressure":
            #Overwrite solution with ICs
            M[indices2(-side,self.ndim,idim)] = BC_array[side]
        else:
            raise("Undetermined boundary type")
                         
def apply_BC(self: SD_Simulator,
             dim: str) -> None:
    """
    Fills up the missing first column of M_L
    and the missing last column of M_R
    """
    shift=self.ndim+self.dims[dim]-1
    self.ML_fp[dim][indices( 0,shift)] = self.BC_fp[dim][0]
    self.MR_fp[dim][indices(-1,shift)] = self.BC_fp[dim][1]

def Boundaries(self: SD_Simulator,
                  M: np.ndarray,
                  dim: str):
    store_BC(self,self.BC_fp[dim],M,dim)
    self.Comms_fp(M,dim)
    apply_BC(self,dim)
