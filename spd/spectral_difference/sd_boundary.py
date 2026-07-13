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
        elif BC[side] == "doublemach":
            store_doublemach_BC(self, BC_array, M, dim, side)
        else:
            raise("Undetermined boundary type")


def _dmr_conservative(self, primitive_state, nd):
    """Conservative state vector (length nvar) broadcast over nd axes.

    SD flux-point boundaries operate on conservative variables.
    """
    xp = self.dm.xp
    rho = primitive_state["rho"]
    vx, vy, P = primitive_state["vx"], primitive_state["vy"], primitive_state["P"]
    U = xp.zeros(self.nvar)
    U[self._d_] = rho
    U[self.vels[0]] = rho * vx
    U[self.vels[1]] = rho * vy
    U[self._p_] = P / (self.gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)
    return U.reshape((self.nvar,) + (1,) * (nd - 1))


def store_doublemach_BC(self: SD_Simulator,
                        BC_array: np.ndarray,
                        M: np.ndarray,
                        dim: str,
                        side: int) -> None:
    """Double Mach Reflection boundary for the SD flux-point arrays.

    x: left = inflow (post-shock state), right = outflow.
    y: lower = reflecting wall for x >= xc, post-shock state for x < xc;
       upper = the (moving, tilted) shock state for x < x_s(t), ambient
       otherwise.

    The post-shock / ambient states are imposed explicitly so the BC is
    self-contained (it does not rely on frozen IC ghost values).
    """
    from spd.initial_conditions.initial_conditions_2d import (
        DMR_XC,
        DMR_ANGLE,
        DMR_SHOCK_SPEED,
        dmr_post_shock,
        dmr_ambient,
    )

    xp = self.dm.xp
    idim = self.dims[dim]
    nd = BC_array[side].ndim
    post = _dmr_conservative(self, dmr_post_shock(self.gamma), nd)

    if dim == "x":
        # Left = inflow (post-shock); right = outflow (zero-gradient copy).
        if side == 0:
            BC_array[0] = post
        else:
            BC_array[1] = M[indices2(-1, self.ndim, idim)]
        return

    # dim == "y"
    x_sp = self.dm.X_sp                    # (N_x, p+1)
    xb = x_sp.reshape((1,) * (nd - 2) + x_sp.shape)
    xc = getattr(self, "dmr_xc", DMR_XC)

    if side == 0:
        # Reflecting wall for x >= xc; post-shock inflow for x < xc.
        refl = M[indices2(0, self.ndim, idim)].copy()
        refl[self.vels[idim]] *= -1
        BC_array[0] = xp.where(xb >= xc, refl, post)
    else:
        angle = getattr(self, "dmr_angle", DMR_ANGLE)
        speed = getattr(self, "dmr_shock_speed", DMR_SHOCK_SPEED)
        t = float(self.time)
        ytop = self.lim["y"][1]           # upper boundary flux points sit at y = ytop
        x_s = speed * t / np.sin(angle) + xc + ytop / np.tan(angle)
        ambient = _dmr_conservative(self, dmr_ambient(self.gamma), nd)
        BC_array[1] = xp.where(xb < x_s, post, ambient)
                         
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
