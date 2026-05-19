import numpy as np
from simulator import Simulator

from numerics.slicing import cut
from numerics.slicing import crop_fv

class Slope_limiter:
    def __init__(self,limiter):
        self.limiter = limiter
        self.compute_gradients = self.gradient_limiter(self.__getattribute__(limiter))

    def minmod(self,
               SlopeL: np.ndarray,
               SlopeR: np.ndarray,
               **kwargs)->np.ndarray:
        """
        Returns the minmod limited slopes

        Parameters
        ----------
            SlopeL/R: Solution vector with Left/Right slopes

        Returns
        -------
            Slopes: Limited slopes
        """
        #First compute ratio between slopes SlopeR/SlopeL
        #Then limit the ratio to be lower than 1
        #Finally, limit the ratio to be positive and multiply
        #  by SlopeL to get the limited slope at the cell center
        #We use "where" instead of "maximum/minimum" as it doesn't
        # propagte the NaNs caused when SlopeL=0
        ratio = SlopeR/SlopeL
        ratio = np.where(ratio<1,ratio,1)
        return np.where(ratio>0,ratio,0)*SlopeL

    def moncen(self,
               dU_L: np.ndarray,
               dU_R: np.ndarray,
               dx_L: np.ndarray,
               dx_R: np.ndarray,
               dx_M: np.ndarray)->np.ndarray:
        """
        Returns the moncen limited slopes

        Parameters
        ----------
            dU_L/R: Solution vector with Left/Right slopes
            dx_L/R: vector of cell sizes (distance between cell centers)
            dx_M:   vector of cell sizes (distance between flux points)

        Returns
        -------
            Slopes: Limited slopes
        """
        dU_C = (dx_L*dU_L + dx_R*dU_R)/(dx_L+dx_R)
        slope = np.minimum(np.abs(2*dU_L*dx_L/dx_M),np.abs(2*dU_R*dx_R/dx_M))
        slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
        return np.where(dU_L*dU_R>=0,slope,0)     

    def gradient_limiter(self,limiter):
        def limit_gradients(
            M: np.ndarray,
            h_cv: np.ndarray,
            h_fp: np.ndarray,
            idim: int,)->np.ndarray:
            dM = (M[cut(1,None,idim)] - M[cut(None,-1,idim)])/h_cv
            dMh = limiter(dM[cut(None,-1,idim)],
                          dM[cut(1,None,idim)],
                          dx_L = h_cv[cut(None,-1,idim)],
                          dx_R = h_cv[cut(1,None,idim)],
                          dx_M = h_fp[cut(1,-1,idim)])
            return dMh
        return limit_gradients 

    def compute_slopes(
            self,
            M: np.ndarray,
            h_cv: np.ndarray,
            h_fp: np.ndarray,
            idim: int,)->np.ndarray:
        """
        Returns array of limited slopes

        Parameters
        ---------- 
            M:          Solution vector (conservatives/primitives)
            h_cv:       vector of cell sizes (distance between cell centers)
            h_fp:       vector of cell sizes (distance between flux points)
            idim:       index of dimension

        Returns
        -------
            S:          Slopes of M
        """
        dMh = self.compute_gradients(M,h_cv,h_fp,idim)
        return 0.5*dMh*h_fp[cut(1,-1,idim)] 
    
def MUSCL_fluxes(self: Simulator,
                 F: dict,
                 dt: float,
                 prims=True)->None:
    """
    Returns the MUSCL scheme fluxes for conserved variales

    Parameters
    ---------- 
        self:   Simulator object
        F:      Dictionary with references to Flux array
                F = {x: Fx, y: Fy, z: Fz}
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    
    Overwrites
    ----------
        F:      Fluxes given by the Riemann solver
    """
    for dim in self.dims:
        idim=self.dims[dim]
        
        S = self.compute_slopes(self.dm.M,idim)    
        
        self.MR_fp[dim][...] = self.interpolate_R(self.dm.M,S,idim)
        self.ML_fp[dim][...] = self.interpolate_L(self.dm.M,S,idim)
        self.solve_riemann_problem(dim,F[dim],prims)
    
def compute_prediction(W: np.ndarray,
                       dWs: np.ndarray,
                       dtW: np.ndarray,
                       vels: np.array,
                       ndim: int,
                       gamma: float,
                       _d_: int,
                       _p_: int,
                       WB: bool,
                       npassive: int = 0,
                       )->None:
    """
    Returns the prediction for conserved variales

    Parameters
    ---------- 
        W:      Solution vector with primitive variables
        dWs:    Solution vector with slopes 
        vels:   vels:   array containing the indices of velocity components [vx,vy,vz]
                in the Solution array. The size of this array has to match the
                number of dimensions
        ndim:   Number of dimensions
        gamma:  Adiabatic index (ratio of specific heats)
        _d_:    Index of density in the Solution array
        _p_:    Index of pressure/energy in the Solution array
        WB:     Wheter to use Well-balanced scheme or not
    Overwrites
    ----------
        dtW:  Solution vector with predictions 
    """
    dtW[...] = 0
    for idim in range(ndim):
        vel = vels[idim]
        dW = dWs[idim]
        dtW[_d_] -= (W[vel]*dW[_d_] +       W[_d_]*dW[vel])
        dtW[_p_] -= (W[vel]*dW[_p_] + gamma*W[_p_]*dW[vel])
        dtW[vel] -= (W[vel]*dW[vel] + dW[_p_]/W[_d_])
        for vel2 in np.roll(vels,-idim)[1:]:
            dtW[vel2] -= W[vel]*dW[vel2]
        if npassive>0:
            _ps_ = _p_+1
            dtW[_ps_:_ps_+npassive] -= W[vel]*dW[_ps_:_ps_+npassive]
        if WB:
            dW = dWs[idim+ndim]
            dtW[_d_] -= (W[vel]*dW[_d_]) 
            dtW[_p_] -= (W[vel]*dW[_p_])

def MUSCL_Hancock_fluxes(self: Simulator,
                         F: dict,
                         dt: float,
                         prims=True)->None:
    """
    Parameters
    ---------- 
        self:   Simulator object
        F:      Dictionary with references to Flux array
                F = {x: Fx, y: Fy, z: Fz}
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    
    Overwrites
    ----------
        F:      Fluxes given by the Riemann solver
    """
    dMhs={}
    S={}
    nghc = getattr(self, "Nghc", 1)
    crop = lambda start, end, idim: crop_fv(start, end, idim, self.ndim, nghc)
    for dim in self.dims:
        idim=self.dims[dim]
        dMh = self.compute_gradients(self.dm.M,idim)
        #Compute and store slopes in a dictionary
        S[idim] = 0.5*dMh*self.h_fp[dim][cut(1,-1,idim)]
        #Store gradients in a dictionary
        dMhs[idim] = dMh[crop(None,None,idim)]
        if self.WB:
            dMhs[idim+self.ndim] = self.compute_gradients(self.dm.M_eq,idim)[crop(None,None,idim)]
    if self.WB:
        self.dm.M += self.dm.M_eq                    
    self.compute_prediction(self.dm.M[crop(1,-1,0)],dMhs)
    if self.WB:
        if self.potential:
            drho = ((self.dm.M[0]-self.dm.M_eq[0])/self.dm.M[0])[crop(1,-1,0)]
            for vel in self.vels[:self.ndim]:
                self.dm.dtM[vel][crop(1,-1,0)] += drho[crop(1,-1,0)]*self.dm.grad_phi[vel-1]
        #We move back to the perturbation
        self.dm.M -= self.dm.M_eq
    self.dm.M[crop(1,-1,0)] += 0.5*self.dm.dtM*dt
    
    for dim in self.dims:
        idim=self.dims[dim]
        self.MR_fp[dim][...] = self.interpolate_R(self.dm.M,S[idim],idim)
        self.ML_fp[dim][...] = self.interpolate_L(self.dm.M,S[idim],idim)
        self.solve_riemann_problem(dim,F[dim],prims)

