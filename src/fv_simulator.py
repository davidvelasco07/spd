from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from collections import defaultdict
from data_management import CupyLocation
from data_management import GPUDataManager
from simulator import Simulator
import riemann_solver as rs
import muscl

from slicing import cut, crop_fv

class FV_Simulator(Simulator):
    def __init__(
        self,
        riemann_solver_fv: Callable = rs.llf,
        slope_limiter: str = "minmod",
        predictor: bool = True,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.riemann_solver_fv = riemann_solver_fv
        self.predictor = predictor
        self.slope_limiter = muscl.Slope_limiter(slope_limiter)

    def array_FV(self,nvar,dim=None,ngh=0)->np.ndarray:
        shape = [nvar] 
        N=[]
        for dim2 in self.dims2:
            N.append(self.N[dim2]+(dim==dim2)+2*ngh)
        return np.ndarray(shape+N[::-1])
    
    def array_FV_BC(self,dim="x")->np.ndarray:
        shape = [2,self.nvar]
        ngh=self.Nghc
        N=[]
        for dim2 in self.dims2:
            N.append(self.N[dim2]+2*ngh if dim!=dim2 else ngh)
        return np.ndarray(shape+N[::-1])
    
    def fv_arrays(self)->None:
        self.dm.M_fv  = self.array_FV(self.p+1,self.nvar,ngh=self.Nghc)
        self.dm.U_new = self.array_FV(self.p+1,self.nvar)
        if self.predictor:
            self.dm.dtM = self.array_FV(self.p+1,self.nvar,ngh=self.Nghc-1)
        for dim in self.dims2:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"F_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"F_faces_FB{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"MR_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"ML_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"BC_fv_{dim}",self.array_FV_BC(dim=dim))

    def create_dicts_fv(self)->None:
        self.F_faces = defaultdict(list)
        self.F_faces_FB = defaultdict(list)
        self.MR_faces = defaultdict(list)
        self.ML_faces = defaultdict(list)
        self.BC_fv = defaultdict(list)
        
        for dim in self.dims2:
            self.faces[dim] = self.dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = self.dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = self.dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = self.dm.__getattribute__(f"d{dim}_cv")
            self.F_faces[dim] = self.dm.__getattribute__(f"F_faces_{dim}")
            self.F_faces_FB[dim] = self.dm.__getattribute__(f"F_faces_FB{dim}")
            self.MR_faces[dim] = self.dm.__getattribute__(f"MR_faces_{dim}")
            self.ML_faces[dim] = self.dm.__getattribute__(f"ML_faces_{dim}")
            self.BC_fv[dim] = self.dm.__getattribute__(f"BC_fv_{dim}")
    
    def compute_slopes(self,
                       M: np.ndarray,
                       idim: int,
                       )->np.ndarray:
        return self.slope_limiter.compute_slopes(M,
                           self.h_cv[self.dims[idim]],
                           self.h_fp[self.dims[idim]],
                           idim)
    
    def compute_gradients(self,
                       M: np.ndarray,
                       idim: int,
                       )->np.ndarray: 
        return self.slope_limiter.compute_gradients(M,
                             self.h_cv[self.dims[idim]],
                             self.h_fp[self.dims[idim]],
                             idim)

    def interpolate_R(self,
                      M: np.ndarray,
                      S: np.ndarray,
                      idim: int)->np.ndarray:
        """
        args: 
            M:      Solution vector (conservatives/primitives)
            idim:   index of dimension
        returns:
            MR:     Values interpolated to the right
        """
        #MR = M - SlopeC*h/2
        ngh=self.Nghc
        crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
        return M[crop(ngh,-1,idim)] - S[crop( 1,None,idim)]

    def interpolate_L(self,
                      M: np.ndarray,
                      S: np.ndarray,
                      idim: int)->np.ndarray:
        """
        args: 
            self:   Simulator object
            M:      Solution vector (conservatives/primitives)
            idim:   index of dimension
        returns:
            MR:     Values interpolated to the left
        """
        #ML = M + SlopeC*h/2
        ngh=self.Nghc
        crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
        return M[crop(1,-ngh,idim)] + S[crop(None,-1,idim)]  
    
    def compute_prediction(self,
                           W: np.ndarray,
                           dWs: np.ndarray)->None:
        muscl.compute_prediction(W,
                                 dWs,
                                 self.dm.dtM,
                                 self.vels,
                                 self.ndim,
                                 self.gamma,
                                 self._d_,
                                 self._p_,
                                 self.WB,
                                 self.isothermal)

    def solve_riemann_problem(self,
                              dim: str,
                              F: np.ndarray,
                              prims: bool)->None:
        """
        args: 
            dim:    dimension name
            F:      Solution vector with Fluxes
            prims:  Wheter values at faces are primitives
                    or conservatives
        overwrites:
            F: Fluxes given by the Riemann solver
        """
        idim=self.dims2[dim]
        vels = np.roll(self.vels,-idim)
        if self.WB:
            #Move to solution at interfaces
            M_eq_faces = self.dm.__getattribute__(f"M_eq_faces_{dim}")
            self.MR_faces[dim][...] += M_eq_faces
            self.ML_faces[dim][...] += M_eq_faces
        F[...] = self.riemann_solver_fv(self.ML_faces[dim],
                                        self.MR_faces[dim],
                                        vels,
                                        self._p_,
                                        self.gamma,
                                        self.min_c2,
                                        prims,
                                        isothermal=self.isothermal)
        if self.WB:
            #We compute the perturbation over the flux for conservative variables
            F -= self.dm.__getattribute__(f"F_eq_faces_{dim}")
    
    def compute_fv_fluxes(self,dt: float)->None:
        #Clean array with ghost cells
        self.dm.M_fv[...]  = 0
        #Copy W_cv to active region of M_fv
        self.fill_active_region(self.dm.W_cv)
        for dim in self.dims2:
            self.fv_Boundaries(self.dm.M_fv,dim)
        return muscl.compute_second_order_fluxes(self,self.F_faces_FB,dt)

    def fill_active_region(self, M):
        ngh=self.Nghc
        self.dm.M_fv[(Ellipsis,)+tuple(repeat(slice(ngh,-ngh),self.ndim))] = M

    def fv_apply_fluxes(self,dt):
        dUdt = self.dm.U_cv.copy()*0
        for dim in self.dims2:
            ndim = self.ndim
            ngh = self.ngh[dim]
            shift=self.dims2[dim]
            dx = self.faces[dim][ngh+1:-ngh] - self.faces[dim][ngh:-(ngh+1)]
            dx = dx[(None,)*(ndim-shift)+(slice(None),)+(None,)*(shift)]
            dUdt += (self.F_faces[dim][cut(1,None,shift)]
                             -self.F_faces[dim][cut(None,-1,shift)])/dx
            
        if self.potential:
            self.apply_potential(dUdt,
                                 self.dm.U_cv,
                                 self.dm.grad_phi_fv)

        self.dm.U_new[...] = self.dm.U_cv - dUdt*dt

    def fv_update(self):
        self.dm.U_new[...] = self.dm.U_cv
        self.compute_fv_fluxes(self.dm.dt)
        self.fv_apply_fluxes(self.dm.dt)
        self.dm.U_cv[...] = self.dm.U_new

    def init_fv_Boundaries(self, M) -> None:
        ngh=self.Nghc
        n = self.p+1
        if n>2:
            M = M[crop_fv(n-ngh,-(n-ngh),0,self.ndim,n-ngh)]
        for dim in self.dims2:
            idim = self.dims2[dim]
            BC_fv = self.dm.__getattribute__(f"BC_fv_{dim}")
            BC_fv[0][...] = M[cut(None, ngh,idim)]
            BC_fv[1][...] = M[cut(-ngh,None,idim)]

    def fv_store_BC(self,
             M: np.ndarray,
             dim: str) -> None:
        """
        Stores the solution of ngh layers in the active region
        """    
        na=np.newaxis
        idim = self.dims2[dim]
        ngh=self.Nghc
        BC = self.BC[dim]
        cuts=(cut(-2*ngh,  -ngh,idim),
              cut(   ngh, 2*ngh,idim))
        for side in [0,1]:
            if  BC[side] == "periodic":
                self.BC_fv[dim][side] = M[cuts[side]]
            elif BC[side] == "reflective":
                self.BC_fv[dim][side] = M[cuts[1-side]]
                self.BC_fv[dim][side][self.vels[idim]] *= -1
            elif BC[side] == "gradfree":
                self.BC_fv[dim][side] = M[cuts[1-side]]
            elif BC[side] == "ic":
                next
            elif BC[side] == "pressure":
                next
            elif BC[side] == "eq":
                self.BC_fv[dim][side][...] = 0
            else:
                raise("Undetermined boundary type")
                         
    def fv_apply_BC(self,
                 dim: str) -> None:
        """
        Fills ghost cells in M_fv
        """
        ngh=self.Nghc
        idim = self.dims2[dim]
        self.dm.M_fv[cut(None, ngh,idim)] = self.BC_fv[dim][0]
        self.dm.M_fv[cut(-ngh,None,idim)] = self.BC_fv[dim][1]

    def fv_Boundaries(self,
                    M: np.ndarray,
                    dim: str):
        self.fv_store_BC(M,dim)
        self.Comms_fv(M,dim)
        self.fv_apply_BC(dim)
    
    def Comms_fv(self,
             M: np.ndarray,
             dim: str):
        comms = self.comms
        rank = comms.rank
        rank_dim = comms.__getattribute__(dim)    
        idim = self.dims2[dim]
        ngh=self.Nghc
        cuts=(cut(-2*ngh,  -ngh,idim),
              cut(   ngh, 2*ngh,idim))
        Buffers={}
        for side in [0,1]:
            Buffer = M[cuts[1-side]]
            Buffer = self.dm.xp.asnumpy(Buffer).flatten()
            Buffers[side] = Buffer
        
        neighbour = comms.left[idim] if rank%2 else comms.right[idim]
        side = rank_dim%2
        self.send_recv_fv(neighbour,Buffers[side],dim,side)

        neighbour = comms.right[idim] if rank%2 else comms.left[idim]
        side = 1-rank_dim%2
        self.send_recv_fv(neighbour,Buffers[side],dim,side)

    def send_recv_fv(self, neighbour, Buffer, dim, side):
        comms = self.comms
        rank = comms.rank
        if neighbour != rank:
            comms.send_recv_replace(Buffer,neighbour,side)
            BC = self.BC_fv[dim][side]
            self.BC_fv[dim][side][...] = self.dm.xp.asarray(Buffer).reshape(BC.shape)
      
