import sys
import os
sys.path.append("torlo/")
sys.path.append("induction/")
from torlo.ADER import ADER

import numpy as np

from induction.sdader_induction_simulator import SDADER_Induction_Simulator
from sdader_simulator import SDADER_Simulator
from sd_simulator import SD_Simulator
from simulator import Simulator
from data_management import CupyLocation
from polynomials import gauss_legendre_quadrature
from polynomials import ader_matrix
from polynomials import quadrature_mean
import sd_boundary as bc
from trouble_detection import detect_troubles
from timeit import default_timer as timer
from slicing import cut, indices, indices2, crop_fv
import mhd
import riemann_solver as rs

class SDADER_MHD_Simulator(SDADER_Simulator,SDADER_Induction_Simulator):
    def __init__(self,
                 equations=mhd,
                 riemann_solver_sd = "llf_mhd",
                 *args,
                 **kwargs):
        
        SD_Simulator.__init__(self,*args,**kwargs)
        self.riemann_solver_sd = rs.Riemann_solver(riemann_solver_sd).solver
        self.equations=equations
        self.godunov=False
        self.b={}
        for dim in "xyz":
            name = f"$B_{dim}$"
            self.variables.append(name)
            self.__setattr__(f"_b{dim}_",self.nvar)
            self.b[dim] = self.nvar
            self.nvar += 1
        self.variables.append(r"$B^2$")
        self.variables.append(r"$S$")
        self.post_init()
        self.init_ader()
        self.ader_arrays()
        self.init_sd_Boundaries()
        self.init_E_Boundaries_sd()

    def post_init(self) -> None:
        SDADER_Simulator.post_init(self)
        SDADER_Induction_Simulator.post_init(self)
        self.B_to_U()
        self.dm.W_sp = self.compute_primitives(self.dm.U_sp)
        self.dm.W_cv = self.compute_cv_from_sp(self.dm.W_sp)

    def B_to_U(self):
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            self.dm.U_sp[self.b[dim]] = self.compute_sp_from_fp(B[np.newaxis],dim=dim)[0]

    def ader_arrays(self):
        SDADER_Simulator.ader_arrays(self)
        SDADER_Induction_Simulator.ader_arrays(self)

    def ader_dt(self,dim):
        self.ader_dudt(self,dim)
        self.ader_dBdt(self,dim)

    def ader_predictor(self,prims: bool = False) -> None:
        na = self.dm.xp.newaxis

        # 1) Initialize u_ader_sp to u_sp, at all ADER time substeps.
        self.dm.U_ader_sp[...] = self.dm.U_sp[:,na, ...]
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            self.B_ader_fp[dim][...] = B[na]
            self.dm.U_ader_sp[self.b[dim]][...] = self.compute_sp_from_fp(B[na],dim=dim)
        # 2) ADER scheme (Picard iteration).
        # nader: number of time slices
        # m+1: order and number of iterations
        for ader_iter in range(self.m + 1):
            self.solve_faces(self.dm.U_ader_sp,ader_iter,prims=prims)
            self.solve_edges(ader_iter)
            if self.viscosity:
                self.add_nabla_terms()
            if ader_iter < self.m:
                # 2c) Compute new iteration value.
                # Axes labels:
                #   u: conservative variables
                #   n: ADER substeps, next
                #   p: ADER substeps, prev

                #Let's store dUdt first
                s = self.ader_string()

                self.dm.U_ader_sp[...] = np.einsum(f"np,up{s}->un{s}",self.dm.invader,
                                                    self.ader_dudt())*self.dt
                #Update
                # U_new = U_old - dUdt
                self.dm.U_ader_sp[...] = self.dm.U_sp[:,na] - self.dm.U_ader_sp

                for dim in self.dims:
                    self.B_ader_fp[dim] = np.einsum(f"np,p{s}->n{s}",self.dm.invader,
                                                    self.ader_dBdt(dim))

                    self.B_ader_fp[dim][...] = self.B_fp[dim][na] - self.B_ader_fp[dim]

    def create_dicts(self):
        SDADER_Simulator.create_dicts(self) 
        SDADER_Induction_Simulator.create_dicts(self)

    def ader_update(self):
        SDADER_Simulator.ader_update(self)
        SDADER_Induction_Simulator.ader_update(self)
        self.B_to_U()
 
    def add_nabla_terms(self):
        SDADER_Simulator.add_nabla_terms(self)
        SDADER_Induction_Simulator.add_nabla_terms(self)

    def perform_update(self) -> bool:
        self.n_step += 1
        self.ader_predictor()
        self.ader_update()
        self.compute_primitives(self.dm.U_cv,W=self.dm.W_cv)
        self.time += self.dt
        return True
    
    def compute_dt(self) -> None:
        W = self.dm.W_cv
        c_max = 0
        for dim in self.dims:
            dim1,dim2 = self.other_dims(dim)
            c_max += self.equations.compute_fast_vel(W[self._p_],
                                                     W[self._d_],
                                                     W[self.b[dim]],
                                                     W[self.b[dim1]] if dim1 in self.dims else 0,
                                                     W[self.b[dim2]] if dim2 in self.dims else 0,
                                                     self.gamma,self.min_c2)
        
        c_max = np.max(c_max)
        h = self.h_min/(self.p + 1) 
        dt = h/c_max
        dt = self.comms.reduce_min(dt).item() 
        if self.viscosity and self.nu>0:
            nu = max(self.nu,self.chi)
            dt_nu=(0.25*self.h_min/(self.p+1))**2/nu
            dt = min(dt,dt_nu)
        self.dt = self.cfl_coeff*dt
    
    def output(self):
        return SDADER_Simulator.output(self)
    
    def compute_vels(self,dim,dim1,dim2,ader=False):
        v1 = self.vels[self.dims[dim1]]
        v2 = self.vels[self.dims[dim2]]
        if ader:
            W = self.compute_primitives(self.compute_fp_from_sp(self.M_ader_fp[dim1],dim=dim2,ader=ader))
        else:   
            W = self.compute_fp_from_sp(self.dm.W_sp,dim=dim1)
            W = self.compute_fp_from_sp(           W,dim=dim2)
        
        return W[v1], W[v2], W[self.b[dim]], W[0], W[self._p_]
    
    def fill_E_array(self,E_ep,B1,B2,dim,ader=False):
        dim1,dim2 = self.other_dims(dim)
        v1,v2,B3,rho,p = self.compute_vels(dim,dim1,dim2,ader=ader)
        E_ep[0] = v1*B2 - v2*B1
        E_ep[1] = B1
        E_ep[2] = B2
        E_ep[3] = v1
        E_ep[4] = v2
        E_ep[5] = B3
        E_ep[6] = rho
        E_ep[7] = p

    def E_riemann_solver(self,EL,ER,_v1_):
        return self.llf_E(EL,ER,_v1_,gamma=self.gamma,min_c2=self.min_c2)
    
    def llf_E(
        self,
        E_L,
        E_R,
        vel,
        *args,
        **kwargs,
    ) -> None:
        B1_L = E_L[1]
        B1_R = E_R[1]
        B2_L = E_L[2]
        B2_R = E_R[2]
        B3_L = E_L[5]
        B3_R = E_R[5]
        c_L = mhd.compute_fast_vel(E_L[7],E_L[6], B1_L, B2_L, B3_L, **kwargs)
        c_R = mhd.compute_fast_vel(E_R[7],E_R[6], B1_R, B2_R, B3_R, **kwargs)
        Ss = np.maximum(abs(E_R[vel]), abs(E_L[vel])) + np.maximum(c_R, c_L)

        #E = v X B 
        #Ez = vx*By - vy*Bx 
        #Ey = vz*Bx - vx*Bz
        #Ex = vy*Bz - vz*By
        
        if vel == 3:
            #Here B1_T=B1_B, so it reduces to:
            Es = 0.5*(E_R+E_L)-0.5*Ss*(B2_R-B2_L)
        elif vel == 4:
            #Here B1_R=B1_L, so it reduces to:
            Es = 0.5*(E_R+E_L)+0.5*Ss*(B2_R-B2_L)
        return Es
