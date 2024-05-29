from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from collections import defaultdict

from data_management import CupyLocation
from data_management import GPUDataManager
from polynomials import gauss_legendre_quadrature
from polynomials import solution_points
from polynomials import flux_points
from polynomials import lagrange_matrix
from polynomials import lagrangeprime_matrix
from polynomials import intfromsol_matrix
from polynomials import ader_matrix
from polynomials import quadrature_mean
from initial_conditions_3d import sine_wave
import sd_ader
import hydro
from transforms import compute_A_from_B
from transforms import compute_A_from_B_full

import riemann_solver as rs

class SD_Simulator:
    def __init__(
        self,
        init_fct: Callable = sine_wave,
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Nghe: int = 1,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        ndim: int = 3,
        gamma: float = 1.4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1E-10,
        use_cupy: bool = True,
        BC: Tuple = ("periodic","periodic","periodic"),
        riemann_solver_sd: Callable = rs.llf,
    ):
        self.init_fct = init_fct
        if m==-1:
            #By default m=p
            m=p
        self.p = p #Space order
        self.m = m #Time  order
        self.Nx = Nx
        self.Y = ndim>1
        self.Z = ndim>2
        self.Ny = ((1,Ny) [self.Y]) 
        self.Nz = ((1,Nz) [self.Z])

        self.Nghe = Nghe #Number of ghost element layers
        self.ndim = ndim
        self.gamma=gamma
        self.cfl_coeff = cfl_coeff
        self.min_c2 = min_c2
        self.riemann_solver_sd = riemann_solver_sd

        assert len(BC) >= ndim
        self.BC = defaultdict(list)
        self.dims = defaultdict(list) 
        dims = ["x","y","z"]
        for dim in range(ndim):
            self.dims[dim] = dims[dim]
            self.BC[dims[dim]] = BC[0]     

        self.dm = GPUDataManager(use_cupy)
        
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0
        
        self.xlen = xlim[1]-xlim[0]
        self.ylen = ylim[1]-ylim[0]
        self.zlen = zlim[1]-zlim[0]

        self.dx = self.xlen/self.Nx
        self.dy = self.ylen/self.Ny
        self.dz = self.zlen/self.Nz

        self.n_step = 0
        
        nvar=0
        self._d_  = nvar
        nvar+=1
        self._vx_ = nvar
        nvar+=1
        if self.Y: 
            self._vy_ = nvar
            nvar+=1
        else:
            self._vy_ = -1
        if self.Z: 
            self._vz_ = nvar
            nvar+=1
        else:
            self._vz_ = -1
        self._p_  = nvar
        nvar+=1
        assert nvar == 2 + self.ndim
        self.nvar = nvar
        self.vels=np.array([self._vx_,self._vy_,self._vz_])[:self.ndim]

        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, p)

        self.x_sp = solution_points(0.0, 1.0, p)
        self.y_sp = (np.ones(1)/2,self.x_sp) [self.Y]
        self.z_sp = (np.ones(1)/2,self.x_sp) [self.Z]
        self.x_fp = flux_points(0.0, 1.0, p)
        self.y_fp = (np.ones(1)/2,self.x_fp) [self.Y]
        self.z_fp = (np.ones(1)/2,self.x_fp) [self.Z]

        # Lagrange matrices to perform interpolation between basis
        self.dm.sp_to_fp = lagrange_matrix(self.x_fp, self.x_sp)
        self.dm.fp_to_sp = lagrange_matrix(self.x_sp, self.x_fp)
        # Spatial derivative of the flux at sol pts from density at flux pts.
        self.dm.dfp_to_sp = lagrangeprime_matrix(self.x_sp, self.x_fp)
        # Mean values in control volumes from values at sol pts.
        self.dm.sp_to_cv = intfromsol_matrix(self.x_sp, self.x_fp)
        self.dm.fp_to_cv = intfromsol_matrix(self.x_fp, self.x_fp)
        self.dm.cv_to_sp = np.linalg.inv(self.dm.sp_to_cv)
        
        # ADER matrix.
        self.dm.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.m + 1)
        ader = ader_matrix(self.dm.x_tp, self.dm.w_tp, 1.0)
        self.dm.invader = np.linalg.inv(ader)
        self.dm.invader = np.einsum("p,np->np",self.dm.w_tp,self.dm.invader)
        #number of time slices
        self.nader = self.m+1
        
        na =  np.newaxis
        
        self.nx = p+1
        self.ny = (1,p+1) [self.Y]
        self.nz = (1,p+1) [self.Z]
  
        self.mesh_cv = self.compute_mesh_cv()
        
        X_sp = xlim[0]+(np.arange(self.Nx)[:,na] + self.x_sp[na,:])*(self.xlen)/(self.Nx)
        Y_sp = ylim[0]+(np.arange(self.Ny)[:,na] + self.y_sp[na,:])*(self.ylen)/(self.Ny)
        Z_sp = zlim[0]+(np.arange(self.Nz)[:,na] + self.z_sp[na,:])*(self.zlen)/(self.Nz)
        
        self.dm.X_sp = X_sp.reshape(self.Nx,self.nx)
        self.dm.Y_sp = Y_sp.reshape(self.Ny,self.ny)
        self.dm.Z_sp = Z_sp.reshape(self.Nz,self.nz)

        self.post_init()
        hydro.compute_dt(self)
        print(f"dt = {self.dm.dt}")

    def compute_mesh_cv(self) -> np.ndarray:
        na = np.newaxis
        Nx = self.Nx+2*self.Nghe
        Ny = self.Ny+2*self.Nghe*self.Y
        Nz = self.Nz+2*self.Nghe*self.Z
        if self.ndim==1:
            mesh_cv = np.ndarray((1, Nx, self.p+2))
            mesh_cv[0] = self.xlim[0]+(np.arange(Nx)[:,na] + self.x_fp[na,:])*(self.xlen+2*self.Nghe*self.dx)/Nx-self.dx
        elif self.ndim==2:
            mesh_cv = np.ndarray((2,Ny, Nx,self.p+2, self.p+2))
            mesh_cv[0] = self.xlim[0]+(np.arange(Nx)[na,:,na,na] + self.x_fp[na,na,na,:])*(self.xlen+2*self.Nghe*self.dx)/Nx-self.dx
            mesh_cv[1] = self.ylim[0]+(np.arange(Ny)[:,na,na,na] + self.y_fp[na,na,:,na])*(self.ylen+2*self.Nghe*self.dy)/Ny-self.dy
        elif self.ndim==3:
            mesh_cv = np.ndarray((3,Nz, Ny, Nx, self.p+2, self.p+2, self.p+2))
            mesh_cv[0] = self.xlim[0]+(np.arange(Nx)[na,na,:,na,na,na] + self.x_fp[na,na,na,na,na,:])*(self.xlen+2*self.Nghe*self.dx)/Nx-self.dx
            mesh_cv[1] = self.ylim[0]+(np.arange(Ny)[na,:,na,na,na,na] + self.y_fp[na,na,na,na,:,na])*(self.ylen+2*self.Nghe*self.dy)/Ny-self.dy
            mesh_cv[2] = self.zlim[0]+(np.arange(Nz)[:,na,na,na,na,na] + self.z_fp[na,na,na,:,na,na])*(self.zlen+2*self.Nghe*self.dz)/Nz-self.dz
        else:
            raise("Incorrect number of dimensions")
        return mesh_cv
        
    def post_init(self) -> None:
        na = np.newaxis
        nvar = self.nvar
        ngh = self.Nghe
        # This arrays contain Nghe layers of ghost elements
        W_gh = self.array_sp(ngh=ngh)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(self.mesh_cv, self.init_fct, self.ndim, var)

        self.W_init_cv = self.crop(W_gh)
        self.dm.W_cv = self.W_init_cv.copy()
        self.dm.W_sp = self.compute_sp_from_cv(self.dm.W_cv)
        self.dm.U_sp = self.compute_conservatives(self.dm.W_sp)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)
        sd_ader.ader_arrays(self)

    def domain_size(self):
        Nx = self.Nx*(self.nx)
        Ny = self.Ny*(self.ny)
        Nz = self.Ny*(self.nz)
        return Nx,Ny,Nz

    def regular_faces(self):
        Nx,Ny,Nz = self.domain_size()
        x=np.linspace(0,self.xlen,Nx+1)
        y=np.linspace(0,self.ylen,Ny+1)
        z=np.linspace(0,self.zlen,Nz+1)
        return x,y,z

    def regular_centers(self):
        Nx,Ny,Nz = self.domain_size()
        x=np.linspace(0,self.xlen,Nx)
        y=np.linspace(0,self.ylen,Ny)
        z=np.linspace(0,self.zlen,Nz)
        return x,y,z
    
    def regular_mesh(self,W):
        #Interpolate to a regular mesh
        p=self.p
        x = np.arange(p+2)/(p+1)
        x = .5*(x[1:]+x[:-1])
        x_sp = solution_points(0.0, 1.0, p)
        m = lagrange_matrix(x, x_sp)
        W_r = self.compute_A_from_B_full(W,m)
        return self.transpose_to_fv(W_r)
    
    def transpose_to_fv(self,M):
        #nvar,Nz,Ny,Nx,nz,ny,nx
        #nvar,Nznz,Nyny,Nxnx
        if self.ndim==1:
            assert M.ndim == 3
            return M.reshape(M.shape[0],M.shape[1]*M.shape[2])   
        elif self.ndim==2:
            assert M.ndim == 5
            return np.transpose(M,(0,1,3,2,4)).reshape(M.shape[0],M.shape[1]*M.shape[3],M.shape[2]*M.shape[4])  
        else:
            assert M.ndim == 7
            return np.transpose(M,(0,1,4,2,5,3,6)).reshape(M.shape[0],M.shape[1]*M.shape[4],M.shape[2]*M.shape[5],M.shape[3]*M.shape[6])   

    def array1d(self,px,ngh=0,ader=False) -> np.ndarray:
        shape = [self.nvar,self.nader] if ader else [self.nvar]
        shape += [self.Nx+2*ngh,px]
        return np.ndarray(shape)
        
    def array2d(self,px,py,ngh=0,ader=False)-> np.ndarray:
        shape = [self.nvar,self.nader] if ader else [self.nvar]
        shape += [self.Ny+2*ngh,self.Nx+2*ngh,py,px]
        return np.ndarray(shape)

    def array3d(self,px,py,pz,ngh=0,ader=False)-> np.ndarray:
        shape = [self.nvar,self.nader] if ader else [self.nvar]
        shape += [self.Nz+2*ngh,self.Ny+2*ngh,self.Nx+2*ngh,
                  pz,py,px]
        return np.ndarray(shape)
    
    def array(self,px,py,pz,**kwargs) -> np.ndarray:
        if self.ndim==1:
            return self.array1d(px,**kwargs)
        if self.ndim==2:
            return self.array2d(px,py,**kwargs)
        if self.ndim==3:
            return self.array3d(px,py,pz,**kwargs)
        
    def array_sp(self,**kwargs):
        p=self.p
        return self.array(
            p+1,
            p+1,
            p+1,
            **kwargs)

    def array_fp(self,dims="xyz",**kwargs):
        p=self.p
        return self.array(
            (p+1+("x" in dims)),
            (p+1+("y" in dims)),
            (p+1+("z" in dims)),
            **kwargs)
    
    def array_RS(self,dim="x",ader=False)->np.ndarray:
        shape = [self.nvar,self.nader] if ader else [self.nvar]
        if self.Z:
            shape += [self.Nz+(dim=="z")]
        if self.Y:
            shape += [self.Ny+(dim=="y")]
        shape += [self.Nx+(dim=="x")]
        if self.Z:    
            shape += [self.p+1]
        if self.Y:
            shape += [self.p+1]
        return np.ndarray(shape)
    
    def array_BC(self,dim="x",ader=False)->np.ndarray:
        shape = [2,self.nvar,self.nader] if ader else [2,self.nvar]
        if self.Z:
            if dim=="x" or dim=="y":
                shape += [self.Nz]
        if self.Y:
            if dim=="x" or dim=="z":
                shape += [self.Ny]
        if dim=="y" or dim=="z":
            shape += [self.Nx]

        if self.Z:    
            shape += [self.p+1]
        if self.Y:
            shape += [self.p+1]
        return np.ndarray(shape)

    def crop_1d(self,M)->np.ndarray:
        ngh = self.Nghe
        return M[:,ngh:-ngh,...]
    
    def crop_2d(self,M)->np.ndarray:
        ngh = self.Nghe
        return M[:,ngh:-ngh,ngh:-ngh,...]
    
    def crop_3d(self,M)->np.ndarray:
        ngh = self.Nghe
        return M[:,ngh:-ngh,ngh:-ngh,ngh:-ngh,...]

    def crop(self,M)->np.ndarray:
        if self.ndim==3:
            return self.crop_3d(M)
        if self.ndim==2:
            return self.crop_2d(M)
        if self.ndim==1:
            return self.crop_1d(M)
    
    def compute_sp_from_cv(self,M_cv)->np.ndarray:
        return compute_A_from_B_full(M_cv,self.dm.cv_to_sp,self.ndim)
        
    def compute_cv_from_sp(self,M_sp)->np.ndarray:
        return compute_A_from_B_full(M_sp,self.dm.sp_to_cv,self.ndim)
    
    def compute_sp_from_fp(self,M_fp,dim,**kwargs) -> np.ndarray:
        return compute_A_from_B(M_fp,self.dm.fp_to_sp,dim,self.ndim,**kwargs)
    
    def compute_fp_from_sp(self,M_sp,dim,**kwargs) -> np.ndarray:
        return compute_A_from_B(M_sp,self.dm.sp_to_fp,dim,self.ndim,**kwargs)
    
    def compute_sp_from_dfp(self,M_fp,dim,**kwargs) -> np.ndarray:
        return compute_A_from_B(M_fp,self.dm.dfp_to_sp,dim,self.ndim,**kwargs)
    
    def compute_sp_from_dfp_x(self,ader=True):
        return self.compute_sp_from_dfp(self.dm.F_ader_fp_x,"x",ader=ader)/self.dx
        
    def compute_sp_from_dfp_y(self,ader=True):
        return self.compute_sp_from_dfp(self.dm.F_ader_fp_y,"y",ader=ader)/self.dy
        
    def compute_sp_from_dfp_z(self,ader=True):
        return self.compute_sp_from_dfp(self.dm.F_ader_fp_z,"z",ader=ader)/self.dz
    
    def compute_primitives(self,U)->np.ndarray:
        W = U.copy()
        W[self._vx_] = U[self._vx_]/U[0]
        K = W[self._vx_]**2
        if self.Y:
            W[self._vy_] = U[self._vy_]/U[0]
            K += W[self._vy_]**2
        if self.Z:
            W[self._vz_] = U[self._vz_]/U[0]
            K += W[self._vz_]**2
        K  *= 0.5*U[0]
        W[self._p_] = (self.gamma-1)*(U[self._p_]-K)
        return W
                
    def compute_conservatives(self,W)->np.ndarray:
        U = W.copy()
        U[self._vx_] = W[self._vx_]*U[0]
        K = W[self._vx_]**2
        if self.Y:
            U[self._vy_] = W[self._vy_]*U[0]
            K += W[self._vy_]**2
        if self.Z:
            U[self._vz_] = W[self._vz_]*U[0]
            K += W[self._vz_]**2
        K  *= 0.5*U[0]
        U[self._p_] = W[self._p_]/(self.gamma-1)+K
        return U
    
    def compute_fluxes(self,F,M,vels,prims)->np.ndarray:
        assert len(vels)==self.ndim
        _p_ = self._p_
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        v_1 = vels[0]
        m_1 = W[0]*W[v_1]
        K = m_1*W[v_1]
        if self.ndim>=2:
            for v in vels[1:]:
                m = W[0]*W[v]
                K += m*W[v]
                F[v,...] = m*W[v_1]
        E = W[_p_]/(self.gamma-1) + 0.5*K
        F[0  ,...] = m_1
        F[v_1,...] = m_1*W[v_1] + W[_p_]
        F[_p_,...] = W[v_1]*(E + W[_p_])

    def perform_update(self) -> bool:
        self.n_step += 1
        na = self.dm.xp.newaxis
        
        sd_ader.ader_predictor(self)
        sd_ader.ader_update(self)
       
        self.time += self.dm.dt
        return True
    
    def perform_iterations(self, n_step: int) -> None:
        self.dm.switch_to(CupyLocation.device)
        sd_ader.create_dicts(self)
        for i in range(n_step):
            hydro.compute_dt(self)
            self.perform_update()
        self.dm.switch_to(CupyLocation.host)
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)
     
    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        self.dm.switch_to(CupyLocation.device)
        sd_ader.create_dicts(self)
        while(self.time < t_end):
            if not self.n_step % 100:
                print(f"Time step #{self.n_step} (t = {self.time})",end="\r")
            hydro.compute_dt(self)   
            if(self.time + self.dm.dt >= t_end):
                self.dm.dt = t_end-self.time
            if(self.dm.dt < 1E-14):
                print(f"dt={self.dm.dt}")
                break
            self.status = self.perform_update()
        self.dm.switch_to(CupyLocation.host)
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)
                    

        