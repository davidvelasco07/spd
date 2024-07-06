from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from collections import defaultdict

from data_management import CupyLocation
from data_management import GPUDataManager
from initial_conditions_3d import sine_wave
import hydro

class Simulator:
    def __init__(
        self,
        init_fct: Callable = sine_wave,
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Nghe: int = 1,
        Nghc: int = 2,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        ndim: int = 3,
        gamma: float = 1.4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1E-10,
        use_cupy: bool = True,
        BC: Tuple = ("periodic","periodic","periodic"),
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

        self.N = defaultdict(list)
        self.N["x"] = self.Nx
        self.N["y"] = self.Ny
        self.N["z"] = self.Nz

        self.Nghe = Nghe #Number of ghost element layers
        self.Nghc = Nghc #Number of ghost cell layers
        self.ndim = ndim
        self.gamma=gamma
        self.cfl_coeff = cfl_coeff
        self.min_c2 = min_c2

        assert len(BC) >= ndim
        self.BC = defaultdict(list)
        self.dims = defaultdict(list)
        self.dims2 = defaultdict(list)
        dims = ["x","y","z"]
        for dim in range(ndim):
            self.dims[dim] = dims[dim]
            self.dims2[dims[dim]] = dim
            self.BC[dims[dim]] = BC[0]     

        self.dm = GPUDataManager(use_cupy)
        
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0
        
        self.xlen = xlim[1]-xlim[0]
        self.ylen = ylim[1]-ylim[0]
        self.zlen = zlim[1]-zlim[0]

        self.len = defaultdict(list)
        self.len["x"] = self.xlen
        self.len["y"] = self.ylen
        self.len["z"] = self.zlen

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

        self.nghx = Nghc
        self.nghy = (0,Nghc) [self.Y]
        self.nghz = (0,Nghc) [self.Z]

        self.ngh = defaultdict(list)
        self.ngh["x"] = self.nghx
        self.ngh["y"] = self.nghy
        self.ngh["z"] = self.nghz

    def shape(self,idim):
        return (None,)*(self.ndim-idim)+(slice(None),)+(None,)*(idim)
    
    def compute_positions(self):
        pass
    
    def compute_mesh_cv(self) -> np.ndarray:
        pass
        
    def post_init(self) -> None:
        pass

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
    
    def crop(self,M,ngh=1)->np.ndarray:
        return M[(slice(None),)+(slice(ngh,-ngh),)*self.ndim+(Ellipsis,)]

    def compute_primitives(self,U,**kwargs)->np.ndarray:
        return hydro.compute_primitives(
                U,
                self.vels,
                self._p_,
                self.gamma,
                **kwargs)
                
    def compute_conservatives(self,W,**kwargs)->np.ndarray:
        return hydro.compute_conservatives(
                W,
                self.vels,
                self._p_,
                self.gamma,
                **kwargs)
    
    def compute_fluxes(self,F,M,vels,prims)->np.ndarray:
        assert len(vels)==self.ndim
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        hydro.compute_fluxes(W,vels,self._p_,self.gamma,F=F)

    def compute_dt(self) -> None:
        pass
