from typing import Callable,Tuple
import numpy as np

from data_management import GPUDataManager
from comms import CommHelper
from initial_conditions_3d import sine_wave
import hydro

class Simulator:
    def __init__(
        self,
        init_fct: Callable = sine_wave,
        eq_fct: Callable = sine_wave,
        p: int =  1, 
        m: int = -1,
        N: Tuple = (32,32),
        Nghe: int = 1,
        Nghc: int = 2,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        gamma: float = 1.4,
        beta: float = 2./3,
        nu: float = 1e-4,
        chi: float = 1e-4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1E-10,
        isothermal: bool = False,
        viscosity: bool = False,
        thdiffusion: bool = False,
        potential: bool = False,
        passives: list = [],
        WB: bool = False,
        use_cupy: bool = True,
        BC: Tuple = (("periodic","periodic"),
                     ("periodic","periodic"),
                     ("periodic","periodic")),
        verbose = True,
        available_time = 3600.0,
    ):
        self.init_fct = init_fct
        self.eq_fct = eq_fct
        if m==-1:
            #By default m=p
            m=p
        self.p = p #Space order
        self.m = m #Time  order
        ndim = len(N)
        self.ndim = ndim
        assert len(BC) >= ndim
        self.BC = {}
        self.idims = {}
        self.dims = {}
        
        dims = ["x","y","z"]
        for idim in range(ndim):
            dim = dims[idim]
            self.idims[idim] = dim
            self.dims[dim] = idim
            self.BC[dim] = BC[idim]
            self.__setattr__(f"N{dim}",N[idim])
            
        self.Y = ndim>1
        self.Z = ndim>2
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0

        self.Nghe = Nghe #Number of ghost element layers
        self.Nghc = Nghc #Number of ghost cell layers
        self.ndim = ndim
        self.gamma=gamma
        self.beta=beta
        self.nu=nu
        self.chi=chi
        self.cfl_coeff = cfl_coeff
        self.min_c2 = min_c2
        self.isothermal = isothermal
        self.viscosity = viscosity
        self.thdiffusion = thdiffusion
        self.potential = potential
        self.WB = WB
        self.verbose = verbose
        self.comms = CommHelper(self.ndim)
        self.use_cupy = use_cupy 
        self.dm = GPUDataManager(use_cupy)
        self.outputs = []
        self.available_time = available_time

        self.nghx = Nghc
        self.nghy = (0,Nghc) [self.Y]
        self.nghz = (0,Nghc) [self.Z]
        
        self.lim = {}
        self.len = {}
        self.h = {}
        self.N = {}
        self.ngh = {}
        self.h_min = 1E10
        for dim in self.dims:
            self.lim[dim] = self.__getattribute__(f"{dim}lim") 
            self.__setattr__(f"{dim}len",self.lim[dim][1]-self.lim[dim][0])
            self.len[dim] = self.__getattribute__(f"{dim}len")
            self.N[dim] =  self.__getattribute__(f"N{dim}")
            self.__setattr__(f"d{dim}", self.len[dim]/self.N[dim])
            self.h[dim] = self.__getattribute__(f"d{dim}") 
            self.ngh[dim] = self.__getattribute__(f"ngh{dim}") 
            self.h_min = min(self.h_min,self.h[dim])
        self.n_step = 0
        
        self.variables = [r"$\rho$"]
        self._d_ = 0
        self.vels = np.arange(ndim)+1
        for dim in self.dims:
            idim = self.dims[dim]
            name = f"v{dim}"
            self.variables.append(name)
            self.__setattr__(f"_{name}_",self.vels[idim])
        self.variables.append("P")
        self._p_  = self.ndim+1
        self.nvar = self.ndim+2
        self.npassive = len(passives)
        for i in range(self.npassive):
            self.variables.append(passives[i])
        self.nvar += self.npassive
        if self.thdiffusion:
            self._t_ = self.nvar
            self.nvar += 1
            self.variables.append("T")
        else:
            self._t_ = None

        for dim in self.dims:
            n=self.comms.__getattribute__(f"n{dim}")
            x=self.comms.__getattribute__(f"{dim}")
            self.N[dim] = int(self.N[dim]//n)
            self.len[dim] = self.len[dim]/n
            start,end = self.lim[dim]
            start += x*self.len[dim]
            end = start+self.len[dim]
            self.lim[dim] = (start,end)
            #print(self.comms.rank,dim,self.N[dim],self.len[dim],self.lim[dim])
        self.rank = self.comms.rank

    def shape(self,idim):
        return (None,)*(self.ndim-idim)+(slice(None),)+(None,)*(idim)
    
    def compute_positions(self):
        pass
    
    def compute_mesh_cv(self) -> np.ndarray:
        pass
        
    def post_init(self) -> None:
        pass

    def domain_size(self):
        return [ self.N[dim]*self.n[dim] for dim in self.dims]

    def regular_faces(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [np.linspace(lim[dim][0],lim[dim][1],N[dim]*n[dim]+1) for dim in self.dims]

    def regular_centers(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [np.linspace(lim[dim][0],lim[dim][1],N[dim]*n[dim]) for dim in self.dims]

    def crop(self,M,ngh=1)->np.ndarray:
        return M[(slice(None),)+(slice(ngh,-ngh),)*self.ndim+(Ellipsis,)]

    def compute_primitives(self,U,**kwargs)->np.ndarray:
        return hydro.compute_primitives(
                U,
                self.vels,
                self._p_,
                self.gamma,
                isothermal=self.isothermal,
                thdiffusion=self.thdiffusion,
                _t_=self._t_,
                npassive=self.npassive,
                **kwargs)
                
    def compute_conservatives(self,W,**kwargs)->np.ndarray:
        return hydro.compute_conservatives(
                W,
                self.vels,
                self._p_,
                self.gamma,
                isothermal=self.isothermal,
                thdiffusion=self.thdiffusion,
                _t_=self._t_,
                npassive=self.npassive,
                **kwargs)
    
    def compute_fluxes(self,F,M,vels,prims)->np.ndarray:
        assert len(vels)==self.ndim
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        hydro.compute_fluxes(W,
                             vels,
                             self._p_,
                             self.gamma,
                             F=F,
                             isothermal=self.isothermal,
                             npassive=self.npassive)

    def compute_viscous_fluxes(self,M,dMs,vels,prims=False,**kwargs)->np.ndarray:
        assert len(vels)==self.ndim
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        return hydro.compute_viscous_fluxes(W,
                                            vels,
                                            dMs,
                                            self._p_,
                                            self.nu,
                                            self.beta,
                                            npassive=self.npassive,
                                            **kwargs)
    
    def compute_thermal_fluxes(self,M,dMs,prims=False)->np.ndarray:
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        return hydro.compute_thermal_fluxes(W,dMs,self.chi,self._t_)

    def apply_potential(self,dUdt,U,grad_phi):
        _p_ = self._p_
        for idim in self.idims:
            vel = self.vels[idim]
            dUdt[vel,...] += U[  0]*grad_phi[idim]
            if not(self.isothermal):
                dUdt[_p_,...] += U[vel]*grad_phi[idim]
            
    def compute_dt(self) -> None:
        pass

    def riemann_solver(self,solver):
        return lambda ML,MR,vels,prims : solver(ML,MR,vels,prims,
                                                _p_ = self._p_,
                                                gamma = self.gamma,
                                                min_c2 = self.min_c2,
                                                isothermal=self.isothermal,
                                                npassive=self.npassive)
    

