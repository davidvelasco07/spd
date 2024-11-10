import numpy as np

from sd_simulator import SD_Simulator
from fv_simulator import FV_Simulator
from data_management import CupyLocation
from polynomials import gauss_legendre_quadrature
from polynomials import ader_matrix
from polynomials import quadrature_mean
import sd_boundary as bc
from trouble_detection import detect_troubles
from timeit import default_timer as timer
from slicing import cut, indices, indices2, crop_fv

class SDADER_Simulator(SD_Simulator,FV_Simulator):
    def __init__(self,
                 FB: bool = False,
                 tolerance: float = 1e-5,
                 SED: bool = True,
                 NAD: str = "",
                 PAD: bool = True,
                 blending: bool = True,
                 min_rho: float = 1e-10,
                 max_rho: float = 1e10,
                 min_P: float = 1e-10,
                 godunov: bool = False,
                 limiting_variables: list = [0],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.FB = FB
        self.tolerance = tolerance
        self.SED = SED
        self.NAD = NAD
        self.PAD = PAD
        self.blending = blending
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.min_P = min_P
        self.godunov = godunov
        self.limiting_variables = limiting_variables

        # ADER matrix.
        self.dm.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.m + 1)
        ader = ader_matrix(self.dm.x_tp, self.dm.w_tp, 1.0)
        self.dm.invader = np.linalg.inv(ader)
        self.dm.invader = np.einsum("p,np->np",self.dm.w_tp,self.dm.invader)
        #number of time slices
        self.nader = self.m+1
        if not(self.godunov):
            self.ader_arrays()
            self.init_sd_Boundaries()
        if self.update=="FV":
            self.fv_arrays()
            if FB:
                self.fb_arrays()
            self.init_fv_Boundaries(self.W_gh)
        if self.potential:
            self.init_potential()
        if self.WB:
            self.init_equilibrium_state()

    def ader_arrays(self):
        """
        Allocate arrays to be used in the ADER time integration
        """
        self.dm.U_ader_sp = self.array_sp(ader=True)
        for dim in self.dims:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"M_ader_fp_{dim}",self.array_fp(dims=dim,ader=True))
            #Conservative fluxes at flux points
            self.dm.__setattr__(f"F_ader_fp_{dim}",self.array_fp(dims=dim,ader=True))
            #Arrays to Solve Riemann problem at the interface between elements
            self.dm.__setattr__(f"ML_fp_{dim}",self.array_RS(dim=dim,ader=True))
            self.dm.__setattr__(f"MR_fp_{dim}",self.array_RS(dim=dim,ader=True))
            #Arrays to communicate boundary values
            self.dm.__setattr__(f"BC_fp_{dim}",self.array_BC(dim=dim,ader=True))

    def fb_arrays(self):
        """
        Allocate arrays to be used in trouble detection
        """
        self.dm.troubles  = self.array_FV(self.p+1,1)
        self.dm.theta = self.array_FV(self.p+1,1,ngh=self.Nghc)
        for dim in self.dims:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"affected_faces_{dim}",self.array_FV(self.p+1,1,dim=dim))

    def create_dicts(self):
        """
        Creates dictonaries for the arrays used in the ADER time
        integration. It enables writting generic functions for all
        dimensions.
        """
        if not(self.godunov):
            names = ["M_ader_fp","F_ader_fp","MR_fp","ML_fp","BC_fp"]
            for name in names:
                self.__setattr__(name,{})
                for dim in self.dims:
                    self.__getattribute__(name)[dim] = self.dm.__getattribute__(f"{name}_{dim}")

        if self.update=="FV":
            self.create_dicts_fv()

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

    def ader_dudt(self):
        dUdt = self.compute_sp_from_dfp_x()
        if self.Y:
            dUdt += self.compute_sp_from_dfp_y()
        if self.Z:
            dUdt += self.compute_sp_from_dfp_z()
        if self.potential:
            self.apply_potential(dUdt,
                                 self.dm.U_ader_sp,
                                 self.dm.grad_phi_sp[:,np.newaxis])
        return dUdt

    def ader_predictor(self,prims: bool = False) -> None:
        na = self.dm.xp.newaxis
        # W -> primivite variables
        # U -> conservative variables
        # Structure of arrays:
        #   U_sp: (nvar,Nz,Ny,Nx,pz,py,px)
        #   U_ader_sp: (nader,nvar,Nz,Ny,Nx,pz,py,px)

        # 1) Initialize u_ader_sp to u_sp, at all ADER time substeps.
        self.dm.U_ader_sp[...] = self.dm.U_sp[:,na, ...]

        # 2) ADER scheme (Picard iteration).
        # nader: number of time slices
        # m+1: order and number of iterations
        for ader_iter in range(self.m + 1):
            if prims:
                # Primitive variables
                if self.WB:
                    raise("SD: Well-balanced scheme is not enabled for interpolation over primitive variables")
                M = self.compute_primitives(self.dm.U_ader_sp)   
            else:
                # Otherwise conservative variables
                M = self.dm.U_ader_sp
            # Once M hosts the correct set of variables,
            # we can interpolate to faces, and solve    
            self.solve_faces(M,ader_iter,prims=prims)

            if ader_iter < self.m:
                # 2c) Compute new iteration value.
                # Axes labels:
                #   u: conservative variables
                #   n: ADER substeps, next
                #   p: ADER substeps, prev

                #Let's store dUdt first
                s = self.ader_string()
                self.dm.U_ader_sp[...] = np.einsum(f"np,up{s}->un{s}",self.dm.invader,
                                                    self.ader_dudt())*self.dm.dt
                #Update
                # U_new = U_old - dUdt
                self.dm.U_ader_sp[...] = self.dm.U_sp[:,na] - self.dm.U_ader_sp

    def ader_update(self):
        # dUdt = (dFxdx + dFydy + dFzdz + S)dt 
        s = self.ader_string()
        dUdt = (np.einsum(f"t,ut{s}->u{s}",self.dm.w_tp,self.ader_dudt())*self.dm.dt)

        # U_new = U_old - dUdt
        self.dm.U_sp -= dUdt

        # Compute primitive variables at solution points from updated solution    
        #self.dm.W_sp =  self.compute_primitives(self.dm.U_sp)

    def solve_faces(self, M, ader_iter, prims=False)->None:
        na=np.newaxis
        # Interpolate M(U or W) to flux points
        # Then compute fluxes at flux points
        for key in self.idims:
            dim = self.idims[key]
            vels = np.roll(self.vels,-key)
            self.M_ader_fp[dim][...] = self.compute_fp_from_sp(M,dim,ader=True)
            if self.WB:
                #U'->U
                self.M_ader_fp[dim]+=self.dm.__getattribute__(f"M_eq_fp_{dim}")[:,na]
            self.compute_fluxes(self.F_ader_fp[dim], self.M_ader_fp[dim],vels,prims)
            bc.Boundaries_sd(self,self.M_ader_fp[dim],dim)
            F = self.riemann_solver_sd(self.ML_fp[dim],
                                       self.MR_fp[dim],
                                       self.MR_fp[dim],
                                       vels,
                                       self._p_,
                                       self.gamma,
                                       self.min_c2,
                                       prims,
                                       isothermal=self.isothermal,
                                       npassive=self.npassive)
            bc.apply_interfaces(self,F,self.F_ader_fp[dim],dim)
            if self.WB:
                #F->F'
                self.F_ader_fp[dim]-=self.dm.__getattribute__(f"F_eq_fp_{dim}")[:,na]
        
        if self.viscosity or self.thdiffusion:
            self.add_nabla_terms()

    def compute_gradient(self,M_fp,dim):
        return self.compute_sp_from_dfp(M_fp,dim,ader=True)/self.h[dim]
    
    def add_nabla_terms(self,):
        """
        This routine adds terms involving second order derivatives in space,
        such as viscosity and thermal diffusion.
        """
        dW_sp = {}
        for dim in self.dims:
            idim = self.dims[dim]
            #Compute gradient of primitive variables at flux points
            self.M_ader_fp[dim][...] = self.compute_primitives(self.M_ader_fp[dim])
            bc.Boundaries_sd(self,self.M_ader_fp[dim],dim)
            #Make a choice of values (here left)
            M = self.ML_fp[dim]
            bc.apply_interfaces(self,M,self.M_ader_fp[dim],dim)
            dW_sp[idim] = self.compute_gradient(self.M_ader_fp[dim],dim)
        dW_fp = {}
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels,-idim)
            idims = self.idims if self.viscosity else [idim]
            for idim in idims:
                #Interpolate gradients(all directions) to flux points at dim
                dW_fp[idim] = self.compute_fp_from_sp(dW_sp[idim],dim,ader=True)
                bc.Boundaries_sd(self,dW_fp[idim],dim)
                #Counter the previous choice of values (now right)
                dW = self.MR_fp[dim]
                bc.apply_interfaces(self,dW,dW_fp[idim],dim)
            #Add viscous flux
            self.F_ader_fp[dim][...] -= self.compute_viscous_fluxes(self.M_ader_fp[dim],dW_fp,vels,prims=True)
            if self.thdiffusion:
                #Add thermal flux
                self.F_ader_fp[dim][self._p_] -= self.compute_thermal_fluxes(self.M_ader_fp[dim],dW_fp[self.dims[dim]],prims=True)

    ####################
    ## Finite volume
    ####################
    def array_FV(self,n,nvar,dim=None,ngh=0)->np.ndarray:
        shape = [nvar] 
        N=[]
        for dim2 in self.dims:
            N.append(self.N[dim2]*n+(dim==dim2)+2*ngh)
        return np.ndarray(shape+N[::-1])
    
    def array_FV_BC(self,dim="x")->np.ndarray:
        shape = [2,self.nvar]
        ngh=self.Nghc
        N=[]
        for dim2 in self.dims:
            N.append(self.N[dim2]*self.n[dim2]+2*ngh if dim!=dim2 else ngh)
        return np.ndarray(shape+N[::-1])

    def switch_to_finite_volume(self):
        #Compute control volume averages
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        #Change to Finite Volume scheme
        self.dm.U_cv = self.transpose_to_fv(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_fv(self.dm.W_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_fv(self.dm.U_eq_cv)
        if not(self.godunov):
            for dim in self.dims:
                self.F_ader_fp[dim][...] = self.integrate_faces(self.F_ader_fp[dim],dim)

    def switch_to_high_order(self):
        #Change back to High-Order scheme
        self.dm.U_cv = self.transpose_to_sd(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_sd(self.dm.W_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_sd(self.dm.U_eq_cv)
        #Compute solution at solution points
        self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)
        

    def store_high_order_fluxes(self,i_ader):
        ndim=self.ndim
        dims  = [(0,1,2),(0,1,3,2,4),(0,1,4,2,5,3,6)]
        dims2 = [(0),(0,1,2),(0,1,3,2,4)]
        Nn = [self.N[dim]*self.n[dim] for dim in self.dims][::-1]
        for dim in self.dims:
            shift=self.dims[dim]
            shape=[self.nvar]+Nn
            self.F_faces[dim][cut(None,-1,shift)] = np.transpose(
                self.F_ader_fp[dim][:,i_ader][cut(None,-1,shift)],dims[ndim-1]
                ).reshape(shape)
            shape.pop(ndim-shift)
            self.F_faces[dim][indices(-1,shift)] = np.transpose(
                self.F_ader_fp[dim][:,i_ader][indices2(-1,ndim,shift)],dims2[ndim-1]).reshape(shape)
    
    def correct_fluxes(self):
        for dim in self.dims:
            if self.godunov:
                theta = 1
            else:
                theta = self.dm.__getattribute__(f"affected_faces_{dim}")
            self.F_faces[dim] = theta*self.F_faces_FB[dim] + (1-theta)*self.F_faces[dim]

    def fv_update(self):
        self.switch_to_finite_volume()
        for i_ader in range(self.nader):
            self.dm.W_cv[...] = self.compute_primitives_cv(self.dm.U_cv)
            dt = self.dm.dt*self.dm.w_tp[i_ader]
            if not(self.godunov):
                self.store_high_order_fluxes(i_ader)
            #Compute candidate solution
            self.fv_apply_fluxes(dt)
            if self.FB:
                detect_troubles(self)
                self.compute_fv_fluxes(dt)
                self.correct_fluxes()
                #Compute corrected solution
                self.fv_apply_fluxes(dt)
            #Update solution
            self.dm.U_cv[...] = self.dm.U_new
        self.switch_to_high_order()

    def compute_primitives_cv(self,U)->np.ndarray:
        if self.WB:
            return (self.compute_primitives(U+self.dm.U_eq_cv)
                    -self.compute_primitives(self.dm.U_eq_cv))
        else:
            return  self.compute_primitives(U)

    ####################
    ## Update functions
    ####################
    def perform_update(self) -> bool:
        self.n_step += 1
        if self.WB:
            #U -> U'
            self.dm.U_sp -= self.dm.U_eq_sp
        if not(self.godunov):
            self.ader_predictor()
        elif self.update=="SD":
            self.ader_update()
        else:
            self.fv_update()
        if self.WB:
            #U' -> U
            self.dm.U_sp[...] += self.dm.U_eq_sp
            self.dm.U_cv[...] += self.dm.U_eq_cv
        self.compute_primitives(self.dm.U_cv,W=self.dm.W_cv)
        self.time += self.dt
        return True

    def init_sim(self):
        self.dm.switch_to(CupyLocation.device)
        self.create_dicts()
        self.execution_time = -timer() 

    def end_sim(self):
        self.dm.switch_to(CupyLocation.host)
        self.execution_time += timer() 
        self.create_dicts()
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)
        if self.rank==0:
            print(f"t={self.time}, steps taken {self.n_step}, time taken {self.execution_time}")

    def perform_iterations(self, n_step: int) -> None:
        self.init_sim()
        for i in range(n_step):
            self.compute_dt()
            self.perform_update()
        self.end_sim()
     
    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        self.init_sim()
        while(self.time < t_end):
            if not self.n_step % 100 and self.rank==0 and self.verbose:
                print(f"Time step #{self.n_step} (t = {self.time})",end="\r")
            self.compute_dt()   
            if(self.time + self.dt >= t_end):
                self.dt = t_end-self.time
            if(self.dt < 1E-14):
                print(f"dt={self.dt}")
                break
            self.status = self.perform_update()
        self.end_sim()          

    def init_sd_Boundaries(self) -> None:
        #This is necessary when the BCs are the ICs
        ndim=self.ndim
        for dim in self.dims:
            idim = self.dims[dim]
            BC = self.dm.__getattribute__(f"BC_fp_{dim}")
            M_fp = self.compute_fp_from_sp(self.dm.U_sp,dim)
            BC[0][...] =  M_fp[:,np.newaxis][indices2( 0,ndim,idim)]
            BC[1][...] =  M_fp[:,np.newaxis][indices2(-1,ndim,idim)]

    def init_potential(self) -> None:
        phi_cv = quadrature_mean(self.mesh_cv, self.init_fct, self.ndim, -1)
        phi_sp = self.compute_sp_from_cv(phi_cv[None])
        self.dm.grad_phi_sp = self.array_sp()[:self.ndim]
        for dim in self.dims:
            idim = self.dims[dim]
            phi_fp = self.compute_fp_from_sp(phi_sp,dim)
            self.dm.grad_phi_sp[idim] = self.crop(self.compute_sp_from_dfp(phi_fp, dim))/self.h[dim]
            # Now for the finite volume update
        self.dm.grad_phi_fv = self.transpose_to_fv(self.compute_cv_from_sp(self.dm.grad_phi_sp))

    def init_equilibrium_state(self) -> None:
        crop = lambda start,end,idim,ngh : crop_fv(start,end,idim,self.ndim,ngh)
        p = self.p
        n = p+1
        nvar = self.nvar
        ngh = self.Nghe
        W_gh = self.array_sp(ngh=ngh)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(self.mesh_cv, self.eq_fct, self.ndim, var)
        
        W_sp = self.compute_sp_from_cv(W_gh)
        U_sp = self.compute_conservatives(W_sp)
        self.dm.U_eq_sp = self.crop(U_sp)
        self.dm.U_eq_cv = self.compute_cv_from_sp(self.dm.U_eq_sp)
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels,-idim)
            U = self.compute_fp_from_sp(U_sp,dim)
            self.dm.__setattr__(f"M_eq_fp_{dim}",self.crop(U))
            M_fp = self.dm.__getattribute__(f"M_eq_fp_{dim}")
            #We force the equilibrium values at flux points to match between elements
            M_fp[cut( 1, None, idim+self.ndim)][indices(0,idim)] = M_fp[cut(None, -1, idim+self.ndim)][indices(-1,idim)]
            F = U.copy()
            W = self.compute_primitives(U)
            self.compute_fluxes(F,W,vels,prims=True)
            self.dm.__setattr__(f"F_eq_fp_{dim}",self.crop(F))
            
            if self.update=="FV":
                W_faces = self.integrate_faces(W,dim,ader=False)[cut(None,-1,idim)]
                W_faces = self.transpose_to_fv(W_faces)
                W_faces = W_faces[crop(p+1,-p,idim,p+1)]
                self.dm.__setattr__(f"M_eq_faces_{dim}",W_faces)
                F=W_faces.copy()
                self.compute_fluxes(F,W_faces,vels,prims=True)
                self.dm.__setattr__(f"F_eq_faces_{dim}",F)
        ngh = self.Nghc
        if self.update=="FV":
            if n>ngh:
                self.dm.M_eq_fv = self.transpose_to_fv(W_gh)[crop(n-ngh,-(n-ngh),0,n-ngh)]
            else:
                self.dm.M_eq_fv = self.transpose_to_fv(W_gh)