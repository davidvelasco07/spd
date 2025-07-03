import sys
import os
sys.path.append("torlo/")
from torlo.ADER import ADER

import numpy as np

from sd_simulator import SD_Simulator
from data_management import CupyLocation
from polynomials import gauss_legendre_quadrature
from polynomials import ader_matrix
from polynomials import quadrature_mean
import sd_boundary as bc
from trouble_detection import detect_troubles
from timeit import default_timer as timer
from slicing import cut, indices, indices2, crop_fv

class SDADER_Induction_Simulator(SD_Simulator):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.variables=[]
        for dim in self.dims:
            name = f"$B_{dim}$"
            self.variables.append(name)
        self.variables.append(r"$B^2$")
        self.variables.append(r"$S$")
        self.nvar = 5
        
        # ADER matrix.
        self.dm.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.m + 1)
        ader = ader_matrix(self.dm.x_tp, self.dm.w_tp, 1.0)
        self.dm.invader = np.linalg.inv(ader)
        self.dm.invader = np.einsum("p,np->np",self.dm.w_tp,self.dm.invader)
        #number of time slices
        self.nader = self.m+1
        ## replacing ADER matrices
        self.ader  = ADER(-1,self.m+1,'gaussLegendre')
        self.nader = self.ader.M_sub+1
        self.dm.invader = self.ader.evolMat
        self.dm.w_tp = self.ader.bADER.flatten()
        self.post_init()
        self.compute_dt()
        self.ader_arrays()
        self.init_E_Boundaries_sd()

    def post_init(self) -> None:
        na=np.newaxis
        nvar = self.ndim
        ngh = self.Nghe
        #Initialization of magnetic field
        A_ep = {}
        
        for dim,idim in zip(self.Edims,self.Eidims):
            x = self.x_sp if dim=="x" else self.x_fp
            dims=[x]
            if self.ndim>1:
                y = self.y_sp if dim=="y" else self.y_fp
                dims.append(y)
            if self.ndim>2:
                z = self.z_sp if dim=="z" else self.z_fp
                dims.append(z)
            mesh = self.compute_mesh(dims)
            A = self.vectorpot_fct(mesh,idim)
            A_ep[dim] = self.crop(A[na])[0]
        # A(z^f,y^f,x^f) -> Bz(z^f,y^s,x^s), By(z^s,y^f,x^s), Bx(z^s,y^s,x^f)
        #Bx = dyAz - dzAy
        #By = dzAx - dxAz
        #Bz = dxAy - dyAx
        for dim in self.dims:
            dim1,dim2 = self.other_dims(dim)
            B_fp = self.array_fp(dims=dim)[0]
            B_fp[...] = 0
            if dim1 in self.dims:
                B_fp += self.compute_sp_from_dfp(A_ep[dim2][na],dim1)[0]/self.h[dim1]    
            if dim2 in self.dims:
                B_fp -= self.compute_sp_from_dfp(A_ep[dim1][na],dim2)[0]/self.h[dim2]
            #B_fp = self.crop(B_fp)
            self.__setattr__(f"B{dim}_init_fp",B_fp)
            self.dm.__setattr__(f"B{dim}_fp",B_fp.copy())

        self.A_ep = A_ep
       
        # This arrays contain Nghe layers of ghost elements
        W_gh = self.array_sp(ngh=ngh,nvar=nvar)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(self.mesh_cv, self.init_fct, self.ndim, var)

        W_cv = self.crop(W_gh)
        W_sp = self.compute_sp_from_cv(W_cv) 
        self.dm.W_cp = self.compute_cp_from_sp(W_sp) 

    def compute_dt(self):
        W = self.dm.W_cp
        vel = W[0].copy()
        for idim in range(1,self.ndim):
            vel += np.abs(W[idim])
        c_max = np.max(vel)
        h = self.h_min/(self.p + 1) 
        dt = h/c_max 
        if self.nu>0:
            dt_nu=(0.25*self.h_min/(self.p+1))**2/self.nu
            dt = min(dt,dt_nu)
        dt = self.comms.reduce_min(dt)
        self.dt = self.cfl_coeff*dt.item()

    def compute_positions(self):
        na = np.newaxis
        ngh=self.Nghc
        self.faces = {}
        self.centers = {}
        self.h_fp = {}
        self.h_cv = {}
        for dim in self.dims:
            idim = self.dims[dim]
            #Solution points
            sp = self.lim[dim][0] + (np.arange(self.N[dim])[:,na] + self.sp[dim][na,:])*self.h[dim]
            self.dm.__setattr__(f"{dim.upper()}_sp",sp.reshape(self.N[dim],self.n[dim]))
            #Flux points
            fp = np.ndarray((self.N[dim] * self.n[dim] + ngh*2+1))
            fp[ngh :-ngh] = (self.h[dim]*np.hstack((np.arange(self.N[dim]).repeat(self.n[dim]) + 
             np.tile(self.fp[dim][:-1],self.N[dim]),self.N[dim])))
            fp[ :ngh] = -fp[(ngh+1):(2*ngh+1)][::-1]
            fp[-ngh:] =  fp[-(ngh+1)] + fp[ngh+1:2*ngh+1]
            self.dm.__setattr__(f"{dim.upper()}_fp",fp)
            self.faces[dim] = fp
            #Cell centers 
            cv = 0.5*(fp[1:]+fp[:-1])
            self.dm.__setattr__(f"{dim.upper()}_cv",cv)
            self.centers[dim] = cv
            #Distance between faces
            h_fp = (fp[1:]-fp[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_fp",h_fp)
            self.h_fp[dim] = h_fp
            #Distance between centers
            h_cv = (cv[1:]-cv[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_cv",h_cv)
            self.h_cv[dim] = h_cv

    def ader_arrays(self):
        """
        Allocate arrays to be used in the ADER time integration
        """
        # ep -> edge points
        # Ez -> (z^s,y^f,x^f)
        # Ey -> (z^f,y^s,x^f)
        # Ex -> (z^f,y^f,x^s)
        dims=["yz","zx","xy"]
        for dim,idim in zip(self.Edims,self.Eidims):
            E_ep = self.array_fp(dims=dims[idim],ader=True)
            self.dm.__setattr__(f"E{dim}_ader_ep",E_ep)
            for dim2 in self.other_dims(dim):
                #Arrays to Solve Riemann problem at the interface between elements
                self.dm.__setattr__(f"E{dim}L_ep_{dim2}",self.array_RS(dim=dim2,dim2=dim,ader=True))
                self.dm.__setattr__(f"E{dim}R_ep_{dim2}",self.array_RS(dim=dim2,dim2=dim,ader=True))
                #Arrays to communicate boundary values
                self.dm.__setattr__(f"BC_E{dim}_ep_{dim2}",self.array_BC(dim=dim2,dim2=dim,ader=True))
        
        for dim in self.dims:
            self.dm.__setattr__(f"B{dim}_ader_fp",self.array_fp(dims=dim,ader=True)[0])

    def create_dicts(self):
        """
        Creates dictonaries for the arrays used in the ADER time
        integration. It enables writting generic functions for all
        dimensions.
        """
        names = ["E_ader_ep","B_ader_fp","B_fp"]
        for name in names:
            self.__setattr__(name,{})
        for dim in self.dims:
            self.__getattribute__("B_ader_fp")[dim] = self.dm.__getattribute__(f"B{dim}_ader_fp")
            self.__getattribute__("B_fp")[dim] = self.dm.__getattribute__(f"B{dim}_fp")
        for dim in self.Edims:
            self.__getattribute__("E_ader_ep")[dim] = self.dm.__getattribute__(f"E{dim}_ader_ep")

        names = ["R_ep","L_ep"]
        for name in names:
            E = {}
            self.__setattr__("E"+name,E)
            for dim in self.Edims:
                E[dim] = {}
                for dim2 in self.other_dims(dim):
                    E[dim][dim2] = self.dm.__getattribute__(f"E{dim}{name}_{dim2}")
        
        BC = {}
        self.BC_E_ep = BC
        for dim in self.Edims:
            BC[dim] = {}
            for dim2 in self.other_dims(dim):
                BC[dim][dim2] = self.dm.__getattribute__(f"BC_E{dim}_ep_{dim2}")

    def other_dims(self,dim):
        dims = ["yz","zx","xy"]
        if dim in self.dims:
            idim = self.dims[dim]
        else:
            idim = 2
        dim1 = dims[idim][0]
        dim2 = dims[idim][1]
        return dim1,dim2
    
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

    def ader_dBdt(self,dim):
        #dBxdt = dEydz - dEzdy
        #dBydt = dEzdx - dExdz
        #dBzdt = dExdy - dEydx
        dim1,dim2 = self.other_dims(dim)
        dBdt  = self.compute_sp_from_dfp(self.E_ader_ep[dim1],dim2,ader=True)[0]/self.h[dim2] if dim1 in self.Edims else 0
        dBdt -= self.compute_sp_from_dfp(self.E_ader_ep[dim2],dim1,ader=True)[0]/self.h[dim1] if dim2 in self.Edims else 0
        return dBdt*self.dt

    def ader_predictor(self,prims: bool = False) -> None:
        na = self.dm.xp.newaxis

        # 1) Initialize u_ader_sp to u_sp, at all ADER time substeps.
        for dim in self.dims:
            self.B_ader_fp[dim][...] = self.dm.__getattribute__(f"B{dim}_fp")[na]

        # 2) ADER scheme (Picard iteration).
        # nader: number of time slices
        # m+1: order and number of iterations
        for ader_iter in range(self.m + 1):
            self.solve_edges(ader_iter)
            if self.nu>0:
                self.add_nabla_terms()
            if ader_iter < self.m:
                # 2c) Compute new iteration value.
                # Axes labels:
                #   u: conservative variables
                #   n: ADER substeps, next
                #   p: ADER substeps, prev

                #Let's store dUdt first
                s = self.ader_string()
                for dim in self.dims:
                    self.B_ader_fp[dim] = np.einsum(f"np,p{s}->n{s}",self.dm.invader,
                                                    self.ader_dBdt(dim))
                    #Update
                    # U_new = U_old - dUdt
                    self.B_ader_fp[dim][...] = self.B_fp[dim][na] - self.B_ader_fp[dim]

    def ader_update(self):
        s = self.ader_string()
        for dim in self.dims:
            dBdt = np.einsum(f"t,t{s}->{s}",self.dm.w_tp,self.ader_dBdt(dim))
            self.B_fp[dim] -= dBdt

    def solve_edges(self, ader_iter):
        na=np.newaxis
        # Interpolate B to edge points
        for dim in self.Edims:
            dim1,dim2 = self.other_dims(dim)
            B1 = self.compute_fp_from_sp(self.B_ader_fp[dim1][na],dim2,ader=True)[0] if dim1 in self.dims else 0
            B2 = self.compute_fp_from_sp(self.B_ader_fp[dim2][na],dim1,ader=True)[0] if dim2 in self.dims else 0
            
            self.fill_E_array(self.E_ader_ep[dim],B1,B2,dim,ader=True)

            _v1_ = 3     
            for dim1,dim2 in [self.other_dims(dim),self.other_dims(dim)[::-1]]:
                self.E_Boundaries_sd(self.E_ader_ep[dim],dim,dim1) 
                E = self.E_riemann_solver(self.EL_ep[dim][dim1],
                                          self.ER_ep[dim][dim1],_v1_)
                _v1_+=1
                self.apply_edges(E,self.E_ader_ep[dim],dim1)

    def compute_vels(self,dim,dim1,dim2):
        if self.ndim==3:
            v =  self.compute_sp_from_fp(self.dm.W_cp,dim)
        else:
            v = self.dm.W_cp
        v1 = v[self.dims[dim1]]
        v2 = v[self.dims[dim2]]
        return v1,v2
    
    def fill_E_array(self,E_ep,B1,B2,dim,ader=False):
        dim1,dim2 = self.other_dims(dim)
        v1,v2 = self.compute_vels(dim,dim1,dim2)
        E_ep[0] = v1*B2 - v2*B1
        E_ep[1] = B1
        E_ep[2] = B2
        if ader:
            E_ep[3] = v1[np.newaxis]
            E_ep[4] = v2[np.newaxis]
        else:  
            E_ep[3] = v1
            E_ep[4] = v2
            
    def E_riemann_solver(self,EL,ER,_v1_):
        v = np.where(np.abs(EL[_v1_]) > np.abs(ER[_v1_]), EL[_v1_], ER[_v1_])
        return np.where(v >= 0, EL, ER)

    def compute_gradient(self,M_fp,dim):
        return self.compute_sp_from_dfp(M_fp,dim,ader=True)/self.h[dim]
    
    def add_nabla_terms(self):
        """
        This routine adds terms involving second order derivatives in space,
        such as viscosity and thermal diffusion.
        """
        na=np.newaxis
        #E = nu*(d1B2-d2B1)
        for dim in self.Edims:
            dB_ep = {}
            dims = self.other_dims(dim)
            for dim1,dim2 in [dims,dims[::-1]]:
                #Move from face to edge points
                B = self.compute_fp_from_sp(self.B_ader_fp[dim1][na],dim=dim2,ader=True)
                self.E_ader_ep[dim][1] = B

                #2D Riemann solver
                for d in [dim1,dim2]:
                    self.E_Boundaries_sd(self.E_ader_ep[dim],dim,d) 
                    #Make a choice of values (here left)
                    self.apply_edges(self.EL_ep[dim][d][1][na],B,d)
                
                #Compute gradient at face points
                dB_fp = self.compute_gradient(B,dim2)
                #Move back to edge points
                dB_ep[dim1] = self.compute_fp_from_sp(dB_fp,dim=dim2,ader=True)
                self.E_ader_ep[dim][1] = dB_ep[dim1]
                
                #2D Riemann solver
                for d in [dim1,dim2]:
                    self.E_Boundaries_sd(self.E_ader_ep[dim],dim,d) 
                    #Make complementary choice of values (here right)
                    self.apply_edges(self.ER_ep[dim][d][1][na],dB_ep[dim1],d)

            self.E_ader_ep[dim][0] -= self.nu*(dB_ep[dim1][0]-dB_ep[dim2][0])
            

    ####################
    ## Update functions
    ####################
    def perform_update(self) -> bool:
        self.n_step += 1
        self.ader_predictor()
        self.ader_update()
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
        #self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        #self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)
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
                break   
            self.status = self.perform_update()
        self.end_sim()          

    def init_E_Boundaries_sd(self) -> None:
        #This is necessary when the BCs are the ICs
        for dim,idim in zip(self.Edims,self.Eidims):
            dim1,dim2 = self.other_dims(dim)
            na = np.newaxis
            B1 = self.dm.__getattribute__(f"B{dim1}_fp") 
            B1 = self.compute_fp_from_sp(B1[na],dim2)[0]
            B2 = self.dm.__getattribute__(f"B{dim2}_fp") 
            B2 = self.compute_fp_from_sp(B2[na],dim1)[0]
            E_ep = self.array_fp(dims=dim1+dim2) 
            print(dim,dim1,dim2,E_ep.shape) 
            self.fill_E_array(E_ep,B1,B2,dim)
            for dim2 in self.other_dims(dim):
                BC = self.dm.__getattribute__(f"BC_E{dim}_ep_{dim2}")
                BC[0][...] =  self.E_cut(E_ep[:,np.newaxis], 0,dim2,dim2)
                BC[1][...] =  self.E_cut(E_ep[:,np.newaxis],-1,dim2,dim2)
    
    def E_cut(self,E,index,dim1,dim2):
        return E[indices( index,self.ndim+self.dims[dim2])][indices( index,self.dims[dim1])]

    def store_edges(self: SD_Simulator,
                     E: np.ndarray,
                     dim: str,
                     dim1: str,
                     ) -> None:
        """
        Stores the values of flux points at the extremes of elements(0,-1)
        These arrays are then used to solve the Riemann problem
        """
        EL = self.EL_ep[dim]
        ER = self.ER_ep[dim]
        #print(dim,dim1,dim2, E.shape, self.ER_ep[dim][dim1].shape, self.ER_ep[dim][dim2].shape)
        
        #Nz,Ny,Nx,p2,p1
        shift=self.dims[dim1]+(self.ndim-1)
        #print(dim, dim1, self.dims[dim1], E.shape, E[indices( 0,self.dims[dim1])].shape)
        ER[dim1][cut(None,-1,shift)] = E[indices( 0,self.dims[dim1])]
        EL[dim1][cut(1 ,None,shift)] = E[indices(-1,self.dims[dim1])]

    def apply_edges(self,
                    E: np.ndarray,
                    E_ep: np.ndarray,
                    dim1: str):
        """
        Applies the values of flux points at the extremes of elements(0,-1)
        This is done after the Riemann problem at element interfaces has been
        solved. 
        """
        shift=self.ndim+self.dims[dim1]-1
        #print(E.shape,shift)
        E_ep[indices( 0,self.dims[dim1])] = E[cut(None,-1,shift)]
        E_ep[indices(-1,self.dims[dim1])] = E[cut(1, None,shift)]

    def store_BC(self,
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
                         
    def apply_BC(self, dim: str, dim1: str) -> None:
        """
        Fills up the missing first column of M_L
        and the missing last column of M_R
        """
        
        shift=self.ndim+self.dims[dim1]-1
        #print(dim,dim1,dim2)
        self.EL_ep[dim][dim1][indices( 0,shift)] = self.BC_E_ep[dim][dim1][0]
        self.ER_ep[dim][dim1][indices(-1,shift)] = self.BC_E_ep[dim][dim1][1]

    def E_Boundaries_sd(self,
                        M: np.ndarray,
                        dim: str,
                        dim1: str,
                        ):
        self.store_BC(self.BC_E_ep[dim][dim1],M,dim1)
        self.store_edges(M,dim,dim1)
        self.Comms_ep(M,dim,dim1)
        self.apply_BC(dim,dim1)
            
    def Comms_ep(self,
             M: np.ndarray,
             dim: str,
             dim1:str):
            comms = self.comms
            comms.Comms_sd(self.dm,
                       M,
                       self.BC_E_ep[dim],
                       self.dims[dim1],
                       dim1,
                       self.Nghc)
            
    def compute_B2(self):
        Bx = self.compute_sp_from_fp(self.dm.Bx_fp[np.newaxis],"x")[0]
        By = self.compute_sp_from_fp(self.dm.By_fp[np.newaxis],"y")[0] if self.ndim>1 else 0
        Bz = self.compute_sp_from_fp(self.dm.Bz_fp[np.newaxis],"z")[0] if self.ndim>2 else 0
        B2 = (Bx**2+By**2+Bz**2)
        B2 = self.compute_cv_from_sp(B2[np.newaxis])
        return B2

    def output(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank==0:
            os.makedirs(folder)
            
        self.comms.barrier()

        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size>1:
            file += f"_{self.comms.rank}"
        B2 = self.compute_B2()
        np.save(file,B2)
        self.outputs.append([self.time,self.noutput])
        if self.rank==0:
            np.savetxt(folder+"/outputs.out",self.outputs)
        self.noutput+=1