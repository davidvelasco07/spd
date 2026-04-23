import numpy as np
import os

from simulator import Simulator
from polynomials import gauss_legendre_quadrature
from polynomials import solution_points
from polynomials import flux_points
from polynomials import lagrange_matrix
from polynomials import lagrangeprime_matrix
from polynomials import intfromsol_matrix
from polynomials import quadrature_mean
import hydro
from transforms import compute_A_from_B
from transforms import compute_A_from_B_full

import riemann_solver as rs
from amr.transfer import build_transfer_matrices

class SD_Simulator(Simulator):
    def __init__(
        self,
        riemann_solver_sd: str = "llf",
        update: str = "SD",
        folder: str = "outputs/",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.folder =  folder
        self.noutput = 0
        self.riemann_solver_sd = rs.Riemann_solver(riemann_solver_sd).solver
        self.update = update
        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, self.p)
        sp = solution_points(0.0, 1.0, self.p)
        fp = flux_points(0.0, 1.0, self.p)
        
        for name in ["sp","fp","n"]:
            self.__setattr__(name,{})
        for dim in self.dims:    
            self.__setattr__(f"{dim}_sp",sp)
            self.sp[dim] = self.__getattribute__(f"{dim}_sp")
            self.__setattr__(f"{dim}_fp",fp)
            self.fp[dim] = self.__getattribute__(f"{dim}_fp")
            self.__setattr__(f"n{dim}",self.p+1)
            self.n[dim] = self.__getattribute__(f"n{dim}")

        # Lagrange matrices to perform interpolation between basis
        self.dm.sp_to_fp = lagrange_matrix(self.x_fp, self.x_sp)
        self.dm.fp_to_sp = lagrange_matrix(self.x_sp, self.x_fp)
        # Spatial derivative of the flux at sol pts from density at flux pts.
        self.dm.dfp_to_sp = lagrangeprime_matrix(self.x_sp, self.x_fp)
        # Mean values in control volumes from values at sol pts.
        self.dm.sp_to_cv = intfromsol_matrix(self.x_sp, self.x_fp)
        self.dm.fp_to_cv = intfromsol_matrix(self.x_fp, self.x_fp)
        self.dm.cv_to_sp = np.linalg.inv(self.dm.sp_to_cv)
        # AMR: coarse <-> fine solution-point transfer operators.
        self.dm.LM_prolong, self.dm.LM_restrict = build_transfer_matrices(self.x_sp)

        self.mesh_cv = self.compute_mesh_cv()
        self.compute_positions()
    
    def compute_mesh_cv(self) -> np.ndarray:
        # Rank-global mesh (single block covering the whole per-rank domain).
        # Kept for back-compat and visualization; IC eval uses per-block meshes
        # from `compute_mesh_cv_block`.
        Nghe=self.Nghe
        Ns = [self.N[dim]+2*Nghe for dim in self.dims]
        shape = (self.ndim,)+tuple(Ns[::-1])+(self.p+2,)*self.ndim
        mesh_cv = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h=self.h[dim]
            lenght = self.len[dim]+2*Nghe*h
            shape1 = (None,)*(self.ndim-1-idim)+(slice(None),)+(None,)*(self.ndim+idim)
            shape2 = (None,)*(2*self.ndim-1-idim)+(slice(None),)+(None,)*(idim)
            mesh_cv[idim] = self.lim[dim][0]+(np.arange(N)[shape1]+self.fp[dim][shape2])*lenght/N-h
        return mesh_cv

    def compute_mesh_cv_block(self, block) -> np.ndarray:
        """Per-meshblock control-volume mesh (shape matches compute_mesh_cv but
        sized by NB[dim] and located within the block's physical extent)."""
        Nghe = self.Nghe
        Ns = [self.NB[dim] + 2*Nghe for dim in self.dims]
        shape = (self.ndim,) + tuple(Ns[::-1]) + (self.p+2,)*self.ndim
        mesh_cv = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h = block.h[dim]
            lo, hi = block.lim[dim]
            lenght = (hi - lo) + 2*Nghe*h
            shape1 = (None,)*(self.ndim-1-idim)+(slice(None),)+(None,)*(self.ndim+idim)
            shape2 = (None,)*(2*self.ndim-1-idim)+(slice(None),)+(None,)*(idim)
            mesh_cv[idim] = lo + (np.arange(N)[shape1] + self.fp[dim][shape2])*lenght/N - h
        return mesh_cv

    def compute_mesh(self,Points) -> np.ndarray:
        Nghe=self.Nghe
        Ns = [self.N[dim]+2*Nghe for dim in self.dims]
        shape = (self.ndim,)+tuple(Ns[::-1])
        for points in Points[::-1]:
            shape += (points.size,)
        mesh = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h=self.h[dim]
            lenght = self.len[dim]+2*Nghe*h
            shape1 = (None,)*(self.ndim-1-idim)+(slice(None),)+(None,)*(self.ndim+idim)
            shape2 = (None,)*(2*self.ndim-1-idim)+(slice(None),)+(None,)*(idim)
            mesh[idim] = self.lim[dim][0]+(np.arange(N)[shape1]+Points[idim][shape2])*lenght/N-h
        return mesh
    
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
        
    def post_init(self) -> None:
        na = np.newaxis
        nvar = self.nvar
        ngh = self.Nghe
        # W_gh layout: [nvar, Nb, NzB+2ngh, NyB+2ngh, NxB+2ngh, pz, py, px].
        # Each meshblock's IC values are computed on that block's own physical
        # mesh (so multi-block grids see position-dependent ICs correctly).
        W_gh = self.array_sp(ngh=ngh)
        for ib, block in enumerate(self.forest.blocks):
            mesh_cv_b = self.compute_mesh_cv_block(block)
            for var in range(nvar):
                W_gh[var, ib] = quadrature_mean(mesh_cv_b, self.init_fct, self.ndim, var)

        self.W_init_cv = self.crop_elements(W_gh)
        self.dm.W_cv = self.W_init_cv.copy()
        self.dm.W_sp = self.compute_sp_from_cv(self.dm.W_cv)
        self.dm.U_sp = self.compute_conservatives(self.dm.W_sp)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)

        if self.update=="FV":
            self.W_gh = self.transpose_to_fv(W_gh)
    
    def regular_mesh(self,W):
        #Interpolate to a regular mesh
        p=self.p
        if p<=1:
            return W
        x = np.arange(p+2)/(p+1)
        x = .5*(x[1:]+x[:-1])
        x_sp = solution_points(0.0, 1.0, p)
        m = lagrange_matrix(x, x_sp)
        W_r = compute_A_from_B_full(W,m,self.ndim)
        return W_r
    
    def block_to_fv(self, M_block):
        """Per-block SD layout -> per-block flat FV layout.

        Input  (ndim=3): [..., NzB, NyB, NxB, pz, py, px]
        Output (ndim=3): [..., NzB*pz, NyB*py, NxB*px]
        """
        if self.ndim==1:
            s = M_block.shape
            return M_block.reshape(s[:-2] + (s[-2]*s[-1],))
        elif self.ndim==2:
            s = M_block.shape
            return np.transpose(
                M_block,
                tuple(range(M_block.ndim-4)) + (M_block.ndim-4, M_block.ndim-2,
                                                M_block.ndim-3, M_block.ndim-1)
            ).reshape(s[:-4] + (s[-4]*s[-2], s[-3]*s[-1]))
        else:
            s = M_block.shape
            lead = M_block.ndim - 6
            perm = tuple(range(lead)) + (lead, lead+3,
                                         lead+1, lead+4,
                                         lead+2, lead+5)
            return np.transpose(M_block, perm).reshape(
                s[:-6] + (s[-6]*s[-3], s[-5]*s[-2], s[-4]*s[-1])
            )

    def transpose_to_fv(self,M):
        """[nvar, Nb, NzB, NyB, NxB, pz, py, px] -> [nvar, Nz*pz, Ny*py, Nx*px].

        Multi-block uniform-level: each block is flattened and placed in its
        rank-global slab determined by block.logical. AMR (mixed levels)
        cannot tile a single flat array; use per-block plotting instead.
        """
        Nb = M.shape[1]
        if Nb == 1:
            return self.block_to_fv(M[:,0])
        # Uniform-level assembly.
        levels = {b.level for b in self.forest.blocks}
        assert levels == {0}, (
            "transpose_to_fv expects all blocks at level 0 (mixed-level "
            "AMR: iterate per-block with plot_field instead)")
        nvar = M.shape[0]
        n = self.p + 1
        dim_keys = list(self.dims.keys())            # ['x'] / ['x','y'] / ['x','y','z']
        # Output shape: [nvar, (Nz*n,) (Ny*n,) Nx*n] — reverse for row-major z,y,x.
        out_shape = [nvar] + [self.N[d]*n for d in reversed(dim_keys)]
        out = np.empty(out_shape)
        for ib, block in enumerate(self.forest.blocks):
            M_b_flat = self.block_to_fv(M[:, ib])
            slabs = [slice(None)]                     # nvar
            for d in reversed(dim_keys):
                i_log = block.logical[dim_keys.index(d)]
                w = self.NB[d] * n
                slabs.append(slice(i_log*w, (i_log+1)*w))
            out[tuple(slabs)] = M_b_flat
        return out

    def transpose_to_sd(self, M):
        # [nvar, Nznz, Nyny, Nxnx] -> [nvar, Nb=1, Nz, Ny, Nx, nz, ny, nx].
        shape=[]
        for dim in self.dims:
            shape+=[self.n[dim],self.N[dim]]
        shape=[M.shape[0]]+shape[::-1]
        if self.ndim==1:
            out = M.reshape(shape)
        elif self.ndim==2:
            out = np.transpose(M.reshape(shape),(0, 1,3, 2,4))
        else:
            out = np.transpose(M.reshape(shape),(0, 1,3,5, 2,4,6))
        return out[:,np.newaxis]
    
    def array(self,px,py,pz,ngh=0,ader=False,nvar=None) -> np.ndarray:
        # Layout: [nvar, (nader,) Nb, NzB, NyB, NxB, pz, py, px]. NzB/NyB/NxB
        # are per-meshblock element counts; Nb = number of meshblocks.
        if type(nvar) == type(None):
            nvar = self.nvar
        shape = [nvar,self.nader] if ader else [nvar]
        shape += [self.forest.Nblocks]
        N = []
        for dim in self.dims:
            N.append(self.NB[dim]+2*ngh)
        N = N[::-1]
        p = [px,py,pz][:self.ndim][::-1]
        return np.ndarray(shape+N+p)
        
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
    
    def array_RS(self,dim="x",dim2=None,ader=False)->np.ndarray:
        shape = [self.nvar,self.nader] if ader else [self.nvar]
        shape += [self.forest.Nblocks]
        N = []
        for odim in self.dims:
            N.append(self.NB[odim]+(odim==dim))
        shape += N[::-1]
        if self.ndim>2:
            if (dim2 == "x") or (dim2== "y" and dim=="x"):
                shape += [self.p+2]
            else:
                shape += [self.p+1]
        if self.ndim>1:
            if (dim2 == "z") or (dim2== "y" and dim=="z"):
                shape += [self.p+2]
            else:
                shape += [self.p+1]
            
        return np.ndarray(shape)
    
    def array_BC(self,dim="x",dim2=None,ader=False)->np.ndarray:
        shape = [2,self.nvar,self.nader] if ader else [2,self.nvar]
        shape += [self.forest.Nblocks]
        if self.Z:
            if dim=="x" or dim=="y":
                shape += [self.NB["z"]]
        if self.Y:
            if dim=="x" or dim=="z":
                shape += [self.NB["y"]]
        if dim=="y" or dim=="z":
            shape += [self.NB["x"]]
        if self.Z:
            if dim2=="x" or (dim2=="y" and dim=="x"):
                shape += [self.p+2]
            else:
                shape += [self.p+1]
        if self.Y:
            if dim2=="z" or (dim2=="y" and dim=="z"):
                shape += [self.p+2]
            else:
                shape += [self.p+1]
        return np.ndarray(shape)
    
    def compute_sp_from_cv(self,M_cv)->np.ndarray:
        return compute_A_from_B_full(M_cv,self.dm.cv_to_sp,self.ndim)
        
    def compute_cv_from_sp(self,M_sp)->np.ndarray:
        return compute_A_from_B_full(M_sp,self.dm.sp_to_cv,self.ndim)
    
    def compute_cp_from_sp(self,M_sp)->np.ndarray:
        return compute_A_from_B_full(M_sp,self.dm.sp_to_fp,self.ndim)
    
    def compute_sp_from_cp(self,M_cp)->np.ndarray:
        return compute_A_from_B_full(M_cp,self.dm.fp_to_sp,self.ndim)
    
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
    
    def integrate_faces(self,M_fp,dim,ader=True):
        for other_dim in self.dims:
            if dim != other_dim:
                M_fp = compute_A_from_B(M_fp,self.dm.sp_to_cv,other_dim,self.ndim,ader=ader)
        return M_fp
    
    def compute_gradient(self,M_fp,dim):
        return self.compute_sp_from_dfp(M_fp,dim,ader=True)/self.h[dim]
    
    def compute_dt(self) -> None:
        W = self.dm.W_cv
        c_s = hydro.compute_cs(W[self._p_],W[self._d_],self.gamma,self.min_c2)
        c = c_s*self.ndim
        for vel in self.vels[:self.ndim]:
            c += np.abs(W[vel])
        c_max = np.max(c)
        h = self.h_min/(self.p + 1) 
        dt = h/c_max
        dt = self.comms.reduce_min(dt).item() 
        if self.viscosity and self.nu>0:
            dt = min(dt,h**2/self.nu*.25)
        self.dt = self.cfl_coeff*dt
        

    def Comms_fp(self,
             M: np.ndarray,
             dim: str):
            comms = self.comms
            comms.Comms_sd(self.dm,
                       M,
                       self.BC_fp,
                       self.dims[dim],
                       dim,
                       self.Nghc)
            
    def output(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank==0:
            os.makedirs(folder)
            
        self.comms.barrier()

        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size>1:
            file += f"_{self.comms.rank}"
        np.save(file,self.dm.W_cv)
        self.outputs.append([self.time,self.noutput])
        if self.rank==0:
            np.savetxt(folder+"/outputs.out",self.outputs)
        self.noutput+=1
    
    def load_output(self):
        folder = self.folder
        outputs = np.loadtxt(folder+"/outputs.out")
        rows = int(outputs.size//2)
        self.outputs = list(outputs.reshape([rows,2]))
        self.time,self.noutput = self.outputs[-1]
        self.noutput = int(self.noutput)
        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size>1:
            file += f"_{self.comms.rank}"
        self.dm.W_cv[...] = np.load(file+".npy")
        self.dm.U_cv[...] = self.compute_conservatives(self.dm.W_cv)
        self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)
        self.noutput+=1

    def checkpoint(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank==0:
            os.makedirs(folder)
            
        self.comms.barrier()

        file = f"{folder}/Checkpoint"
        if self.comms.size>1:
            file += f"_{self.comms.rank}"
        np.save(file,self.dm.W_cv)
        self.outputs.append([self.time,self.noutput])
        if self.rank==0:
            np.savetxt(folder+"/outputs.out",self.outputs)
        self.noutput+=1    