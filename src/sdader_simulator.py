import sys
sys.path.append("torlo/")
from torlo.ADER import ADER

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
from amr.transfer import prolongate_block, restrict_blocks_overlap_sp

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
                 refine_fn=None,
                 derefine_fn=None,
                 adapt_interval: int = None,
                 amr_max_level: int = None,
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
        # Dynamic AMR hooks. Can also be set/modified as plain attributes
        # between steps. When ``adapt_interval`` is not None and at least
        # one of refine_fn/derefine_fn is set, ``perform_update`` will tag
        # and adapt after every ``adapt_interval`` steps.
        self.refine_fn = refine_fn
        self.derefine_fn = derefine_fn
        self.adapt_interval = adapt_interval
        self.amr_max_level = amr_max_level

        self.post_init()
        self.compute_dt()

        self.init_ader()
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

    def init_ader(self):
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
        Allocate arrays to be used in trouble detection. Zero-initialize
        to avoid inf/NaN bits leaking from uninitialized memory.
        """
        import numpy as _np
        self.dm.troubles  = _np.zeros_like(self.array_FV(self.p+1,1))
        self.dm.theta     = _np.zeros_like(self.array_FV(self.p+1,1,ngh=self.Nghc))
        for dim in self.dims:
            self.dm.__setattr__(f"affected_faces_{dim}",_np.zeros_like(self.array_FV(self.p+1,1,dim=dim)))

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

        # Per-block 1/h metrics also live on dm (GPU-managed); refresh the
        # dict entries so they point at dm's current-location copy.
        for dim in self.dims:
            self._inv_h_block[dim] = self.dm.__getattribute__(f"inv_h_block_{dim}")
            self._inv_h_block_ader[dim] = self.dm.__getattribute__(f"inv_h_block_ader_{dim}")

        if self.update=="FV":
            self.create_dicts_fv()

    def ader_string(self)->str:
        """
        Einsum index string for ADER updates. Covers the Nb (b) axis, the
        ndim cell axes (z,y,x), and the ndim point axes (k,j,i).
        Layout: [nvar, nader, Nb, (Nz,) (Ny,) Nx, (pz,) (py,) px].
        """
        if self.ndim==3:
            return "bzyxkji"
        elif self.ndim==2:
            return "byxji"
        else:
            return "bxi"

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
            if self.viscosity or self.thdiffusion:
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

    def ader_update(self):
        # dUdt = (dFxdx + dFydy + dFzdz + S)dt 
        s = self.ader_string()
        dUdt = (np.einsum(f"t,ut{s}->u{s}",self.dm.w_tp,self.ader_dudt())*self.dt)

        # U_new = U_old - dUdt
        self.dm.U_sp -= dUdt
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        
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
                                       self.equations,
                                       npassive=self.npassive,
                                       thdiffusion=self.thdiffusion,
                                       _t_=self._t_)
            bc.apply_interfaces(self,F,self.F_ader_fp[dim],dim)
            # AMR conservation: at coarse-fine faces, overwrite the coarse
            # block's face flux with the restriction of the fine-side
            # Riemann flux so the two sides agree on the integrated flux.
            if self.forest.max_level > 0:
                bc.correct_coarse_fine_flux(self,self.F_ader_fp[dim],dim)
            if self.WB:
                #F->F'
                self.F_ader_fp[dim]-=self.dm.__getattribute__(f"F_eq_fp_{dim}")[:,na]
    
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
            vels = np.roll(self.vels[:self.ndim],-idim)
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
        # Per-meshblock FV layout: [nvar, Nb, NzB*n+2ngh, NyB*n+2ngh, NxB*n+2ngh(+1 on face-axis)].
        shape = [nvar, self.forest.Nblocks]
        N=[]
        for dim2 in self.dims:
            N.append(self.NB[dim2]*n+(dim==dim2)+2*ngh)
        return np.ndarray(shape+N[::-1])

    def array_FV_BC(self,dim="x")->np.ndarray:
        shape = [2, self.nvar, self.forest.Nblocks]
        ngh=self.Nghc
        N=[]
        for dim2 in self.dims:
            N.append(self.NB[dim2]*self.n[dim2]+2*ngh if dim!=dim2 else ngh)
        return np.ndarray(shape+N[::-1])

    def switch_to_finite_volume(self):
        #Compute control volume averages
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        #Change to Finite Volume scheme
        self.dm.U_cv = self.transpose_to_fv(self.dm.U_cv)
        # W_cv gets recomputed from U_cv in fv_update; avoid an extra transpose.
        self.dm.W_cv = np.zeros_like(self.dm.U_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_fv(self.dm.U_eq_cv)
        if not(self.godunov):
            for dim in self.dims:
                self.F_ader_fp[dim][...] = self.integrate_faces(self.F_ader_fp[dim],dim)

    def switch_to_high_order(self):
        #Change back to High-Order scheme
        self.dm.U_cv = self.transpose_to_sd(self.dm.U_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_sd(self.dm.U_eq_cv)
        #Compute solution at solution points
        self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)
        # Rebuild W_cv directly in SD layout (cheaper than FV->SD transpose).
        self.dm.W_cv = self.compute_primitives(self.dm.U_cv)
        

    def store_high_order_fluxes(self,i_ader):
        ndim=self.ndim
        # Transpose tuples now include the leading (nvar, Nb) batch axes.
        # Original single-batch (nvar only) tuples for 1D/2D/3D:
        #   1D: (0,1,2)          -- [nvar, Nx, px]          -> no change
        #   2D: (0,1,3,2,4)      -- interleave cells + points
        #   3D: (0,1,4,2,5,3,6)
        # With Nb inserted at axis 1, shift everything after 0 by +1:
        dims  = [(0,1,2,3),
                 (0,1,2,4,3,5),
                 (0,1,2,5,3,6,4,7)]
        # `indices2` keeps only Nx-last-element + px-last-pt; result has
        # one fewer cell axis. Old dims2:
        #   1D: (0)               -- [nvar] (scalar)
        #   2D: (0,1,2)           -- [nvar, Ny, py]
        #   3D: (0,1,3,2,4)
        # Shift by +1 for Nb:
        dims2 = [(0,1),
                 (0,1,2,3),
                 (0,1,2,4,3,5)]
        Nb = self.forest.Nblocks
        Nn = [self.NB[dim]*self.n[dim] for dim in self.dims][::-1]
        for dim in self.dims:
            shift=self.dims[dim]
            shape=[self.nvar, Nb]+Nn
            Fd = self.F_ader_fp[dim][:,i_ader]                # [nvar, Nb, ...cells, ...pts]
            self.F_faces[dim][cut(None,-1,shift)] = np.transpose(
                Fd[cut(None,-1,shift)],dims[ndim-1]
                ).reshape(shape)
            shape.pop(ndim-shift+1)                            # +1 for the Nb axis
            self.F_faces[dim][indices(-1,shift)] = np.transpose(
                Fd[indices2(-1,ndim,shift)],dims2[ndim-1]).reshape(shape)
    
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
            dt = self.dt*self.dm.w_tp[i_ader]
            if not(self.godunov):
                self.store_high_order_fluxes(i_ader)
            # Apply the CF flux correction once here too when FB is on: the
            # candidate U_new built from the non-CF-corrected F_faces would
            # bias the DMP check and alter theta. (Non-FB paths are the
            # caller's problem - they handle CF-correction upstream at the
            # SD level.)
            if self.FB:
                if self.forest.max_level > 0:
                    for dim in self.dims:
                        self.correct_coarse_fine_fv_flux(self.F_faces[dim], dim)
                for dim in self.dims:
                    self.symmetrize_same_level_fv_flux(self.F_faces[dim], dim)
            #Compute candidate solution
            self.fv_apply_fluxes(dt)
            if self.FB:
                detect_troubles(self)
                self.compute_fv_fluxes(dt)
                self.correct_fluxes()
                # CF-consistent fluxes for AMR: the (theta*F_FB + (1-theta)*F_HO)
                # blend is generally NOT equal on the two sides of a coarse-fine
                # face because theta varies across it, so it leaks mass across
                # levels. Overwrite the coarse-side face flux with the average
                # of the fine-side fluxes to restore strict conservation.
                if self.forest.max_level > 0:
                    for dim in self.dims:
                        self.correct_coarse_fine_fv_flux(self.F_faces[dim], dim)
                # Also enforce one shared flux value at SAME-level block
                # interfaces, eliminating residual pair mismatch there.
                for dim in self.dims:
                    self.symmetrize_same_level_fv_flux(self.F_faces[dim], dim)
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
        if self.update=="SD":
            self.ader_update()
        else:
            self.fv_update()
        if self.WB:
            #U' -> U
            self.dm.U_sp[...] += self.dm.U_eq_sp
            self.dm.U_cv[...] += self.dm.U_eq_cv
        self.compute_primitives(self.dm.U_cv,W=self.dm.W_cv)
        self.time += self.dt
        # Dynamic AMR: tag + adapt after the step completes so new children
        # prolongate from the just-updated solution, not the pre-step state.
        if (self.adapt_interval is not None
                and self.n_step % self.adapt_interval == 0
                and (self.refine_fn is not None or self.derefine_fn is not None)):
            # perform_update only refreshes W_cv; refresh W_sp too so the
            # tagging predicates see the post-step primitive state.
            self.dm.W_sp[...] = self.compute_primitives(self.dm.U_sp)
            to_r, to_d = self.tag_blocks(
                refine_fn=self.refine_fn,
                derefine_fn=self.derefine_fn,
                max_level=self.amr_max_level,
            )
            if to_r or to_d:
                self.adapt(to_refine=to_r, to_derefine=to_d)
        return True

    def init_sim(self):
        self.checkpoint = False
        self.dm.switch_to(CupyLocation.device)
        self.create_dicts()
        self.execution_time = -timer()

    # ------------------------------------------------------------------ AMR
    def _sd_arrays_realloc(self) -> None:
        """Re-allocate every Nb-sized array after a forest change.

        Primary solution arrays (U_sp/W_sp/U_cv/W_cv) are reallocated; the
        caller is responsible for populating U_sp. ADER flux/RS/BC buffers,
        FV buffers (when update=="FV"), and the per-dim dicts are rebuilt.
        """
        self.dm.W_cv = self.array_sp(ader=False)
        self.dm.W_sp = self.array_sp(ader=False)
        self.dm.U_sp = self.array_sp(ader=False)
        self.dm.U_cv = self.array_sp(ader=False)
        if not self.godunov:
            self.ader_arrays()
            self.init_sd_Boundaries()
        # FV buffers also carry the Nb axis and must track forest changes.
        # fv_arrays / fb_arrays zero-initialize (see their docstrings).
        if self.update == "FV":
            self.fv_arrays()
            if self.FB:
                self.fb_arrays()
            # init_fv_Boundaries needs the (newly allocated) FV-layout W_gh;
            # it is used only during trouble detection's "ic" / "eq" BCs and
            # at pressure-type domain boundaries, so we rebuild it from the
            # current U_cv instead of the long-gone post_init snapshot.
            # Skip the explicit re-init here; fv_Boundaries will refill the
            # ghost slabs on every fv_update iteration from the live M_fv.
        self.create_dicts()
        self._refresh_block_metrics()

    def _snapshot_U_sp(self) -> dict:
        """Snapshot U_sp keyed by (level, logical)."""
        return {
            (b.level, b.logical): self.dm.U_sp[:, ib].copy()
            for ib, b in enumerate(self.forest.blocks)
        }

    def _transfer_from_snapshot(self, snapshot: dict) -> None:
        """Populate self.dm.U_sp for every block in the new forest, using
        direct copy / prolongate / restrict as appropriate. Recomputes the
        derived W_sp / U_cv / W_cv afterwards."""
        ndim = self.ndim
        dim_keys = list(self.dims.keys())
        LM_p = self.dm.LM_prolong
        R_side_sp = self.dm.RS_sp
        for ib, block in enumerate(self.forest.blocks):
            key = (block.level, block.logical)
            if key in snapshot:
                self.dm.U_sp[:, ib] = snapshot[key]
                continue
            # Try prolongation from parent.
            parent_logical = tuple(c // 2 for c in block.logical)
            parent_key = (block.level - 1, parent_logical)
            if parent_key in snapshot:
                parent_U = snapshot[parent_key]
                children_U = prolongate_block(parent_U, LM_p, ndim)
                # child_id = sum_k 2^k * sub_offset[k], k indexing dim_keys order.
                sub_idx = 0
                for k, d in enumerate(dim_keys):
                    offset = block.logical[k] - 2 * parent_logical[k]
                    sub_idx += (1 << k) * offset
                self.dm.U_sp[:, ib] = children_U[:, sub_idx]
                continue
            # Try restriction from children.
            child_keys = []
            for sub_idx in range(2 ** ndim):
                child_logical = tuple(
                    2 * block.logical[k] + ((sub_idx >> k) & 1)
                    for k in range(ndim)
                )
                child_keys.append((block.level + 1, child_logical))
            if all(k in snapshot for k in child_keys):
                stack = np.stack([snapshot[k] for k in child_keys], axis=1)
                self.dm.U_sp[:, ib] = restrict_blocks_overlap_sp(
                    stack, R_side_sp, self.dm.cv_to_sp, ndim
                )
                continue
            raise RuntimeError(
                f"No source data for block {key} (ib={ib}): neither the "
                f"block itself, its parent {parent_key}, nor its full set "
                f"of children was in the snapshot."
            )
        self.dm.W_sp[...] = self.compute_primitives(self.dm.U_sp)
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)

    def tag_blocks(self,
                   refine_fn=None,
                   derefine_fn=None,
                   max_level: int = None):
        """Tag blocks for refinement / derefinement via user predicates.

        Parameters
        ----------
        refine_fn(block, W_block) -> bool : True to refine this block.
                 Blocks already at max_level are skipped.
        derefine_fn(parent_logical, sibling_blocks, sibling_W) -> bool :
                 True to merge this sibling group. Only called for groups
                 where all 2**ndim siblings exist and are at the same level.

        Returns
        -------
        to_refine : list[int]
        to_derefine : list[list[int]]
        """
        to_refine = []
        if refine_fn is not None:
            for ib, block in enumerate(self.forest.blocks):
                if max_level is not None and block.level >= max_level:
                    continue
                if refine_fn(block, self.dm.W_sp[:, ib]):
                    to_refine.append(ib)

        to_derefine = []
        if derefine_fn is not None:
            # Group blocks by parent logical; only full sibling groups qualify.
            groups = {}
            for ib, b in enumerate(self.forest.blocks):
                if b.level == 0:
                    continue
                pl_key = (b.level, tuple(c // 2 for c in b.logical))
                groups.setdefault(pl_key, []).append(ib)
            n_sib = 2 ** self.ndim
            for (lvl, parent_logical), ibs in groups.items():
                if len(ibs) != n_sib:
                    continue
                sibs = [self.forest.blocks[i] for i in ibs]
                sib_W = [self.dm.W_sp[:, i] for i in ibs]
                if derefine_fn(parent_logical, sibs, sib_W):
                    to_derefine.append(ibs)
        return to_refine, to_derefine

    def adapt(self, to_refine=None, to_derefine=None) -> None:
        """Apply refinement and/or derefinement to the forest and transfer
        existing data onto the new block layout.

        Parameters
        ----------
        to_refine : list of int, optional
            Block ibs to refine. Each one becomes 2**ndim children.
        to_derefine : list of list of int, optional
            Each inner list is 2**ndim sibling ibs to merge into one
            coarser parent.

        2:1 balance is enforced automatically (cascading refinement for
        any remaining level gaps > 1). Block data propagates via
        prolongate_block (for newly refined children) and overlap-aware
        SP restriction (for newly coarsened parents).
        """
        to_refine = list(to_refine or [])
        to_derefine = list(to_derefine or [])
        if not to_refine and not to_derefine:
            return
        # Snapshot BEFORE mutating the forest.
        snapshot = self._snapshot_U_sp()
        # Resolve ib references to block objects so the forest-mutation
        # order is independent.
        refine_refs = [self.forest.blocks[i] for i in to_refine]
        derefine_refs = [[self.forest.blocks[i] for i in sibs]
                         for sibs in to_derefine]
        # Apply changes.
        for block in refine_refs:
            ib = self.forest.blocks.index(block)
            self.forest.refine_block(ib)
        for sibs in derefine_refs:
            sib_ibs = [self.forest.blocks.index(b) for b in sibs]
            self.forest.derefine_block(sib_ibs)
        self.forest.enforce_2to1_balance()
        # Reallocate SD arrays with the new Nb, then populate from snapshot.
        self._sd_arrays_realloc()
        self._transfer_from_snapshot(snapshot)
        # dt may shrink because new fine blocks have smaller h.
        self.compute_dt()

    def end_sim(self):
        self.dm.switch_to(CupyLocation.host)
        self.execution_time += timer() 
        self.create_dicts()
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)
        if self.rank==0:
            print(f"t={self.time}, steps taken {self.n_step}, time taken {self.execution_time}")

    def elapsed_time(self):
        return self.execution_time + timer() 

    def cost_per_step(self):
        cost = 0 if self.n_step==0 else self.elapsed_time()/self.n_step
        return cost

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
            if not(self.checkpoint):
                if ((self.available_time-self.elapsed_time())<120) and self.rank==0:
                    self.checkpoint=True
                self.checkpoint = self.comms.reduce_max(self.checkpoint)
                if self.checkpoint:
                    self.output()
                    print("Checkpoint")
                    self.noutput-=1
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
            vels = np.roll(self.vels[:self.ndim],-idim)
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