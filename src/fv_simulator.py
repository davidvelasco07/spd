import numpy as np
from itertools import repeat
from collections import defaultdict
from simulator import Simulator
import riemann_solver as rs
import muscl

from slicing import cut, crop_fv

class FV_Simulator(Simulator):
    def __init__(
        self,
        riemann_solver_fv: str = "llf",
        slope_limiter: str = "minmod",
        predictor: bool = True,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.riemann_solver_fv = rs.Riemann_solver(riemann_solver_fv).solver
        self.predictor = predictor
        self.slope_limiter = muscl.Slope_limiter(slope_limiter)
        self.fv_scheme = muscl.MUSCL_Hancock_fluxes if predictor else muscl.MUSCL_fluxes

    def array_FV(self,nvar,dim=None,ngh=0)->np.ndarray:
        shape = [nvar] 
        N=[]
        for dim2 in self.dims:
            N.append(self.N[dim2]+(dim==dim2)+2*ngh)
        return np.ndarray(shape+N[::-1])
    
    def array_FV_BC(self,dim="x")->np.ndarray:
        shape = [2,self.nvar]
        ngh=self.Nghc
        N=[]
        for dim2 in self.dims:
            N.append(self.N[dim2]+2*ngh if dim!=dim2 else ngh)
        return np.ndarray(shape+N[::-1])
    
    def fv_arrays(self)->None:
        self.dm.M_fv  = self.array_FV(self.p+1,self.nvar,ngh=self.Nghc)
        self.dm.U_new = self.array_FV(self.p+1,self.nvar)
        if self.predictor:
            self.dm.dtM = self.array_FV(self.p+1,self.nvar,ngh=self.Nghc-1)
        for dim in self.dims:
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

        for dim in self.dims:
            # faces / centers / h_fp / h_cv were set per-block by
            # sd_simulator.compute_positions; do not overwrite them with the
            # rank-global X_fp / X_cv (which are kept for viz).
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
                           self.h_cv[self.idims[idim]],
                           self.h_fp[self.idims[idim]],
                           idim)
    
    def compute_gradients_fv(self,
                       M: np.ndarray,
                       idim: int,
                       )->np.ndarray: 
        return self.slope_limiter.compute_gradients(M,
                             self.h_cv[self.idims[idim]],
                             self.h_fp[self.idims[idim]],
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
        return M[crop(2,-1,idim) ] - S[crop( 1,None,idim)]

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
            ML:     Values interpolated to the left
        """
        #ML = M + SlopeC*h/2
        ngh=self.Nghc
        crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
        return M[crop(1,-2,idim)] + S[crop(None,-1,idim)]

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
                                 self.npassive)

    def solve_riemann_problem_fv(self,
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
        idim=self.dims[dim]
        vels = np.roll(self.vels,-idim)
        if self.WB:
            #Move to solution at interfaces
            M_eq_faces = self.dm.__getattribute__(f"M_eq_faces_{dim}")
            self.MR_faces[dim][...] += M_eq_faces
            self.ML_faces[dim][...] += M_eq_faces
        F[...] = self.riemann_solver_fv(self.ML_faces[dim],
                                        self.MR_faces[dim],
                                        F,
                                        vels,
                                        self._p_,
                                        self.gamma,
                                        self.min_c2,
                                        prims,
                                        self.equations,
                                        npassive=self.npassive)
        if self.WB:
            #We compute the perturbation over the flux for conservative variables
            F -= self.dm.__getattribute__(f"F_eq_faces_{dim}")
    
    def compute_fv_fluxes(self,dt: float)->None:
        #Clean array with ghost cells
        self.dm.M_fv[...]  = 0
        #Copy W_cv to active region of M_fv
        self.fill_active_region(self.dm.W_cv)
        #Fill ghost zones
        self.fv_Boundaries(self.dm.M_fv)
        #Compute fluxes
        self.fv_fluxes(self.F_faces_FB,dt)

    def fill_active_region(self, M):
        ngh=self.Nghc
        self.dm.M_fv[(Ellipsis,)+tuple(repeat(slice(ngh,-ngh),self.ndim))] = M

    def fv_fluxes(self,
                  F: dict,
                  dt: float,
                  **kwargs)->None:
        self.fv_scheme(self,F,dt,**kwargs)
        if self.viscosity or self.thdiffusion:
            muscl.compute_nabla_terms(self,F)

    def fv_apply_fluxes(self,dt):
        dUdt = self.dm.U_cv.copy()*0
        for dim in self.dims:
            ngh = self.ngh[dim]
            shift=self.dims[dim]
            h = self.h_fp[dim][cut(ngh,-ngh,shift)] 
            dUdt += (self.F_faces[dim][cut(1,None,shift)]
                             -self.F_faces[dim][cut(None,-1,shift)])/h
            
        if self.potential:
            self.apply_potential(dUdt,
                                 self.dm.U_cv,
                                 self.dm.grad_phi_fv)

        self.dm.U_new[...] = self.dm.U_cv - dUdt*dt

    def fv_update(self):
        self.dm.U_new[...] = self.dm.U_cv
        self.compute_fv_fluxes(self.dt)
        self.fv_apply_fluxes(self.dt)
        self.dm.U_cv[...] = self.dm.U_new

    def init_fv_Boundaries(self, M) -> None:
        ngh=self.Nghc
        n = self.p+1
        if n>2:
            M = M[crop_fv(n-ngh,-(n-ngh),0,self.ndim,n-ngh)]
        for dim in self.dims:
            idim = self.dims[dim]
            BC_fv = self.dm.__getattribute__(f"BC_fv_{dim}")
            BC_fv[0][...] = M[cut(None, ngh,idim)]
            BC_fv[1][...] = M[cut(-ngh,None,idim)]

    def fv_store_BC(self,
             M: np.ndarray,
             dim: str,
             all: bool = True) -> None:
        """Populate BC_fv[dim][side, ..., ib, ...] per block via the forest
        neighbor table. Same dispatch convention as store_BC for SD:
        SAME neighbors (including periodic self-wrap) copy a slab of the
        neighbor block; BC-tagged sides apply reflective / gradfree / ic /
        pressure / eq semantics.
        """
        from amr.tree import SAME, FINER, COARSER, BC as BC_TAG
        na = np.newaxis
        idim = self.dims[dim]
        ngh = self.Nghc
        BC = self.BC[dim]
        # For side=0 (left ghost slab of ib): pull the RIGHT-interior slab
        # of the neighbor block -> M[..., jb, ..., -2*ngh:-ngh, ...].
        # For side=1 (right ghost slab of ib): pull the LEFT-interior slab
        # of the neighbor block -> M[..., jb, ..., ngh:2*ngh, ...].
        interior_cuts = (cut(-2*ngh, -ngh, idim),
                         cut(ngh, 2*ngh, idim))
        # `M` has layout [nvar, Nb, ...cells with ghosts...]; pick a block
        # by indexing position 1 (after nvar). BC_fv has an extra leading
        # 'side' axis already consumed via [side], so the block axis is
        # at position 1 there too after the side slice.
        def _view(arr, ib, n):
            return arr[(slice(None),)*n + (ib,)]

        for ib, block in enumerate(self.forest.blocks):
            for side in (0, 1):
                entries = block.neighbors[dim][side]
                bc_slot = _view(self.BC_fv[dim][side], ib, 1)
                if len(entries) == 1 and entries[0][1] == SAME:
                    jb, _rel, _sub = entries[0]
                    src = _view(M, jb, 1)
                    bc_slot[...] = src[interior_cuts[side]]
                elif len(entries) == 1 and entries[0][1] == BC_TAG:
                    if BC[side] == "reflective":
                        if all:
                            src = _view(M, ib, 1)
                            bc_slot[...] = src[interior_cuts[1-side]]
                            bc_slot[self.vels[idim]] *= -1
                    elif BC[side] == "gradfree":
                        if all:
                            src = _view(M, ib, 1)
                            bc_slot[...] = src[interior_cuts[1-side]]
                    elif BC[side] in ("ic", "pressure"):
                        pass   # BC_fv left as-is / pre-populated by caller
                    elif BC[side] == "eq":
                        if all:
                            bc_slot[...] = 0
                    else:
                        raise ValueError(f"Undetermined boundary type: {BC[side]}")
                else:
                    rels = [e[1] for e in entries]
                    raise NotImplementedError(
                        f"Coarse-fine FV boundary not supported yet "
                        f"(ib={ib}, dim={dim}, side={side}, rels={rels}).")

    def fv_apply_BC(self,
                 dim: str) -> None:
        """Copy BC_fv[dim][side] into the ghost slabs of M_fv."""
        ngh=self.Nghc
        idim = self.dims[dim]
        self.dm.M_fv[cut(None, ngh,idim)] = self.BC_fv[dim][0]
        self.dm.M_fv[cut(-ngh,None,idim)] = self.BC_fv[dim][1]

    def fv_Boundaries(self,
                      M: np.ndarray,
                      all=True):
        for dim in self.dims:
            self.fv_store_BC(M,dim,all)
            self.Comms_fv(M,dim)
            self.fv_apply_BC(dim)
    
    def Comms_fv(self,
             M: np.ndarray,
             dim: str):
        comms = self.comms
        comms.Comms_fv(self.dm,
                       M,
                       self.BC_fv,
                       self.dims[dim],
                       dim,
                       self.Nghc)
      
