import numpy as np
from itertools import repeat
from collections import defaultdict
from simulator import Simulator
import riemann_solver as rs
import muscl

from slicing import cut, crop_fv
from amr.tree import SAME, FINER, COARSER, BC as BC_TAG

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

    @staticmethod
    def _fv_avg_pairs(arr: np.ndarray, axes) -> np.ndarray:
        """Average adjacent pairs along each axis in ``axes`` (FV restriction).

        Each named axis must have an even length; it gets halved by mean-
        reducing consecutive pairs of cells.
        """
        for ax in sorted(axes, reverse=True):
            s = arr.shape
            assert s[ax] % 2 == 0, (
                f"FV restriction requires even cell count on axis {ax}; got {s}")
            arr = arr.reshape(s[:ax] + (s[ax] // 2, 2) + s[ax + 1:]).mean(axis=ax + 1)
        return arr

    @staticmethod
    def _fv_inject_pairs(arr: np.ndarray, axes) -> np.ndarray:
        """Replicate each cell twice along each axis in ``axes`` (FV prolongation)."""
        for ax in sorted(axes):
            arr = np.repeat(arr, 2, axis=ax)
        return arr

    def fv_store_BC(self,
             M: np.ndarray,
             dim: str,
             all: bool = True) -> None:
        """Populate BC_fv[dim][side, ..., ib, ...] per block via the forest
        neighbor table. SAME neighbors (including periodic self-wrap) copy
        a slab of the neighbor block; BC-tagged sides apply reflective /
        gradfree / ic / pressure / eq semantics; FINER/COARSER dispatch to
        _fv_fill_from_{finer,coarser} for the coarse-fine slab transfer.
        """
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
                elif len(entries) == 1 and entries[0][1] == COARSER:
                    # I'm fine; coarse neighbor has my ngh fine ghost cells'
                    # worth of data in ceil(ngh/2) of its active cells.
                    # Simplest conservative fill: inject (replicate) coarse
                    # values onto the fine ghost slab, selecting the sub-face
                    # region corresponding to my position.
                    self._fv_fill_from_coarser(M, ib, dim, side, idim, ngh,
                                                interior_cuts, bc_slot,
                                                entries[0])
                elif False not in [e[1] == FINER for e in entries]:
                    # I'm coarse; stack the 2^(ndim-1) fine neighbors'
                    # interior 2*ngh slabs and average 2^ndim fine cells
                    # down to one coarse cell (per-dim average).
                    self._fv_fill_from_finer(M, ib, dim, side, idim, ngh,
                                              bc_slot, entries)
                else:
                    rels = [e[1] for e in entries]
                    raise NotImplementedError(
                        f"Mixed FV neighbor relations {rels} at "
                        f"(ib={ib}, dim={dim}, side={side}).")

    def _fv_fill_from_coarser(self, M, ib, dim, side, idim, ngh,
                               interior_cuts, bc_slot, entry):
        """Populate a fine block's ghost slab from its coarser neighbor.

        Implementation: take the coarse neighbor's inner ceil(ngh/2) cells
        on the appropriate side, crop its transverse-active region to the
        sub-face region this fine block covers, then inject (replicate) by
        2 along each cell dim so the result has fine resolution.

        Currently supports 1D and 2D. Requires ngh to be even.
        """
        ndim = self.ndim
        jb, _rel, sub = entry
        _view = lambda arr, ib_, n: arr[(slice(None),) * n + (ib_,)]
        src = _view(M, jb, 1)
        # Coarse cells needed in normal dim = ngh / 2. Take the coarse
        # neighbor's slab of that depth at the appropriate side.
        assert ngh % 2 == 0, "fv_store_BC CF path requires even Nghc"
        half = ngh // 2
        coarse_cut = (cut(-ngh - half, -ngh, idim) if side == 0
                      else cut(ngh, ngh + half, idim))
        coarse_slab = src[coarse_cut]
        # Select sub-face region in transverse dims. The coarse neighbor's
        # active transverse length is NB_d*n (=twice my own). My block
        # occupies the half determined by `sub`: bit k of sub picks lower
        # (0) / upper (1) half along the k-th non-dim direction in
        # self.dims order (same convention as _sub_face_index).
        dim_keys = list(self.dims.keys())
        # Decode sub_idx -> per-transverse-dim (bit 0, 1, ...) in the same
        # natural self.dims order used by _sub_face_index.
        tv_ranges = {}
        non_dim_k = 0
        for d in dim_keys:
            if self.dims[d] == idim:
                continue
            half = self.NB[d] * self.n[d] // 2
            bit = (sub >> non_dim_k) & 1
            tv_ranges[d] = slice(ngh + bit * half,
                                  ngh + (bit + 1) * half)
            non_dim_k += 1
        # Build the array slicer in axis order: FV layout has cell axes in
        # (z, y, x) order -> iterate reversed(dim_keys).
        tv_slice = [slice(None)]    # nvar
        for d in reversed(dim_keys):
            if self.dims[d] == idim:
                tv_slice.append(slice(None))
            else:
                tv_slice.append(tv_ranges[d])
        coarse_sub = coarse_slab[tuple(tv_slice)]
        # Inject: replicate by 2 on each cell dim -> fine resolution.
        cell_axes = tuple(range(1, ndim + 1))
        fine = self._fv_inject_pairs(coarse_sub, cell_axes)
        # Fill bc_slot's active transverse region; leave transverse-ghost
        # corners for the subsequent dim pass. FV per-block layout has cell
        # axes in (z, y, x) order — iterate `dim_keys` in reverse.
        dest_slice = [slice(None)]
        for d in reversed(dim_keys):
            id_d = self.dims[d]
            if id_d == idim:
                dest_slice.append(slice(None))    # full normal extent
            else:
                dest_slice.append(slice(ngh, -ngh))
        bc_slot[tuple(dest_slice)] = fine

    def _fv_fill_from_finer(self, M, ib, dim, side, idim, ngh,
                              bc_slot, entries):
        """Populate a coarse block's ghost slab from its 2^(ndim-1) fine
        neighbors by stacking their interior 2*ngh slabs and averaging
        2^ndim fine cells down to one coarse cell.

        Supports 1D and 2D. 3D is analogous but not yet implemented.
        """
        ndim = self.ndim
        _view = lambda arr, ib_, n: arr[(slice(None),) * n + (ib_,)]
        # Take each fine neighbor's interior 2*ngh cells on the side that
        # touches me.
        fine_interior = (cut(-3 * ngh, -ngh, idim) if side == 0
                         else cut(ngh, 3 * ngh, idim))
        # Active-transverse slicer: drop each fine neighbor's transverse
        # ghost cells so stacking has consistent shape across sub-faces.
        # FV layout: cell axes in (z, y, x) order.
        dim_keys = list(self.dims.keys())
        active_tv = [slice(None)]     # nvar
        for d in reversed(dim_keys):
            if self.dims[d] == idim:
                active_tv.append(slice(None))    # normal slab
            else:
                active_tv.append(slice(ngh, -ngh))
        active_tv = tuple(active_tv)
        n_sub = 2 ** (ndim - 1)
        fine_slabs = [None] * n_sub
        for (jb, _rel, sub) in entries:
            src = _view(M, jb, 1)
            fine_slabs[sub] = src[fine_interior][active_tv]

        if ndim == 1:
            combined = fine_slabs[0]                  # [nvar, 2*ngh]
            avg_axes = (1,)
        elif ndim == 2:
            # Transverse axis position in the slab: for FV per-block view
            # [nvar, cells_y, cells_x], idim=0 -> transverse is axis 1 (y),
            # idim=1 -> transverse is axis 2 (x). After active_tv, each
            # sub-slab has shape [nvar, NyB*n, 2*ngh] for idim=0.
            tv_axis = 1 if idim == 1 else 2
            # Axis 1 is y for 2D. Wait — FV per-block is [nvar, cells_y,
            # cells_x]. For idim=0 (x), transverse dim y is axis 1. For
            # idim=1 (y), transverse dim x is axis 2. Axes count includes
            # leading nvar.
            tv_axis = {0: 1, 1: 2}[idim]
            combined = np.concatenate(fine_slabs, axis=tv_axis)
            avg_axes = (1, 2)
        else:
            raise NotImplementedError(
                "FV CF fill is 1D/2D-only in this phase; 3D (4 fine "
                "neighbors) will follow in a later pass.")
        coarse_avg = self._fv_avg_pairs(combined, avg_axes)
        # Fill the active-transverse region of bc_slot; transverse corners
        # will be overwritten by the subsequent dim passes of fv_Boundaries.
        dest_slice = [slice(None)]
        for d in reversed(dim_keys):
            if self.dims[d] == idim:
                dest_slice.append(slice(None))
            else:
                dest_slice.append(slice(ngh, -ngh))
        bc_slot[tuple(dest_slice)] = coarse_avg

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
      
