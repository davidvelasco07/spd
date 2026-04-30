import numpy as np
from itertools import repeat
from collections import defaultdict
from simulator import Simulator
import riemann_solver as rs
import muscl

from slicing import cut, crop_fv, indices
from amr.tree import SAME, FINER, COARSER, BC as BC_TAG
from amr.transfer import restrict_blocks_overlap_cv

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
        # Zero-initialize everything. Raw np.ndarray returns uninitialized
        # memory; after reallocation the buffers can contain inf/NaN bits
        # and `correct_fluxes` (theta*F_FB + (1-theta)*F_faces with theta=1
        # for godunov) then evaluates 0*inf = NaN, poisoning the blend.
        self.dm.M_fv  = np.zeros_like(self.array_FV(self.p+1,self.nvar,ngh=self.Nghc))
        self.dm.U_new = np.zeros_like(self.array_FV(self.p+1,self.nvar))
        if self.predictor:
            self.dm.dtM = np.zeros_like(self.array_FV(self.p+1,self.nvar,ngh=self.Nghc-1))
        for dim in self.dims:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"F_faces_{dim}",np.zeros_like(self.array_FV(self.p+1,self.nvar,dim=dim)))
            self.dm.__setattr__(f"F_faces_FB{dim}",np.zeros_like(self.array_FV(self.p+1,self.nvar,dim=dim)))
            self.dm.__setattr__(f"MR_faces_{dim}",np.zeros_like(self.array_FV(self.p+1,self.nvar,dim=dim)))
            self.dm.__setattr__(f"ML_faces_{dim}",np.zeros_like(self.array_FV(self.p+1,self.nvar,dim=dim)))
            self.dm.__setattr__(f"BC_fv_{dim}",np.zeros_like(self.array_FV_BC(dim=dim)))

    def create_dicts_fv(self)->None:
        self.F_faces = defaultdict(list)
        self.F_faces_FB = defaultdict(list)
        self.MR_faces = defaultdict(list)
        self.ML_faces = defaultdict(list)
        self.BC_fv = defaultdict(list)

        for dim in self.dims:
            # Refresh h_fp / h_cv / centers dict entries from dm so they
            # track whichever location (host/device) dm currently holds.
            self.h_fp[dim] = self.dm.__getattribute__(f"h_fp_nb_{dim}")
            self.h_cv[dim] = self.dm.__getattribute__(f"h_cv_nb_{dim}")
            self.centers[dim] = self.dm.__getattribute__(f"centers_{dim}")
            self.F_faces[dim] = self.dm.__getattribute__(f"F_faces_{dim}")
            self.F_faces_FB[dim] = self.dm.__getattribute__(f"F_faces_FB{dim}")
            self.MR_faces[dim] = self.dm.__getattribute__(f"MR_faces_{dim}")
            self.ML_faces[dim] = self.dm.__getattribute__(f"ML_faces_{dim}")
            self.BC_fv[dim] = self.dm.__getattribute__(f"BC_fv_{dim}")
        # Generic per-face neighbor metadata for all dimensions.
        self._build_fv_face_neighbor_map()
        # Precompute AMR-aware CV neighbor metadata (2D) for future SD-layout
        # FV operators that avoid global SD<->FV transpositions.
        self._build_fv_cv_neighbor_map_2d()

    def _build_fv_face_neighbor_map(self) -> None:
        """Build generic FV face-neighbor metadata for all dimensions."""
        face_map = {}
        same_pairs = {}
        same_pair_arrays = {}
        for dim in self.dims:
            side_map = {0: [], 1: []}
            seen = set()
            pairs = []
            pair_buckets = {}
            for side in (0, 1):
                for ib, block in enumerate(self.forest.blocks):
                    entries = block.neighbors[dim][side]
                    rel = entries[0][1] if entries else None
                    info = {
                        "ib": int(ib),
                        "dim": dim,
                        "side": int(side),
                        "entries": entries,
                        "relation": rel,
                    }
                    side_map[side].append(info)
                    if rel == SAME and len(entries) == 1:
                        jb = int(entries[0][0])
                        side_j = 1 - side
                        key = tuple(sorted(((int(ib), int(side)),
                                            (jb, int(side_j))))) + (dim,)
                        if key not in seen:
                            seen.add(key)
                            pairs.append((int(ib), int(side), jb, int(side_j)))
                            pair_buckets.setdefault((int(side), int(side_j)), [[], []])
                            pair_buckets[(int(side), int(side_j))][0].append(int(ib))
                            pair_buckets[(int(side), int(side_j))][1].append(int(jb))
            face_map[dim] = side_map
            same_pairs[dim] = pairs
            same_pair_arrays[dim] = {
                k: {
                    "ib": np.asarray(v[0], dtype=np.int64),
                    "jb": np.asarray(v[1], dtype=np.int64),
                }
                for k, v in pair_buckets.items()
            }
        self.fv_face_neighbor_map = face_map
        self.fv_same_face_pairs_generic = same_pairs
        self.fv_same_face_pair_arrays = same_pair_arrays

    def _build_fv_cv_neighbor_map_2d(self) -> None:
        """Build per-block FV CV neighbor metadata for 2D.

        This map is intentionally lightweight: it captures, for each block face,
        how transverse FV control-volumes align across SAME/COARSER/FINER
        relations. It does not yet replace the runtime FV kernels; it is the
        scaffolding needed to implement SD-layout FV operators.
        """
        self.fv_cv_neighbor_map = {}
        self.fv_same_face_pairs = {}
        if self.ndim != 2:
            return

        n = self.p + 1
        face_data = {}
        same_pairs_by_dim = {}
        cf_pairs_by_dim = {}
        group_map_by_dim = {}
        for dim in ("x", "y"):
            tdim = "y" if dim == "x" else "x"
            Nt = self.NB[tdim] * n
            side_dict = {0: [], 1: []}
            pair_seen = set()
            same_pairs = []
            cf_side = {0: {"coarse_ib": [], "fine_jb": []},
                       1: {"coarse_ib": [], "fine_jb": []}}
            coarser_side = {
                0: {0: {"fine_ib": [], "coarse_jb": []},
                    1: {"fine_ib": [], "coarse_jb": []}},
                1: {0: {"fine_ib": [], "coarse_jb": []},
                    1: {"fine_ib": [], "coarse_jb": []}},
            }
            groups = {
                0: {"same_ib": [], "same_jb": [], "bc_ib": []},
                1: {"same_ib": [], "same_jb": [], "bc_ib": []},
            }
            for side in (0, 1):
                for ib, block in enumerate(self.forest.blocks):
                    entries = block.neighbors[dim][side]
                    rel = entries[0][1] if entries else None
                    info = {
                        "ib": ib,
                        "dim": dim,
                        "side": side,
                        "relation": rel,
                        "entries": entries,
                        "Nt": Nt,
                    }
                    if rel == SAME and len(entries) == 1:
                        info["transverse"] = {
                            "mode": "same",
                            "map": np.arange(Nt, dtype=np.int64),
                        }
                        groups[side]["same_ib"].append(int(ib))
                        groups[side]["same_jb"].append(int(entries[0][0]))
                    elif rel == COARSER and len(entries) == 1:
                        # Fine block boundary -> coarser neighbor boundary.
                        # Two fine transverse CVs collapse into one coarse CV.
                        _jb, _rel, sub = entries[0]
                        info["transverse"] = {
                            "mode": "to_coarser",
                            "sub": int(sub),
                            "map": (np.arange(Nt, dtype=np.int64) // 2
                                    + int(sub) * (Nt // 2)),
                        }
                        if sub < 2:
                            coarser_side[side][int(sub)]["fine_ib"].append(int(ib))
                            coarser_side[side][int(sub)]["coarse_jb"].append(int(_jb))
                    elif rel == FINER and len(entries) > 1:
                        # Coarse block boundary -> finer neighbors boundary.
                        # One coarse transverse CV corresponds to two fine CVs.
                        sub_to_jb = {int(sub): int(jb) for (jb, _rel, sub) in entries}
                        pairs = []
                        for jt in range(Nt):
                            sub = 0 if jt < Nt // 2 else 1
                            local = jt - sub * (Nt // 2)
                            j0 = 2 * local
                            j1 = j0 + 1
                            jb = sub_to_jb[sub]
                            pairs.append(((jb, j0), (jb, j1)))
                        info["transverse"] = {
                            "mode": "to_finer",
                            "pairs": pairs,
                        }
                        sub_to_jb = [None, None]
                        for (jb, _rel, sub) in entries:
                            if sub < 2:
                                sub_to_jb[sub] = int(jb)
                        if (sub_to_jb[0] is not None) and (sub_to_jb[1] is not None):
                            cf_side[side]["coarse_ib"].append(int(ib))
                            cf_side[side]["fine_jb"].append([sub_to_jb[0], sub_to_jb[1]])
                    elif rel == BC_TAG and len(entries) == 1:
                        groups[side]["bc_ib"].append(int(ib))
                    side_dict[side].append(info)
                    if rel == SAME and len(entries) == 1:
                        jb = int(entries[0][0])
                        side_j = 1 - side
                        key = tuple(sorted(((int(ib), int(side)),
                                            (jb, int(side_j))))) + (dim,)
                        if key not in pair_seen:
                            pair_seen.add(key)
                            same_pairs.append((int(ib), int(side), jb, int(side_j)))
            face_data[dim] = side_dict
            same_pairs_by_dim[dim] = same_pairs
            cf_pairs_by_dim[dim] = {
                side: {
                    "coarse_ib": np.asarray(cf_side[side]["coarse_ib"], dtype=np.int64),
                    "fine_jb": np.asarray(cf_side[side]["fine_jb"], dtype=np.int64),
                }
                for side in (0, 1)
            }
            group_map_by_dim[dim] = {
                side: {
                    "same_ib": np.asarray(groups[side]["same_ib"], dtype=np.int64),
                    "same_jb": np.asarray(groups[side]["same_jb"], dtype=np.int64),
                    "bc_ib": np.asarray(groups[side]["bc_ib"], dtype=np.int64),
                    "coarser": {
                        sub: {
                            "fine_ib": np.asarray(
                                coarser_side[side][sub]["fine_ib"], dtype=np.int64
                            ),
                            "coarse_jb": np.asarray(
                                coarser_side[side][sub]["coarse_jb"], dtype=np.int64
                            ),
                        }
                        for sub in (0, 1)
                    },
                }
                for side in (0, 1)
            }
        self.fv_cv_neighbor_map = face_data
        self.fv_same_face_pairs = same_pairs_by_dim
        self.fv_cf_pairs_2d = cf_pairs_by_dim
        self.fv_cv_groups_2d = group_map_by_dim
    
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

        xp = self.dm.xp
        for side in (0, 1):
            same_jb = self.forest.same_jb[dim][side]
            if same_jb is not None:
                # Fast path: every block's neighbor on this face is SAME.
                # One vectorized gather along Nb replaces the Python loop —
                # crucial on GPU where per-block launches dominate.
                self.BC_fv[dim][side][...] = xp.take(
                    M[interior_cuts[side]], xp.asarray(same_jb), axis=1)
                continue
            # 2D map-driven path: same physics as the block-neighbor loop
            # below, but avoids repeatedly traversing the forest topology.
            if self.ndim == 2 and dim in self.fv_cv_neighbor_map:
                infos = self.fv_cv_neighbor_map[dim][side]
                groups = self.fv_cv_groups_2d[dim][side]
                same_ib = groups["same_ib"]
                if same_ib.size:
                    same_jb = groups["same_jb"]
                    self.BC_fv[dim][side][:, same_ib, ...] = (
                        M[interior_cuts[side]][:, same_jb, ...]
                    )
                bc_ib = groups["bc_ib"]
                if bc_ib.size:
                    if BC[side] == "reflective":
                        if all:
                            self.BC_fv[dim][side][:, bc_ib, ...] = (
                                M[interior_cuts[1-side]][:, bc_ib, ...]
                            )
                            self.BC_fv[dim][side][self.vels[idim], bc_ib, ...] *= -1
                    elif BC[side] == "gradfree":
                        if all:
                            self.BC_fv[dim][side][:, bc_ib, ...] = (
                                M[interior_cuts[1-side]][:, bc_ib, ...]
                            )
                    elif BC[side] in ("ic", "pressure"):
                        pass
                    elif BC[side] == "eq":
                        if all:
                            self.BC_fv[dim][side][:, bc_ib, ...] = 0
                    else:
                        raise ValueError(f"Undetermined boundary type: {BC[side]}")
                # COARSER (fine block receiving coarse ghost data), batched by
                # sub-face index in the transverse direction.
                dim_keys = list(self.dims.keys())
                tdim = [d for d in dim_keys if self.dims[d] != idim][0]
                half_t = self.NB[tdim] * self.n[tdim] // 2
                assert ngh % 2 == 0, "fv_store_BC CF path requires even Nghc"
                half_n = ngh // 2
                for sub in (0, 1):
                    cgrp = groups["coarser"][sub]
                    fine_ib = cgrp["fine_ib"]
                    if fine_ib.size == 0:
                        continue
                    coarse_jb = cgrp["coarse_jb"]
                    src = M[:, coarse_jb, ...]
                    coarse_cut = (cut(-ngh - half_n, -ngh, idim) if side == 0
                                  else cut(ngh, ngh + half_n, idim))
                    slab = src[coarse_cut]
                    tv = slice(ngh + sub * half_t, ngh + (sub + 1) * half_t)
                    if dim == "x":
                        coarse_sub = slab[:, :, tv, :]
                        fine = np.repeat(np.repeat(coarse_sub, 2, axis=2), 2, axis=3)
                        self.BC_fv[dim][side][:, fine_ib, ngh:-ngh, :] = fine
                    else:
                        coarse_sub = slab[:, :, :, tv]
                        fine = np.repeat(np.repeat(coarse_sub, 2, axis=2), 2, axis=3)
                        self.BC_fv[dim][side][:, fine_ib, :, ngh:-ngh] = fine
                # FINER (coarse block receiving restricted fine ghost data),
                # batched with precomputed fine-pair block ids.
                fgrp = self.fv_cf_pairs_2d[dim][side]
                coarse_ib = fgrp["coarse_ib"]
                if coarse_ib.size:
                    fine_jb = fgrp["fine_jb"]
                    fine_interior = (cut(-3 * ngh, -ngh, idim) if side == 0
                                     else cut(ngh, 3 * ngh, idim))
                    src0 = M[:, fine_jb[:, 0], ...][fine_interior]
                    src1 = M[:, fine_jb[:, 1], ...][fine_interior]
                    if dim == "x":
                        src0 = src0[:, :, ngh:-ngh, :]
                        src1 = src1[:, :, ngh:-ngh, :]
                        combined = np.concatenate([src0, src1], axis=2)
                        coarse_avg = self._fv_avg_pairs(combined, (2, 3))
                        self.BC_fv[dim][side][:, coarse_ib, ngh:-ngh, :] = coarse_avg
                    else:
                        src0 = src0[:, :, :, ngh:-ngh]
                        src1 = src1[:, :, :, ngh:-ngh]
                        combined = np.concatenate([src0, src1], axis=3)
                        coarse_avg = self._fv_avg_pairs(combined, (2, 3))
                        self.BC_fv[dim][side][:, coarse_ib, :, ngh:-ngh] = coarse_avg
                for info in infos:
                    ib = info["ib"]
                    entries = info["entries"]
                    rel = info["relation"]
                    if rel in (SAME, BC_TAG, COARSER, FINER):
                        continue
                    bc_slot = _view(self.BC_fv[dim][side], ib, 1)
                    rels = [e[1] for e in entries] if entries else []
                    raise NotImplementedError(
                        f"Mixed FV neighbor relations {rels} at "
                        f"(ib={ib}, dim={dim}, side={side}).")
                continue
            for ib, block in enumerate(self.forest.blocks):
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

        # bit_axes: which per-block cell axis each sub_idx bit addresses.
        # Bit k iterates the k-th non-dim in natural self.dims order (matches
        # _sub_face_index). For a per-block FV array [nvar, cells_z, cells_y,
        # cells_x], the cell axis for dim `d` is (ndim - self.dims[d]).
        bit_axes = [ndim - self.dims[d] for d in dim_keys
                    if self.dims[d] != idim]

        if ndim == 1:
            combined = fine_slabs[0]                  # [nvar, 2*ngh]
        elif ndim == 2:
            combined = np.concatenate(fine_slabs, axis=bit_axes[0])
        else:    # ndim == 3: 4 fine neighbors in a 2x2 tile on the face.
            # Inner bit (bit 0) concatenates sub 0+1 and sub 2+3 along
            # the inner transverse axis; outer bit (bit 1) joins those
            # two pairs along the outer transverse axis.
            inner_lo = np.concatenate([fine_slabs[0], fine_slabs[1]],
                                       axis=bit_axes[0])
            inner_hi = np.concatenate([fine_slabs[2], fine_slabs[3]],
                                       axis=bit_axes[0])
            combined = np.concatenate([inner_lo, inner_hi],
                                       axis=bit_axes[1])
        # Average pairs on every cell axis (normal + transverse) to
        # restrict 2**ndim fine cells -> 1 coarse cell.
        avg_axes = tuple(range(1, ndim + 1))
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

    def correct_coarse_fine_fv_flux(self, F_faces: np.ndarray, dim: str) -> None:
        """Enforce conservation across coarse-fine FV faces with overlap-aware
        restriction in the transverse dimensions.

        For p>1 the coarse and fine face control-volumes do not overlap as
        simple pairwise halves; arithmetic averaging leaks conservation.
        This uses the same overlap infrastructure as SD restriction, applied
        to face-CV data in ndim-1 dimensions.
        """
        idim = self.dims[dim]
        ndim = self.ndim
        face_ndim = ndim - 1
        R_side_cv = self.dm.RS_cv
        _view = lambda arr, ib_, n: arr[(slice(None),) * n + (ib_,)]

        if face_ndim == 0:
            for ib, block in enumerate(self.forest.blocks):
                for side in (0, 1):
                    entries = block.neighbors[dim][side]
                    if not entries or entries[0][1] != FINER:
                        continue
                    my_face_idx = 0 if side == 0 else -1
                    fine_face_idx = -1 if side == 0 else 0
                    jb, _rel, _sub = entries[0]
                    _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = (
                        _view(F_faces, jb, 1)[indices(fine_face_idx, idim)]
                    )
            return

        dim_keys = list(self.dims.keys())
        trans_dims = [d for d in dim_keys[::-1] if d != dim]
        n = self.p + 1

        def _face_to_nested(arr_face: np.ndarray) -> np.ndarray:
            # [nvar, ..., NB[d]*n, ...] -> [nvar, NB..., n...]
            shape = [arr_face.shape[0]]
            n_trans = len(trans_dims)
            for d in trans_dims:
                shape += [self.NB[d], n]
            reshaped = arr_face.reshape(shape)
            perm = [0] + [1 + 2 * i for i in range(n_trans)] + [
                2 + 2 * i for i in range(n_trans)
            ]
            return np.transpose(reshaped, perm)

        def _nested_to_face(arr_nested: np.ndarray) -> np.ndarray:
            # [nvar, NB..., n...] -> [nvar, ..., NB[d]*n, ...]
            n_trans = len(trans_dims)
            perm = [0]
            for i in range(n_trans):
                perm += [1 + i, 1 + n_trans + i]
            interleaved = np.transpose(arr_nested, perm)
            out_shape = [arr_nested.shape[0]]
            for d in trans_dims:
                out_shape.append(self.NB[d] * n)
            return interleaved.reshape(out_shape)

        if self.ndim == 2 and dim in self.fv_cf_pairs_2d:
            n = self.p + 1
            tdim = "y" if dim == "x" else "x"
            NB_t = self.NB[tdim]
            for side in (0, 1):
                pair = self.fv_cf_pairs_2d[dim][side]
                coarse_ib = pair["coarse_ib"]
                if coarse_ib.size == 0:
                    continue
                fine_jb = pair["fine_jb"]
                my_face_idx = 0 if side == 0 else -1
                fine_face_idx = -1 if side == 0 else 0
                coarse_face = F_faces[indices(my_face_idx, idim)]   # [nvar, Nb, Nt]
                fine_face = F_faces[indices(fine_face_idx, idim)]   # [nvar, Nb, Nt]
                fine0 = fine_face[:, fine_jb[:, 0], :]              # [nvar, Nif, Nt]
                fine1 = fine_face[:, fine_jb[:, 1], :]
                fine0_nested = fine0.reshape(fine0.shape[0], fine0.shape[1], NB_t, n)
                fine1_nested = fine1.reshape(fine1.shape[0], fine1.shape[1], NB_t, n)
                stack = np.stack((fine0_nested, fine1_nested), axis=2)  # child axis=2
                coarse_nested = restrict_blocks_overlap_cv(stack, R_side_cv, 1)
                coarse_flat = coarse_nested.reshape(
                    coarse_nested.shape[0], coarse_nested.shape[1], NB_t * n
                )
                coarse_face[:, coarse_ib, :] = coarse_flat
            return

        if dim in self.fv_face_neighbor_map:
            for side in (0, 1):
                for info in self.fv_face_neighbor_map[dim][side]:
                    ib = info["ib"]
                    entries = info["entries"]
                    if not entries or info["relation"] != FINER:
                        continue
                    my_face_idx   = 0 if side == 0 else -1
                    fine_face_idx = -1 if side == 0 else 0
                    n_sub = 2 ** face_ndim
                    fine_fluxes = [None] * n_sub
                    for (jb, _rel, sub) in entries:
                        src = _view(F_faces, jb, 1)
                        fine_fluxes[sub] = _face_to_nested(src[indices(fine_face_idx, idim)])
                    stack = np.stack(fine_fluxes, axis=1)
                    coarse_nested = restrict_blocks_overlap_cv(
                        stack, R_side_cv, face_ndim
                    )
                    _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = (
                        _nested_to_face(coarse_nested)
                    )
            return

        for ib, block in enumerate(self.forest.blocks):
            for side in (0, 1):
                entries = block.neighbors[dim][side]
                if not entries or entries[0][1] != FINER:
                    continue
                my_face_idx   = 0 if side == 0 else -1
                fine_face_idx = -1 if side == 0 else 0
                n_sub = 2 ** face_ndim
                fine_fluxes = [None] * n_sub
                for (jb, _rel, sub) in entries:
                    src = _view(F_faces, jb, 1)
                    fine_fluxes[sub] = _face_to_nested(src[indices(fine_face_idx, idim)])
                stack = np.stack(fine_fluxes, axis=1)
                coarse_nested = restrict_blocks_overlap_cv(
                    stack, R_side_cv, face_ndim
                )
                _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = (
                    _nested_to_face(coarse_nested)
                )

    def symmetrize_same_level_fv_flux(self, F_faces: np.ndarray, dim: str) -> None:
        """Enforce a unique flux at SAME-level block interfaces.

        At block interfaces where both neighbors are SAME-level, each side can
        carry slightly different face fluxes (notably in FB/Godunov AMR paths).
        This routine projects each interface pair onto the conservative manifold:
          - opposite sides (1<->0):    F_i =  F_j = 0.5*(F_i + F_j)
          - same sides (0<->0,1<->1): F_i = -F_j = 0.5*(F_i - F_j)
        restoring pairwise cancellation in the global divergence sum.
        """
        idim = self.dims[dim]
        _view = lambda arr, ib_, n: arr[(slice(None),) * n + (ib_,)]
        if self.ndim == 2 and dim in self.fv_same_face_pairs:
            for (ib, side, jb, side_j) in self.fv_same_face_pairs[dim]:
                my_face_idx = 0 if side == 0 else -1
                nb_face_idx = 0 if side_j == 0 else -1
                my_face = _view(F_faces, ib, 1)[indices(my_face_idx, idim)]
                nb_face = _view(F_faces, jb, 1)[indices(nb_face_idx, idim)]
                avg = 0.5 * (my_face + nb_face)
                _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = avg
                _view(F_faces, jb, 1)[indices(nb_face_idx, idim)] = avg
            return
        if dim in self.fv_same_face_pair_arrays:
            F0 = F_faces[indices(0, idim)]
            Fm1 = F_faces[indices(-1, idim)]
            for (side, side_j), pair in self.fv_same_face_pair_arrays[dim].items():
                ib = pair["ib"]
                jb = pair["jb"]
                if ib.size == 0:
                    continue
                my_face = F0[:, ib, ...] if side == 0 else Fm1[:, ib, ...]
                nb_face = F0[:, jb, ...] if side_j == 0 else Fm1[:, jb, ...]
                avg = 0.5 * (my_face + nb_face)
                if side == 0:
                    F0[:, ib, ...] = avg
                else:
                    Fm1[:, ib, ...] = avg
                if side_j == 0:
                    F0[:, jb, ...] = avg
                else:
                    Fm1[:, jb, ...] = avg
            return
        if dim in self.fv_same_face_pairs_generic:
            for (ib, side, jb, side_j) in self.fv_same_face_pairs_generic[dim]:
                my_face_idx = 0 if side == 0 else -1
                nb_face_idx = 0 if side_j == 0 else -1
                my_face = _view(F_faces, ib, 1)[indices(my_face_idx, idim)]
                nb_face = _view(F_faces, jb, 1)[indices(nb_face_idx, idim)]
                avg = 0.5 * (my_face + nb_face)
                _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = avg
                _view(F_faces, jb, 1)[indices(nb_face_idx, idim)] = avg
            return

        seen = set()

        for ib, block in enumerate(self.forest.blocks):
            for side in (0, 1):
                entries = block.neighbors[dim][side]
                if not entries or len(entries) != 1 or entries[0][1] != SAME:
                    continue
                jb, _rel, _sub = entries[0]
                # A block pair can share two SAME interfaces under periodic
                # wrapping (e.g. 2 blocks in a periodic direction). Distinguish
                # interfaces by side pairing, not just by block ids.
                side_j = 1 - side
                key = tuple(sorted(((ib, side), (jb, side_j)))) + (dim,)
                if key in seen:
                    continue
                seen.add(key)
                my_face_idx = 0 if side == 0 else -1
                nb_face_idx = 0 if side_j == 0 else -1
                my_face = _view(F_faces, ib, 1)[indices(my_face_idx, idim)]
                nb_face = _view(F_faces, jb, 1)[indices(nb_face_idx, idim)]
                avg = 0.5 * (my_face + nb_face)
                _view(F_faces, ib, 1)[indices(my_face_idx, idim)] = avg
                _view(F_faces, jb, 1)[indices(nb_face_idx, idim)] = avg

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
      
