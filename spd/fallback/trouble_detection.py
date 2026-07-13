import numpy as np
from spd.simulator import Simulator
from spd.numerics.slicing import cut, crop_fv
from spd.runtime.gpu import CUPY_AVAILABLE, is_gpu_array

if CUPY_AVAILABLE:
    import cupy as cp

    # Fused smooth-extrema kernel: replaces the ~10 elementwise kernels
    # (and their temporaries) between the derivative stencils and the
    # neighborhood minimum.  Outputs both the raw left alpha (needed for
    # the boundary handling of compute_min) and min(alphaL, alphaR).
    smooth_extrema_k = cp.ElementwiseKernel(
        "T U0, T U1, T U2, T U3, T U4, "
        "T x0, T x1, T x2, T x3, T x4, T h",
        "T aL, T alpha",
        """
        // First/second derivative stencils computed inline from the
        // 5-point neighborhood (fused; no dU/d2U temporaries).
        T dUm = (U2 - U0) / (x2 - x0);
        T dU0 = (U3 - U1) / (x3 - x1);
        T dUp = (U4 - U2) / (x4 - x2);
        T dv = 0.5 * h * ((dUp - dUm) / (x3 - x1));
        if (dv == 0) {
            aL = 1;
            alpha = 1;
        } else {
            T vL = dUm - dU0;
            T vR = dUp - dU0;
            aL = -((dv < 0) ? (vL > 0 ? vL : (T)0) : (vL < 0 ? vL : (T)0)) / dv;
            aL = aL < 1 ? aL : (T)1;
            T aR = ((dv > 0) ? (vR > 0 ? vR : (T)0) : (vR < 0 ? vR : (T)0)) / dv;
            aR = aR < 1 ? aR : (T)1;
            alpha = aL < aR ? aL : aR;
        }
        """,
        "sd_smooth_extrema_k",
    )

    # Fused NAD check: tolerance-relaxed DMP bounds, bound violation test,
    # and the smooth-extrema relaxation in a single kernel (replaces the
    # ~8 elementwise kernels of the where/arithmetic chain).
    # The conditions are phrased so NaN candidates are flagged as troubled
    # (comparisons with NaN are false), never silently admitted.
    nad_k = cp.ElementwiseKernel(
        "T W, T Wmin, T Wmax, T alpha, float64 tol, bool delta_mode, bool use_sed",
        "T out",
        """
        T lo, hi;
        if (delta_mode) {
            T eps = tol * (Wmax - Wmin);
            lo = Wmin - eps;
            hi = Wmax + eps;
        } else {
            lo = Wmin - fabs(Wmin) * tol;
            hi = Wmax + fabs(Wmax) * tol;
        }
        out = (W >= lo && W <= hi) ? (T)0 : (T)1;
        if (use_sed && alpha >= 1) out = 0;
        """,
        "sd_nad_k",
    )

    # Fused PAD check: flag non-physical (or NaN) density/pressure on top
    # of the existing trouble markers (in place).
    pad_k = cp.ElementwiseKernel(
        "T tr, T rho, T p, float64 min_rho, float64 max_rho, float64 min_P",
        "T out",
        """
        out = (rho >= min_rho && rho <= max_rho && p >= min_P) ? tr : (T)1;
        """,
        "sd_pad_k",
    )

    # Neighbor-class offsets for the filter-based apply_blending path:
    # 0.5 for the edge-diagonal (second) neighbors and 0.375 for the corner
    # (third) neighbors, matching the symmetric CPU slicing loops.
    _blending_offsets = {
        1: [],
        2: [(0.5, [(-1, -1), (-1, 1), (1, -1), (1, 1)])],
        3: [
            (0.5, [(sk, sj, si) for sk in (-1, 0, 1) for sj in (-1, 0, 1)
                   for si in (-1, 0, 1)
                   if (sk != 0) + (sj != 0) + (si != 0) == 2]),
            (0.375, [(sk, sj, si) for sk in (-1, 1) for sj in (-1, 1)
                     for si in (-1, 1)]),
        ],
    }

    _blending_fps_cache = {}

    def _blending_footprints(ndim):
        if ndim not in _blending_fps_cache:
            center = np.ones(ndim, dtype=int)
            fps = []
            face_offsets = [
                tuple(s * e for e in row)
                for row in np.eye(ndim, dtype=int)
                for s in (-1, 1)
            ]
            classes = [(0.75, face_offsets)] + _blending_offsets[ndim]
            for w, offsets in classes:
                fp = np.zeros((3,) * ndim, dtype=bool)
                for o in offsets:
                    # maximum_filter footprints are mirrored: reading
                    # offset o corresponds to footprint index center - o.
                    fp[tuple(center - np.array(o))] = True
                fps.append((w, cp.asarray(fp)))
            _blending_fps_cache[ndim] = fps
        return _blending_fps_cache[ndim]


def detect_troubles_induction(
    scheme,
    tolerance: float = 0.05,
    blending: bool = True,
) -> None:
    """
    Mark troubled CVs from |B^2| variation (induction / MHD fallback).
    Fills ``scheme.dm.troubles`` and optional ``theta`` / ``affected_faces_*``.
    """
    if getattr(scheme, "godunov", False):
        return
    ngh = scheme.Nghc
    crop = lambda start, end, idim: crop_fv(start, end, idim, scheme.ndim, ngh)
    B2 = scheme.compute_B2()
    trouble = np.zeros_like(B2, dtype=np.int32)
    for idim in scheme.idims:
        jump = np.abs(B2[crop(2, -1, idim)] - B2[crop(None, -2, idim)])
        mx = np.maximum(
            np.abs(B2[crop(2, -1, idim)]), np.abs(B2[crop(1, -2, idim)])
        )
        local = jump / (mx + 1e-20) > tolerance
        trouble[crop(1, -1, idim)] = np.maximum(
            trouble[crop(1, -1, idim)], local.astype(np.int32)
        )
    scheme.dm.troubles[...] = trouble
    th0 = scheme.dm.theta[0]
    th0[...] = trouble.astype(th0.dtype)
    if blending:
        tr = trouble.astype(np.float64)
        for idim in scheme.idims:
            th0[crop(None, -1, idim)] = np.maximum(
                th0[crop(None, -1, idim)], 0.75 * tr[crop(1, None, idim)]
            )
            th0[crop(1, None, idim)] = np.maximum(
                th0[crop(1, None, idim)], 0.75 * tr[crop(None, -1, idim)]
            )
    for dim in scheme.dims:
        idim = scheme.dims[dim]
        affected = scheme.dm.__getattribute__(f"affected_faces_{dim}")
        affected[...] = 0
        affected[...] = np.maximum(
            scheme.dm.theta[0][crop(ngh - 1, -ngh, idim)],
            scheme.dm.theta[0][crop(ngh, -(ngh - 1), idim)],
        )

def nad_check(W_new, W_min, W_max, alpha, tolerance, delta_mode):
    """Numerical-admissibility check (dispatcher).

    Flags cells whose new solution leaves the tolerance-relaxed DMP bounds
    [W_min, W_max], except at smooth extrema (alpha >= 1, when given).
    Single fused kernel on the GPU; numpy chain on the CPU.
    """
    if is_gpu_array(W_new):
        return nad_k(
            W_new, W_min, W_max,
            alpha if alpha is not None else W_min,
            tolerance,
            delta_mode,
            alpha is not None,
        )
    if delta_mode:
        epsilon = tolerance * (W_max - W_min)
        W_min = W_min - epsilon
        W_max = W_max + epsilon
    else:
        W_min = W_min - np.abs(W_min) * tolerance
        W_max = W_max + np.abs(W_max) * tolerance
    # Phrased so NaN candidates are flagged (NaN comparisons are false).
    trouble = np.where((W_new >= W_min) & (W_new <= W_max), 0, 1)
    if alpha is not None:
        trouble *= np.where(alpha >= 1, 0, 1)
    return trouble


def pad_check(troubles, rho, P, min_rho, max_rho, min_P):
    """Physical-admissibility check (dispatcher).

    Flags cells with non-physical density or pressure on top of the
    existing trouble markers and returns the updated array.
    """
    if is_gpu_array(troubles):
        pad_k(troubles, rho, P, min_rho, max_rho, min_P, troubles)
        return troubles
    # Phrased so NaN density/pressure is flagged (NaN comparisons are false).
    return np.where(
        (rho >= min_rho) & (rho <= max_rho) & (P >= min_P), troubles, 1
    )


def detect_troubles(self: Simulator):
    if self.godunov:
        return

    # Reset to check troubled control volumes
    ngh=self.Nghc
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
    lv = list(self.limiting_variables)
    self.dm.troubles[...] = 0
    # W_cv was filled from U_cv by compute_corrected_fluxes just before this
    # call, so reuse it instead of recomputing the primitives.
    # W_old -> s.dm.M
    W_new = self.compute_primitives_cv(self.dm.U_new)
    ##############################################
    # NAD Check for numerically adimissible values
    ##############################################
    # First check if DMP criteria is met, if it is we can avoid computing alpha
    # The NAD/SED pipeline only feeds the trouble flag through the limiting
    # variables, so restrict the (expensive) neighborhood work to those rows.
    if not self._W_ghosts_current:
        # Otherwise dm.M already holds the ghosted W_cv (filled by the MUSCL
        # flux pipeline just before this call, with boundaries applied).
        self.fill_active_region(self.W_cv)
        self.Boundaries(self.dm.M)
    W_max, W_min = neighborhood_extrema(
        self.dm.M[lv], self.ndim, getattr(self, "NAD_neighbors", "1st")
    )

    W_max = self.crop(W_max,ngh=ngh)
    W_min = self.crop(W_min,ngh=ngh)

    W_new_lv = W_new[lv]

    # Smooth extrema detection: alpha < 1 marks a genuine extremum.
    # Computed before the bound check so the fused GPU kernel can apply
    # bounds and SED relaxation in one pass.
    alpha = None
    if self.p > 1 and self.SED:
        self.fill_active_region(W_new)
        self.Boundaries(self.dm.M)
        M_lv = self.dm.M[lv]
        alpha = W_new_lv*0 + 1
        for dim in self.dims:
            idim = self.dims[dim]
            alpha_new = compute_smooth_extrema(self, M_lv, dim)[crop(None,None,idim)]
            alpha = np.where(alpha_new < alpha, alpha_new, alpha)

    tolerance = self.tolerance if self.p > 0 else 0.0
    possible_trouble = nad_check(
        W_new_lv, W_min, W_max, alpha, tolerance, self.NAD == "delta"
    )

    self.dm.troubles[...] = np.amax(possible_trouble,axis=0)
    
    ###########################
    # PAD Check for physically admissible values
    ###########################
    if self.PAD:
        if self.WB:
            # U_eq_cv resolves through the state-owning dm (the primary's,
            # for a fallback scheme), already in FV layout at this point.
            W_new += self.compute_primitives(self.U_eq_cv)
        self.dm.troubles = pad_check(
            self.dm.troubles,
            W_new[self._d_, ...], W_new[self._p_, ...],
            self.min_rho, self.max_rho, self.min_P,
        )

    #self.n_troubles += self.dm.troubles.sum()
    self.dm.M[...] = 0
    self.fill_active_region(self.dm.troubles)
    self.Boundaries_scalar(self.dm.M)
    trouble = self.dm.M[0]
    self.dm.theta[0][...] = trouble
    theta = self.dm.theta[0]

    if self.blending:
        apply_blending(self,trouble,theta)

    for dim in self.dims:
        idim = self.dims[dim]
        affected_faces = self.dm.__getattribute__(f"affected_faces_{dim}")
        affected_faces[...] = 0
        affected_faces[...] = np.maximum(theta[crop(ngh-1,-ngh,idim)],theta[crop(ngh,-(ngh-1),idim)])

def neighborhood_extrema(M, ndim, neighbors):
    """Per-cell (max, min) over the neighborhood of the trailing ndim axes.

    neighbors="2nd" uses the Moore (3^ndim box) neighborhood, anything else
    the von Neumann (face/cross) neighborhood; both include the center.
    On the GPU this is a single filter call each instead of the chained
    per-dimension slicing kernels (which are kept as the CPU path).
    ``mode='nearest'`` reproduces the edge handling of the chained version.
    """
    if is_gpu_array(M):
        from cupyx.scipy import ndimage as cundi
        if neighbors == "2nd":
            size = (1,) + (3,) * ndim
            return (
                cundi.maximum_filter(M, size=size, mode="nearest"),
                cundi.minimum_filter(M, size=size, mode="nearest"),
            )
        fp = np.zeros((1,) + (3,) * ndim, dtype=bool)
        center = (0,) + (1,) * ndim
        fp[center] = True
        for ax in range(ndim):
            for off in (0, 2):
                idx = list(center)
                idx[1 + ax] = off
                fp[tuple(idx)] = True
        return (
            cundi.maximum_filter(M, footprint=cp.asarray(fp), mode="nearest"),
            cundi.minimum_filter(M, footprint=cp.asarray(fp), mode="nearest"),
        )
    W_max = M.copy()
    W_min = M.copy()
    if neighbors == "2nd":
        # Chain the extrema so each successive dimension operates on the
        # already-extended array, pulling in the diagonal neighbors too.
        for idim in range(ndim):
            W_max = compute_W_max(W_max, idim)
            W_min = compute_W_min(W_min, idim)
    else:
        # Per-dimension extrema combined: only the face-adjacent neighbors.
        for idim in range(ndim):
            W_max = np.maximum(compute_W_max(M, idim), W_max)
            W_min = np.minimum(compute_W_min(M, idim), W_min)
    return W_max, W_min

def compute_W_ex(W, idim, f):
    W_f = W.copy()
    # W_f(i) = f(W(i-1),W(i),W(i+1))
    # First comparing W(i) and W(i+1)
    W_f[cut(None,-1,idim)] = f(  W[cut(None,-1,idim)],W[cut(1, None,idim)])
    # Now comparing W_f(i) and W_(i-1)
    W_f[cut( 1,None,idim)] = f(W_f[cut( 1,None,idim)],W_f[cut(None,-1,idim)])
    return W_f

def compute_W_max(W, idim):
    return compute_W_ex(W, idim, np.maximum)

def compute_W_min(W, idim):
    return compute_W_ex(W, idim, np.minimum)

def first_order_derivative(U, h, idim):
    dU = (U[cut(2,None,idim)] - U[cut(None,-2,idim)])/(h[cut(2,None,idim)] - h[cut(None,-2,idim)])
    return dU

def compute_min(A, Amin, idim):
    Amin[cut(None,-1,idim)] = np.minimum(A[cut(None,-1,idim)],   A[cut(1,None,idim)])
    Amin[cut( 1,None,idim)] = np.minimum(A[cut(None,-1,idim)],Amin[cut(1,None,idim)])

def compute_smooth_extrema(self, U, dim):
    eps = 0
    idim = self.dims[dim]
    centers = self.centers[dim][self.shape(idim)]
    if is_gpu_array(U):
        # Fully fused path: derivatives and the alpha_L/alpha_R chain are
        # computed inside one kernel from the shifted 5-point views.
        sl = lambda a, b: cut(a, b, idim)
        aL, alpha = smooth_extrema_k(
            U[sl(None, -4)], U[sl(1, -3)], U[sl(2, -2)],
            U[sl(3, -1)], U[sl(4, None)],
            centers[sl(None, -4)], centers[sl(1, -3)], centers[sl(2, -2)],
            centers[sl(3, -1)], centers[sl(4, None)],
            self.h_fp[dim][sl(2, -2)],
        )
        compute_min(alpha, aL, idim)
        return aL
    # First derivative dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    dU  = first_order_derivative( U, centers, idim)
    # Second derivative d2Udx2(i) = [dU(i+1)-dU(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    d2U = first_order_derivative(dU, centers[cut(1,-1,idim)], idim)
    dv = 0.5 * self.h_fp[dim][cut(2,-2,idim)] * d2U
    # vL = dU(i-1)-dU(i)
    vL = dU[cut(None,-2,idim)] - dU[cut(1,-1,idim)]
    # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
    alphaL = (
        -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0)) / dv
    )
    alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
    alphaL = np.where(alphaL < 1, alphaL, 1)
    # vR = dU(i+1)-dU(i)
    vR = dU[cut( 2,None,idim)] - dU[cut(1,-1,idim)]
    # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
    alphaR = np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0)) / dv
    alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
    alphaR = np.where(alphaR < 1, alphaR, 1)
    alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
    compute_min(alphaR, alphaL, idim)
    return alphaL

def apply_blending(self,trouble,theta):
    if is_gpu_array(theta):
        return apply_blending_gpu(self, trouble, theta)
    a = slice(None,-1)
    b = slice( 1,None)
    cuts = [(a,a),(a,b),(b,a),(b,b)]
    #First neighbors
    for idim in self.idims:
        theta[cut(None,-1,idim)] = np.maximum(.75*trouble[cut( 1,None,idim)],theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.75*trouble[cut(None,-1,idim)],theta[cut( 1,None,idim)])
          
    if self.ndim==2:
        #Second neighbors
        for i in range(len(cuts)):
            theta[cuts[i]] = np.maximum(.5*trouble[cuts[::-1][i]],theta[cuts[i]])
                
    elif self.ndim==3:
        #Second neighbors
        for i in range(len(cuts)):
            for idim in self.idims:
                shape1 = tuple(np.roll(np.array((slice(None),)+cuts[ i]),-idim))
                shape2 = tuple(np.roll(np.array((slice(None),)+cuts[::-1][i]),-idim))
                theta[shape1] = np.maximum(.5*trouble[shape2],theta[shape1])
        #Third neighbors
        cuts1 = [(x,y,z) for x in (a,b) for y in (a,b) for z in (a,b)]
        cuts2 = [(x,y,z) for x in (b,a) for y in (b,a) for z in (b,a)]
        for i in range(len(cuts1)):
            theta[cuts1[i]] = np.maximum(.375*trouble[cuts2[i]],theta[cuts1[i]])
    
    #Last layer
    for idim in self.idims:
        theta[cut(None,-1,idim)] = np.maximum(.25*(theta[cut( 1,None,idim)]>0),theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.25*(theta[cut(None,-1,idim)]>0),theta[cut( 1,None,idim)])


def apply_blending_gpu(self, trouble, theta):
    """Filter-based equivalent of apply_blending for device arrays.

    The CPU version spreads trouble to neighbors through ~15 sliced
    ``np.maximum`` updates.  All of those read only ``trouble`` (the
    neighbor classes) or amount to a binary dilation of the positive
    region (the sequential 0.25 "last layer" adds exactly one box-shaped
    ring, since every positive theta is already >= 0.375 > 0.25), so the
    whole operation reduces to a few maximum_filter calls.  Out-of-range
    neighbors never contribute in the sliced version, hence
    ``mode='constant', cval=0``.
    """
    from cupyx.scipy import ndimage as cundi

    for w, fp in _blending_footprints(self.ndim):
        f = cundi.maximum_filter(trouble, footprint=fp, mode="constant", cval=0.0)
        cp.maximum(theta, w * f, out=theta)
    ring = cundi.maximum_filter(theta, size=3, mode="constant", cval=0.0)
    cp.maximum(theta, 0.25 * (ring > 0), out=theta)
     
        
