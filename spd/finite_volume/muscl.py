import numpy as np
from spd.simulator import Simulator

from spd.numerics.slicing import cut
from spd.numerics.slicing import crop_fv
from spd.runtime.gpu import CUPY_AVAILABLE, is_gpu_array

if CUPY_AVAILABLE:
    import cupy as cp

    # Slope limiters as device functions.  Each entry provides
    # ``limited_slope(dL, dR, hL, hR, hM)`` used by the fused face
    # reconstruction kernel below; adding a limiter only requires a new
    # device-function string here (the CPU path keeps the numpy methods).
    _LIMITER_DEVICE = {
        "minmod": """
        template<typename T>
        __device__ T limited_slope(T dL, T dR, T hL, T hR, T hM) {
            // Mirrors the numpy version: r=dR/dL clipped to [0,1]; the
            // NaN from dL==0 compares false and falls through to 0*dL=0.
            T r = dR / dL;
            r = r < 1 ? r : (T)1;
            r = r > 0 ? r : (T)0;
            return r * dL;
        }
        """,
        "moncen": """
        template<typename T>
        __device__ T limited_slope(T dL, T dR, T hL, T hR, T hM) {
            T dC = (hL * dL + hR * dR) / (hL + hR);
            T s = min(fabs(2 * dL * hL / hM), fabs(2 * dR * hR / hM));
            s = min(s, fabs(dC));
            s = dC >= 0 ? s : -s;
            return (dL * dR >= 0) ? s : (T)0;
        }
        """,
    }

    _muscl_face_kernels = {}

    def _muscl_face_kernel(limiter: str):
        """Fused MUSCL face reconstruction (slopes + L/R interpolation).

        Elementwise over faces: face k sits between cells k+1 and k+2 of the
        ghosted array; the kernel receives the four cells of the stencil as
        shifted views and computes both limited cell slopes inline.  Replaces
        the slope/interpolation kernel chain and all its temporaries with a
        single launch per dimension.
        """
        if limiter not in _muscl_face_kernels:
            if limiter not in _LIMITER_DEVICE:
                raise NotImplementedError(
                    f"No device function for slope limiter '{limiter}'"
                )
            _muscl_face_kernels[limiter] = cp.ElementwiseKernel(
                "T M0, T M1, T M2, T M3, "
                "T hcv0, T hcv1, T hcv2, T hfp1, T hfp2",
                "T ML, T MR",
                """
                T dM0 = (M1 - M0) / hcv0;
                T dM1 = (M2 - M1) / hcv1;
                T dM2 = (M3 - M2) / hcv2;
                ML = M1 + 0.5 * hfp1 * limited_slope(dM0, dM1, hcv0, hcv1, hfp1);
                MR = M2 - 0.5 * hfp2 * limited_slope(dM1, dM2, hcv1, hcv2, hfp2);
                """,
                f"fv_muscl_faces_{limiter}",
                preamble=_LIMITER_DEVICE[limiter],
            )
        return _muscl_face_kernels[limiter]


def enforce_face_positivity(self, dim: str, idim: int) -> None:
    """RAMSES-style positivity guard on the reconstructed face states:
    where the reconstructed density or pressure falls at/below its floor,
    that variable reverts to the donor-cell (cell-centered) value.  The
    face states are primitives, so only the density and pressure rows are
    touched (the reconstruction of the other variables stays intact)."""
    ngh = self.Nghc
    crop = lambda start, end: crop_fv(start, end, idim, self.ndim, ngh)
    M = self.dm.M
    min_rho = getattr(self, "min_rho", 1e-10)
    min_P = getattr(self, "min_P", 1e-10)
    for var, floor in ((self._d_, min_rho), (self._p_, min_P)):
        ML = self.ML_fp[dim][var]
        MR = self.MR_fp[dim][var]
        ML[...] = np.where(ML > floor, ML, M[crop(1, -2)][var])
        MR[...] = np.where(MR > floor, MR, M[crop(2, -1)][var])


def reconstruct_faces(self, dim: str, idim: int) -> None:
    """Fill ML_fp/MR_fp for *dim* with the MUSCL reconstruction (dispatcher).

    On the GPU (for limiters with a device function) this is a single fused
    kernel; otherwise slopes and interpolations run as separate numpy ops.
    is_gpu_array is False whenever CuPy is unavailable, so _LIMITER_DEVICE
    (only defined under CUPY_AVAILABLE) is never referenced in that case.
    """
    if is_gpu_array(self.dm.M) and self.slope_limiter.limiter in _LIMITER_DEVICE:
        reconstruct_faces_gpu(self, dim, idim)
    else:
        S = self.compute_slopes(self.dm.M, idim)
        self.MR_fp[dim][...] = self.interpolate_R(self.dm.M, S, idim)
        self.ML_fp[dim][...] = self.interpolate_L(self.dm.M, S, idim)
    enforce_face_positivity(self, dim, idim)


def reconstruct_faces_gpu(self, dim: str, idim: int) -> None:
    """Fill ML_fp/MR_fp for *dim* with the fused GPU reconstruction."""
    ngh = self.Nghc
    M = self.dm.M
    crop = lambda start, end: crop_fv(start, end, idim, self.ndim, ngh)
    h_cv = self.h_cv[dim]
    h_fp = self.h_fp[dim]
    kernel = _muscl_face_kernel(self.slope_limiter.limiter)
    kernel(
        M[crop(None, -3)],
        M[crop(1, -2)],
        M[crop(2, -1)],
        M[crop(3, None)],
        h_cv[cut(None, -2, idim)],
        h_cv[cut(1, -1, idim)],
        h_cv[cut(2, None, idim)],
        h_fp[cut(1, -2, idim)],
        h_fp[cut(2, -1, idim)],
        self.ML_fp[dim],
        self.MR_fp[dim],
    )


class Slope_limiter:
    def __init__(self,limiter):
        self.limiter = limiter
        self.compute_gradients = self.gradient_limiter(self.__getattribute__(limiter))

    def minmod(self,
               SlopeL: np.ndarray,
               SlopeR: np.ndarray,
               **kwargs)->np.ndarray:
        """
        Returns the minmod limited slopes

        Parameters
        ----------
            SlopeL/R: Solution vector with Left/Right slopes

        Returns
        -------
            Slopes: Limited slopes
        """
        #First compute ratio between slopes SlopeR/SlopeL
        #Then limit the ratio to be lower than 1
        #Finally, limit the ratio to be positive and multiply
        #  by SlopeL to get the limited slope at the cell center
        #We use "where" instead of "maximum/minimum" as it doesn't
        # propagte the NaNs caused when SlopeL=0
        ratio = SlopeR/SlopeL
        ratio = np.where(ratio<1,ratio,1)
        return np.where(ratio>0,ratio,0)*SlopeL

    def moncen(self,
               dU_L: np.ndarray,
               dU_R: np.ndarray,
               dx_L: np.ndarray,
               dx_R: np.ndarray,
               dx_M: np.ndarray)->np.ndarray:
        """
        Returns the moncen limited slopes

        Parameters
        ----------
            dU_L/R: Solution vector with Left/Right slopes
            dx_L/R: vector of cell sizes (distance between cell centers)
            dx_M:   vector of cell sizes (distance between flux points)

        Returns
        -------
            Slopes: Limited slopes
        """
        dU_C = (dx_L*dU_L + dx_R*dU_R)/(dx_L+dx_R)
        slope = np.minimum(np.abs(2*dU_L*dx_L/dx_M),np.abs(2*dU_R*dx_R/dx_M))
        slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
        return np.where(dU_L*dU_R>=0,slope,0)     

    def gradient_limiter(self,limiter):
        def limit_gradients(
            M: np.ndarray,
            h_cv: np.ndarray,
            h_fp: np.ndarray,
            idim: int,)->np.ndarray:
            dM = (M[cut(1,None,idim)] - M[cut(None,-1,idim)])/h_cv
            dMh = limiter(dM[cut(None,-1,idim)],
                          dM[cut(1,None,idim)],
                          dx_L = h_cv[cut(None,-1,idim)],
                          dx_R = h_cv[cut(1,None,idim)],
                          dx_M = h_fp[cut(1,-1,idim)])
            return dMh
        return limit_gradients 

    def compute_slopes(
            self,
            M: np.ndarray,
            h_cv: np.ndarray,
            h_fp: np.ndarray,
            idim: int,)->np.ndarray:
        """
        Returns array of limited slopes

        Parameters
        ---------- 
            M:          Solution vector (conservatives/primitives)
            h_cv:       vector of cell sizes (distance between cell centers)
            h_fp:       vector of cell sizes (distance between flux points)
            idim:       index of dimension

        Returns
        -------
            S:          Slopes of M
        """
        dMh = self.compute_gradients(M,h_cv,h_fp,idim)
        return 0.5*dMh*h_fp[cut(1,-1,idim)] 
    
def MUSCL_fluxes(self: Simulator,
                 F: dict,
                 dt: float,
                 prims=True)->None:
    """
    Returns the MUSCL scheme fluxes for conserved variales

    Parameters
    ---------- 
        self:   Simulator object
        F:      Dictionary with references to Flux array
                F = {x: Fx, y: Fy, z: Fz}
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    
    Overwrites
    ----------
        F:      Fluxes given by the Riemann solver
    """
    for dim in self.dims:
        idim=self.dims[dim]
        reconstruct_faces(self, dim, idim)
        self.solve_riemann_problem(dim,F[dim],prims)
    
def compute_prediction(W: np.ndarray,
                       dWs: np.ndarray,
                       dtW: np.ndarray,
                       vels: np.array,
                       ndim: int,
                       gamma: float,
                       _d_: int,
                       _p_: int,
                       WB: bool,
                       npassive: int = 0,
                       )->None:
    """
    Returns the prediction for conserved variales

    Parameters
    ---------- 
        W:      Solution vector with primitive variables
        dWs:    Solution vector with slopes 
        vels:   vels:   array containing the indices of velocity components [vx,vy,vz]
                in the Solution array. The size of this array has to match the
                number of dimensions
        ndim:   Number of dimensions
        gamma:  Adiabatic index (ratio of specific heats)
        _d_:    Index of density in the Solution array
        _p_:    Index of pressure/energy in the Solution array
        WB:     Wheter to use Well-balanced scheme or not
    Overwrites
    ----------
        dtW:  Solution vector with predictions 
    """
    dtW[...] = 0
    for idim in range(ndim):
        vel = vels[idim]
        dW = dWs[idim]
        dtW[_d_] -= (W[vel]*dW[_d_] +       W[_d_]*dW[vel])
        dtW[_p_] -= (W[vel]*dW[_p_] + gamma*W[_p_]*dW[vel])
        dtW[vel] -= (W[vel]*dW[vel] + dW[_p_]/W[_d_])
        for vel2 in np.roll(vels,-idim)[1:]:
            dtW[vel2] -= W[vel]*dW[vel2]
        if npassive>0:
            _ps_ = _p_+1
            dtW[_ps_:_ps_+npassive] -= W[vel]*dW[_ps_:_ps_+npassive]
        if WB:
            dW = dWs[idim+ndim]
            dtW[_d_] -= (W[vel]*dW[_d_]) 
            dtW[_p_] -= (W[vel]*dW[_p_])

def compute_prediction_mhd(W: np.ndarray,
                           dWs: np.ndarray,
                           dtW: np.ndarray,
                           vels: np.array,
                           ndim: int,
                           gamma: float,
                           _d_: int,
                           _p_: int,
                           b: list,
                           WB: bool,
                           npassive: int = 0,
                           )->None:
    """
    Hancock half-step prediction (dW/dt) for ideal MHD in primitive
    variables, following the RAMSES trace2d/trace3d source terms
    (mhd/umuscl.f90).  Per sweep direction n with velocity v_n and normal
    field B_n (t are the two transverse components):

        drho/dt -= v_n drho + rho dv_n
        dp/dt   -= v_n dp   + gamma p dv_n
        dv_n/dt -= v_n dv_n + (dp + sum_t B_t dB_t)/rho
        dv_t/dt -= v_n dv_t - B_n dB_t / rho          (magnetic tension)
        dB_t/dt -= v_n dB_t + B_t dv_n - B_n dv_t     (induction)

    The B_n dB_n pressure/tension pair cancels in the normal momentum
    equation, and B_n has no source from its own sweep (it is constant in
    the 1D subsystem); the v(div B) terms are dropped as in RAMSES.

    Parameters
    ----------
        b:  indices of the magnetic field components [Bx, By, Bz] in W.
    """
    dtW[...] = 0
    rho = W[_d_]
    for idim in range(ndim):
        vel = vels[idim]
        bn = b[idim]
        dW = dWs[idim]
        others = [k for k in range(3) if k != idim]
        dtW[_d_] -= W[vel]*dW[_d_] + rho*dW[vel]
        dtW[_p_] -= W[vel]*dW[_p_] + gamma*W[_p_]*dW[vel]
        dptot = dW[_p_] + sum(W[b[k]]*dW[b[k]] for k in others)
        dtW[vel] -= W[vel]*dW[vel] + dptot/rho
        for k in others:
            vt = vels[k]
            bt = b[k]
            dtW[vt] -= W[vel]*dW[vt] - W[bn]*dW[bt]/rho
            dtW[bt] -= W[vel]*dW[bt] + W[bt]*dW[vel] - W[bn]*dW[vt]
        if npassive > 0:
            _ps_ = _p_+1
            dtW[_ps_:_ps_+npassive] -= W[vel]*dW[_ps_:_ps_+npassive]

def MUSCL_Hancock_fluxes(self: Simulator,
                         F: dict,
                         dt: float,
                         prims=True)->None:
    """
    Parameters
    ---------- 
        self:   Simulator object
        F:      Dictionary with references to Flux array
                F = {x: Fx, y: Fy, z: Fz}
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    
    Overwrites
    ----------
        F:      Fluxes given by the Riemann solver
    """
    dMhs={}
    S={}
    # Predictor arrays (dtM and directional gradients) live on the one-cell
    # stripped stencil used by MUSCL-Hancock, independent of simulator Nghc.
    # Using Nghc here over-crops transverse axes (e.g. 2D: 64x66 vs 66x66).
    crop = lambda start, end, idim: crop_fv(start, end, idim, self.ndim, 1)
    for dim in self.dims:
        idim=self.dims[dim]
        dMh = self.compute_gradients(self.dm.M,idim)
        #Compute and store slopes in a dictionary
        S[idim] = 0.5*dMh*self.h_fp[dim][cut(1,-1,idim)]
        #Store gradients in a dictionary
        dMhs[idim] = dMh[crop(None,None,idim)]
        if self.WB:
            dMhs[idim+self.ndim] = self.compute_gradients(self.dm.M_eq,idim)[crop(None,None,idim)]
    if self.WB:
        self.dm.M += self.dm.M_eq                    
    self.compute_prediction(self.dm.M[crop(1,-1,0)],dMhs)
    if self.WB:
        if self.potential:
            drho = ((self.dm.M[0]-self.dm.M_eq[0])/self.dm.M[0])[crop(1,-1,0)]
            for vel in self.vels[:self.ndim]:
                self.dm.dtM[vel][crop(1,-1,0)] += drho[crop(1,-1,0)]*self.dm.grad_phi[vel-1]
        #We move back to the perturbation
        self.dm.M -= self.dm.M_eq
    self.dm.M[crop(1,-1,0)] += 0.5*self.dm.dtM*dt
    
    for dim in self.dims:
        idim=self.dims[dim]
        self.MR_fp[dim][...] = self.interpolate_R(self.dm.M,S[idim],idim)
        self.ML_fp[dim][...] = self.interpolate_L(self.dm.M,S[idim],idim)
        enforce_face_positivity(self, dim, idim)
        self.solve_riemann_problem(dim,F[dim],prims)

