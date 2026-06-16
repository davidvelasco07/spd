import numpy as np
from . import mhd

def llf(W_L: np.ndarray,
        W_R: np.ndarray,
        U_L: np.ndarray,
        U_R: np.ndarray,
        F,
        vels: np.array,
        _p_: int,
        gamma: float,
        min_c2: float,
        **kwargs,
    ) -> np.ndarray:
        #Density index
        _d_=0 
        v_1 = vels[0]
        _b_ = _p_ + kwargs["thdiffusion"] + kwargs["npassive"]
        
        B1_L = W_L[v_1+_b_] 
        B1_R = W_R[v_1+_b_]
        B2_L = W_L[vels[1]+_b_] 
        B2_R = W_R[vels[1]+_b_]
        B3_L = W_L[vels[2]+_b_] 
        B3_R = W_R[vels[2]+_b_]

        c_L = mhd.compute_fast_vel(W_L[_p_],W_L[0], B1_L, B2_L, B3_L,gamma=gamma, min_c2=min_c2)
        c_R = mhd.compute_fast_vel(W_R[_p_],W_R[0], B1_R, B2_R, B3_R,gamma=gamma, min_c2=min_c2)
        c_max = np.maximum(abs(W_L[v_1]),abs(W_R[v_1])) + np.maximum(c_L,c_R)
    
        F_L = mhd.compute_fluxes(W_L,vels,_p_,gamma,**kwargs)
        F_R = mhd.compute_fluxes(W_R,vels,_p_,gamma,**kwargs)

        F = 0.5*(F_R+F_L)-0.5*c_max[np.newaxis]*(U_R-U_L)
        return F