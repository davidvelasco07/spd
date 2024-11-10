import numpy as np
import hydro

class Riemann_solver:
    def __init__(self,name):
        self.name = name
        self.solver = self.riemann_solver(self.__getattribute__(name))

    def riemann_solver(self,solver):
        def solve(M_L: np.ndarray,
                  M_R: np.ndarray,
                  F,
                  vels: np.array,
                  _p_: int, 
                  gamma: float,
                  min_c2: float,
                  prims: bool,
                  **kwargs,):
            if prims:
                W_L = M_L
                W_R = M_R
                U_L = hydro.compute_conservatives(W_L,vels,_p_,gamma,**kwargs)
                U_R = hydro.compute_conservatives(W_R,vels,_p_,gamma,**kwargs)
            else:
                U_L = M_L
                U_R = M_R
                W_L = hydro.compute_primitives(U_L,vels,_p_,gamma,**kwargs)
                W_R = hydro.compute_primitives(U_R,vels,_p_,gamma,**kwargs)
            return solver(W_L,W_R,U_L,U_R,F,vels,_p_,gamma,min_c2,**kwargs)
        return solve

    def llf(
        self,
        W_L: np.ndarray,
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
        """
        LLF Riemann Solver
        W_L/W_R: Primitive variables
        U_L/U/R: Conservative variables
        vels: arrays of velocity indices
        _p_: pressure/energy index
        gamma: adiabatic index (ratio of specific heats)
        min_c2: minimum value allowed for the square of the sound speed
        Output:
        F: Fluxes
        """
        #Density index
        _d_=0 
        v_1 = vels[0]
        F_L = hydro.compute_fluxes(W_L,vels,_p_,gamma,**kwargs)
        F_R = hydro.compute_fluxes(W_R,vels,_p_,gamma,**kwargs)

        c_L = hydro.compute_cs(W_L[_p_],W_L[_d_],gamma,min_c2) + np.abs(W_L[v_1])
        c_R = hydro.compute_cs(W_R[_p_],W_R[_d_],gamma,min_c2) + np.abs(W_R[v_1])

        c_max = np.where(c_L>c_R,c_L,c_R)[np.newaxis,...]
        F = 0.5*(F_R+F_L)-0.5*c_max*(U_R-U_L)
        return F

    def hllc(
        self,
        W_L: np.ndarray,
        W_R: np.ndarray,
        U_L: np.ndarray,
        U_R: np.ndarray,
        F,
        vels: np.array,
        _p_: int,
        gamma: float,
        min_c2: float,
        npassive: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        HLLC Riemann Solver
        W_L/W_R: Primitive variables
        U_L/U/R: Conservative variables
        vels: arrays of velocity indices
        _p_: pressure/energy index
        gamma: adiabatic index (ratio of specific heats)
        min_c2: minimum value allowed for the square of the sound speed
        Output:
        F: Fluxes
        """
        #Density index
        _d_ = 0
        v_1 = vels[0]

        c_L = hydro.compute_cs(W_L[_p_],W_L[_d_],gamma,min_c2) + np.abs(W_L[v_1])
        c_R = hydro.compute_cs(W_R[_p_],W_R[_d_],gamma,min_c2) + np.abs(W_R[v_1])
    
        c_max = np.where(c_L>c_R,c_L,c_R)

        v_L = W_L[v_1]
        v_R = W_R[v_1]
        #Compute HLL wave speed
        s_L = np.where(v_L<v_R,v_L,v_R)-c_max
        s_R = np.where(v_L>v_R,v_L,v_R)+c_max

        #Compute lagrangian sound speed
        rc_L = W_L[0]*(v_L-s_L)
        rc_R = W_R[0]*(s_R-v_R)

        #Compute acoustic star state
        v_star = (rc_R*v_R + rc_L*v_L + (W_L[_p_]-W_R[_p_]))/(rc_R+rc_L)
        P_star = (rc_R*W_L[_p_] + rc_L*W_R[_p_] + rc_L*rc_R*(v_L-v_R))/(rc_R+rc_L)

        #Left star region variables
        r_starL = W_L[0]*(s_L-v_L)/(s_L-v_star)
        e_starL =((s_L-v_L)*U_L[_p_]-W_L[_p_]*v_L+P_star*v_star)/(s_L-v_star)

        #Right star region variables
        r_starR = W_R[0]*(s_R-v_R)/(s_R-v_star)
        e_starR = ((s_R-v_R)*U_R[_p_]-W_R[_p_]*v_R+P_star*v_star)/(s_R-v_star)

        # If   s_L>0 -> U_gdv = U_L
        # elif v*>0  -> U_gdv = U*_L
        # eilf s_R>0 -> U_gdv = U*_R
        # else       -> U_gnv = U_R
        r_gdv = np.where(s_L>0,W_L[0],
                        np.where(v_star>0,r_starL,
                                np.where(s_R>0,r_starR,W_R[_d_])))
        v_gdv = np.where(s_L>0,W_L[v_1],
                        np.where(v_star>0,v_star,
                                np.where(s_R>0,v_star,W_R[v_1])))
        P_gdv = np.where(s_L>0,W_L[_p_],
                        np.where(v_star>0,P_star,
                                np.where(s_R>0,P_star,W_R[_p_])))
        e_gdv = np.where(s_L>0,U_L[_p_],
                        np.where(v_star>0,e_starL,
                                np.where(s_R>0,e_starR,U_R[_p_])))

        #F = W_L.copy()             
        F[_d_,...] = r_gdv*v_gdv
        F[v_1,...] = F[_d_]*v_gdv + P_gdv
        F[_p_,...] = v_gdv*(e_gdv + P_gdv)
        F[_p_] *= 0 if kwargs["isothermal"] else 1
        for vel in vels[1:]:
            F[vel,...] = F[_d_]*np.where(v_star>0,W_L[vel],W_R[vel])
        if npassive>0:
            _ps_ = _p_+1
            F[_ps_:_ps_+npassive,...] = F[_d_]*np.where(v_star>0,W_L[_ps_:_ps_+npassive],W_R[_ps_:_ps_+npassive])
        return F

    def lhllc(
        self,
        W_L: np.ndarray,
        W_R: np.ndarray,
        U_L: np.ndarray,
        U_R: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        min_c2: float,
        npassive: int = 0,
        **kwargs,
    ) -> np.ndarray:
        #Density index
        _d_ = 0
        v_1 = vels[0]

        c_L = hydro.compute_cs(W_L[_p_],W_L[_d_],gamma,min_c2) + np.abs(W_L[v_1])
        c_R = hydro.compute_cs(W_R[_p_],W_R[_d_],gamma,min_c2) + np.abs(W_R[v_1])
    
        c_max = np.where(c_L>c_R,c_L,c_R)
    
        v_L = W_L[v_1]
        v_R = W_R[v_1]
        #Compute HLL wave speed
        s_L = np.where(v_L<v_R,v_L,v_R)-c_max
        s_R = np.where(v_L>v_R,v_L,v_R)+c_max

        #Compute lagrangian sound speed
        rc_L = W_L[_d_]*(v_L-s_L)
        rc_R = W_R[_d_]*(s_R-v_R)

        Mach  = np.minimum(1,np.maximum(np.abs(v_L),np.abs(v_R))/c_max)
        phi   = Mach*(2-Mach)
    
        #Compute acoustic star state
        v_star = (rc_R*v_R + rc_L*v_L + (W_L[_p_]-W_R[_p_]))/(rc_R+rc_L)
        P_star = (rc_R*W_L[_p_] + rc_L*W_R[_p_] + phi*rc_L*rc_R*(v_L-v_R))/(rc_R+rc_L)
    
        #Left star region variables
        r_starL = W_L[_d_]*(s_L-v_L)/(s_L-v_star)
        e_starL =((s_L-v_L)*U_L[_p_]-W_L[_p_]*v_L+P_star*v_star)/(s_L-v_star)

        #Right star region variables
        r_starR = W_R[_d_]*(s_R-v_R)/(s_R-v_star)
        e_starR = ((s_R-v_R)*U_R[_p_]-W_R[_p_]*v_R+P_star*v_star)/(s_R-v_star)

        r_gdv = np.where(s_L>0,W_L[_d_],
                    np.where(v_star>0,r_starL,
                            np.where(s_R>0,r_starR,W_R[_d_])))
        v_gdv = np.where(s_L>0,W_L[v_1],
                        np.where(v_star>0,v_star,
                                np.where(s_R>0,v_star,W_R[v_1])))
        P_gdv = np.where(s_L>0,W_L[_p_],
                        np.where(v_star>0,P_star,
                                np.where(s_R>0,P_star,W_R[_p_])))
        e_gdv = np.where(s_L>0,U_L[_p_],
                        np.where(v_star>0,e_starL,
                                np.where(s_R>0,e_starR,U_R[_p_])))
          
        F = W_L.copy()             
        F[_d_,...] = r_gdv*v_gdv
        F[v_1,...] = F[_d_]*v_gdv + P_gdv
        F[_p_,...] = v_gdv*(e_gdv + P_gdv)
        F[_p_] *= 0 if kwargs["isothermal"] else 1
        for vel in vels[1:]:
            F[vel,...] = F[_d_]*np.where(v_star>0,W_L[vel],W_R[vel])
        if npassive>0:
            _ps_ = _p_+1
            F[_ps_:_ps_+npassive,...] = F[_d_]*np.where(v_star>0,W_L[_ps_:_ps_+npassive],W_R[_ps_:_ps_+npassive])
        return F