import numpy as np
from spd import hydro
from spd.hydro import riemann_solver as rs
from spd.MHD import mhd
from spd.MHD import riemann_solver as mrs    

class Riemann_solver_1D:
    def __init__(self,name, soe):
        self.name = name
        self.soe = soe
        self.equations = hydro if soe == "hydro" else mhd
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
                U_L = self.equations.compute_conservatives(W_L,vels,_p_,gamma,**kwargs)
                U_R = self.equations.compute_conservatives(W_R,vels,_p_,gamma,**kwargs)
            else:
                U_L = M_L
                U_R = M_R
                # min_c2 is forwarded so the MHD conversion can apply its
                # ctoprim-style floors (hydro absorbs it in **kwargs).
                W_L = self.equations.compute_primitives(U_L,vels,_p_,gamma,min_c2=min_c2,**kwargs)
                W_R = self.equations.compute_primitives(U_R,vels,_p_,gamma,min_c2=min_c2,**kwargs)
            return solver(W_L,W_R,U_L,U_R,F,vels,_p_,gamma,min_c2,**kwargs)
        return solve

    def llf(self,*args,**kwargs):
        if self.soe == "hydro":
            return rs.llf(*args,**kwargs)
        elif self.soe == "mhd":
            return mrs.llf(*args,**kwargs)

    def llf_mhd(self,*args,**kwargs):
        # Backward-compatible name from the monolithic code; the MHD branch
        # is normally selected through soe="mhd" with name "llf".
        return mrs.llf(*args,**kwargs)

    def hllc(self,*args,**kwargs):
        if self.soe == "hydro":
            return rs.hllc(*args,**kwargs)
        elif self.soe == "mhd":
            raise NotImplementedError("HLLC is not implemented for MHD; use llf or hlld")

    def hlld(self,*args,**kwargs):
        if self.soe == "mhd":
            return mrs.hlld(*args,**kwargs)
        raise NotImplementedError("HLLD is an MHD solver; use hllc for hydro")

    def lhllc(self,*args,**kwargs):
        return rs.lhllc(*args,**kwargs)

