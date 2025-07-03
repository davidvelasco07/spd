import numpy as np
import hydro

def compute_fast_vel(p,rho,Bn,Bt1,Bt2,gamma,min_c2):
    B2 = Bn**2+Bt1**2+Bt2**2
    c2 = hydro.compute_cs2(p,rho,gamma,min_c2)
    d2 = 0.5*(B2/rho+c2)
    return np.sqrt(d2 + np.sqrt(d2**2-c2*Bn**2/rho))

def compute_primitives(
        U: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        **kwargs)->np.ndarray:
    """
    Transforms array of conservative to primitive variables

    Parameters
    ----------

    U:      Solution array of conseved variables
    vels:   array containing the indices of velocity components [vx,vy,vz]
            in the Solution array. The size of this array has to match the
            number of dimensions
    _p_:    index of pressure/energy in the Solution array
    gamma:  Adiabatic index
    
    Returns
    -------
    W: Solution array of primitive variables

    """
    shift = _p_+kwargs["thdiffusion"]+kwargs["npassive"]
    if not("W" in kwargs):
        W = U.copy()
    W = hydro.compute_primitives(U,vels,_p_,gamma,**kwargs)
    for vel in vels:
        W[_p_] -= (gamma-1)*0.5*W[vel+shift]**2
    return W
                
def compute_conservatives(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        U=None,
        **kwargs)->np.ndarray:
    """
    Transforms array of primitive to conservative variables

    Parameters
    ----------

    W:      Solution array of primitive variables
    vels:   array containing the indices of velocity components [vx,vy,vz]
            in the Solution array. The size of this array has to match the
            number of dimensions
    _p_:    index of pressure/energy in the Solution array
    gamma:  Adiabatic index
    
    Returns
    -------
    U:      Solution array of conseved variables
    """
    shift = _p_+kwargs["thdiffusion"]+kwargs["npassive"]
    if not("U" in kwargs):
        U = W.copy()
    U = hydro.compute_conservatives(W,vels,_p_,gamma,**kwargs)
    for vel in vels:
        U[_p_] += 0.5*W[vel+shift]**2
    return U

def compute_fluxes(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        F=None,
        thdiffusion: bool = False,
        npassive=0,
        **kwargs)->np.ndarray:
    """
    Returns array of conservative fluxes

    Parameters
    ----------
    W:      Solution array of primitive variables
    vels:   array containing the indices of velocity components [vx,vy,vz]
            in the Solution array. The size of this array has to match the
            number of dimensions
    _p_:    index of pressure/energy in the Solution array
    gamma:  Adiabatic index
    _d_:    index of density in the Solution array
    F:      None or array of similar shape to W

    Returns
    -------
    F: Solution array of fluxes for the conserved variables
    """
    if type(F)==type(None):
        F = W.copy()
    K = F[_p_]
    K[...] = 0
    v1=vels[0]
    shift = _p_+thdiffusion+npassive
    B2 = F[shift+1]
    B2[...] = 0
    Bv = F[shift+2]
    Bv[...]=0
    B_1 = W[v1+shift]

    for v in vels[::-1]:
        #Iterate over inverted array of vels
        #so that the last value of m correspond to the 
        #normal component
        m = W[_d_]*W[v]
        K += m*W[v]
        B2 += W[v+shift]**2
        F[v,...] = m*W[v1] - B_1*W[v+shift]
        Bv += W[v+shift] * W[v]

    pT = W[_p_] + 0.5*B2  # Total pressure
    E = W[_p_]/(gamma-1) + 0.5*(K+B2)
    F[0  ,...] = m
    F[v1,...] = m*W[v1] + pT - B_1**2
    F[_p_,...] = W[v1]*(E + pT) -B_1*Bv
    if npassive>0:
        _ps_ = _p_+1
        F[_ps_:_ps_+npassive,...] = m*W[_ps_:_ps_+npassive]
    for v in vels:
        F[v+shift,...] = W[v1]*W[v+shift] - W[v]*B_1
    return F

def compute_viscous_fluxes(
        W: np.ndarray,
        vels: np.array,
        dWs: dict,
        _p_: int,
        nu: float,
        beta: float,
        _d_: int=0,
        F=None,
        thdiffusion: bool = False,
        npassive=0,
        **kwargs)->np.ndarray:
    """
    Returns array of viscous fluxes for conservative variables

    Parameters
    ----------
    W:      Solution array of primitive variables
    vels:   array containing the indices of velocity components [vx,vy,vz]
            in the Solution array. The size of this array has to match the
            number of dimensions
    dWs:    Dictionary with references to arrays containing the gradient of
            the primitive variables along a given dimension
    _p_:    index of pressure/energy in the Solution array
    nu:     Viscous coefficient
    beta:   ------
    _d_:    index of density in the Solution array
    F:      None or array of similar shape to W

    Returns
    -------
    F: Solution array of fluxes for the conserved variables

    Notes
    -----
    It assumes the dictionary of gradients has keys
    dUs  = {vx-1: dUx, vy-1: dUy, vz-1: dUz}

    Examples
    --------
    indices of velocities in the Solution array:
    vx = 1, vy = 2, vz = 3

    1D:
    vels = [1]
    dUs  = {0: dUx}

    2D:
    vels = [1,2] or [2,1]
    dUs  = {0: dUx, 1: dUy}

    3D:
    vels = [1,2,3], [2,3,1] or [3,2,1]
    dUs  = {0: dUx, 1: dUy, 2: dUz}
    """
    _b_ = _p_+thdiffusion+npassive+1
    if type(F)==type(None):
        F = W.copy()
    F[...] = 0
    #index of normal component
    v1  = vels[0]
    #Gradient in normal dimension
    dW1 = dWs[v1-1]
    #Flux in normal dimension
    F[v1] = 2*dW1[v1] - beta*dW1[v1]
    #Energy flux
    F[_p_] = W[v1]*F[v1]
    for vel in vels[1:]:
        idim = vel-1
        dW = dWs[idim]
        F[v1]  -= beta*dW[vel]
        F[vel]  = (dW1[vel]+dW[v1])
        F[_p_] += W[vel]*F[vel]
    if npassive>0:
        _ps_ = _p_+1
        F[_ps_:_ps_+npassive] = dW1[_ps_:_ps_+npassive]
    F*=W[_d_]*nu
    dim = vels[0]-1
    i,j,k = np.roll(np.arange(3),dim)
    # nu * B x J  
    # X: ByJz - BzJy
    # Y: BzJx - BxJz
    # Z: BxJy - ByJx
    for dim2 in [j,k]:
        J=0
        if dim in dWs:
            J += dWs[dim ][_b_+dim2]
        if dim2 in dWs:
            J -= dWs[dim2][_b_+dim ]
        F[_p_] += nu*J*W[_b_+dim2]
    return F