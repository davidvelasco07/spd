import numpy as np

def compute_cs2(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float)->np.ndarray:
    """
    Returns the square of the sound speed, with minimum value min_c2

    Parameters
    ----------
    P:      Array of Pressure values
    rho:    Array of density values
    gamma:  Adiabatic index
    min_c2: Minimum value allowed for the square of the sound speed

    Returns
    -------
    Cs^2: Array with Sound-speed square values
    """
    c2 = gamma*P/rho
    #np.maximum propagates NaNs, so we use np.where
    c2 = np.where(c2>min_c2,c2,min_c2)
    return c2

def compute_cs(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float)->np.ndarray:
    """
    Returns the sound speed, with minimum value sqrt(min_c2)

    Parameters
    ----------
    P:      Array of Pressure values
    rho:    Array of density values
    gamma:  Adiabatic index
    min_c2: Minimum value allowed for the square of the sound speed

    Returns
    -------
    Cs: Array with Sound-speed values
    """
    return np.sqrt(compute_cs2(P,rho,gamma,min_c2))

def compute_primitives(
        U: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        _t_=None,
        W=None,
        isothermal: bool = False,
        thdiffusion: bool = False,
        npassive=0)->np.ndarray:
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
    _d_:    index of density in the Solution array
    W:      None or array of similar shape to U
    
    Returns
    -------
    W: Solution array of primitive variables

    """
    if type(W)==type(None):
        W = U.copy()
    assert W.shape == U.shape
    K = W[_p_]
    K *= 0
    for vel in vels:
        W[vel] = U[vel]/U[_d_]
        K += W[vel]**2
    K *= 0.5*U[0]
    K *= 0 if isothermal else 1
    W[_p_] = (gamma-1)*(U[_p_]-K)
    if thdiffusion:
        W[_t_] = W[_p_]/W[_d_]
    if npassive>0:
        _ps_ = _p_+1
        W[_ps_:_ps_+npassive] = U[_ps_:_ps_+npassive]/U[_d_]
    return W
                
def compute_conservatives(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        _t_ = None,
        U=None,
        isothermal: bool = False,
        thdiffusion: bool = False,
        npassive=0)->np.ndarray:
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
    _d_:    index of density in the Solution array
    U:      None or array of similar shape to W
    
    Returns
    -------
    U:      Solution array of conseved variables
    """
    if type(U)==type(None):
        U = W.copy()
    assert U.shape == W.shape
    K = U[_p_]
    K *= 0
    for vel in vels:
        U[vel] = W[vel]*U[_d_]
        K += W[vel]**2
    K *= 0.5*U[_d_]
    K *= 0 if isothermal else 1
    U[_p_] = W[_p_]/(gamma-1)+K
    if thdiffusion:
        U[_t_] = W[_p_]/W[_d_]
    if npassive>0:
        _ps_ = _p_+1
        U[_ps_:_ps_+npassive] = W[_ps_:_ps_+npassive]*U[_d_]
    return U

def compute_fluxes(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        _d_: int = 0,
        F=None,
        isothermal: bool = False,
        npassive=0)->np.ndarray:
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
    K *= 0
    v1=vels[0]
    for v in vels[::-1]:
        #Iterate over inverted array of vels
        #so that the last value of m correspond to the 
        #normal component
        m = W[_d_]*W[v]
        K += m*W[v]
        F[v,...] = m*W[v1]
        
    E = W[_p_]/(gamma-1) + 0.5*K
    F[0  ,...] = m
    F[v1,...] = m*W[v1] + W[_p_]
    F[_p_,...] = W[v1]*(E + W[_p_])
    F[_p_,...] *= 0 if isothermal else 1
    if npassive>0:
        _ps_ = _p_+1
        F[_ps_:_ps_+npassive,...] = m*W[_ps_:_ps_+npassive]
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
        npassive=0)->np.ndarray:
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
    if type(F)==type(None):
        F = W.copy()
    F[...] = 0
    #index of normal component
    v1  = vels[0]
    #Gradient in normal dimension
    dW1 = dWs[v1-1]
    #Flux is normal dimension
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
    return F*W[_d_]*nu

def compute_thermal_fluxes(
        W:  np.ndarray,
        dW: np.ndarray,
        chi: float,
        _t_: int,
        _d_: int=0)->np.ndarray:
    return chi*W[_d_]*dW[_t_]