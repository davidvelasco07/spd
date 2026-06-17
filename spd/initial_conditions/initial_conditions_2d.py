import numpy as np

def step_function(xy: np.ndarray,case: int, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,
                        np.where(np.fabs(y-0.5)<0.25,2,1),1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, A=0.125, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*(x+y)))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)

def RTI(
    xy: np.ndarray,
    case: int,
    P0=1.0,
    gamma=5 / 3,
    g=-1.0,
    rho1=2.0,
    rho2=1.0,
    yc=0.5,
) -> np.ndarray:
    """Rayleigh-Taylor instability (single-mode, hydrostatic background).

    A heavy fluid ``rho1`` sits on top of a light fluid ``rho2`` at the
    interface ``y = yc`` in a constant downward gravitational field, with a
    small single-mode vertical velocity perturbation seeded at the interface.

    The gravitational potential is returned for ``case == -1`` so the same
    function can be passed as ``init_fct`` with ``potential=True``.
    """
    x = xy[0]
    y = xy[1]
    if case == 0:
        # density
        return np.where(y > yc, rho2, rho1)
    elif case == 1:
        # vx
        return np.zeros(x.shape)
    elif case == 2:
        # vy: single-mode perturbation scaled by the local sound speed
        dv = np.sqrt(gamma * (P0 + rho1 * yc - P0 + 1) / rho1)
        return -0.025 * dv * np.cos(8 * np.pi * x)
    elif case == 4:
        # Pressure (hydrostatic equilibrium: dP/dy = rho * g, with g = -1)
        return np.where(
            y > yc, P0 + rho2 * y + (rho1 - rho2) * yc, P0 + rho1 * y
        )
    elif case == -1:
        # Gravitational potential phi (acceleration g_y = -dphi/dy = g)
        return g * y
    else:
        return np.zeros(x.shape)


def KH_instability(xy: np.ndarray, case: int) -> np.ndarray:
    y=xy[1]
    w0=0.1
    sigma = 0.05/np.sqrt(2)
    if case==0:
        return np.where(y<0.25,1,np.where(y<0.75,2,1))
    elif case==1:
        return np.where(y<0.25,-0.5,np.where(y<0.75,0.5,-0.5))
    elif case==2:
        return w0*np.sin(4*np.pi*xy[0])*(np.exp(-(y-0.25)**2/(2*sigma**2))+np.exp(-(y-0.75)**2/(2*sigma**2)))
    elif case==4:
        #Pressure
        return 2.5*np.ones(y.shape)
    else:
        return np.zeros(xy[0].shape)
