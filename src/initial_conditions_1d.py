import numpy as np

def step_function(xy: np.ndarray,case: int, vx=1, P=1):
    x=xy[0]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,2,1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, A=0.125, vx=1, P=1):
    x=xy[0]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*x))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sod_shock_tube(x: np.ndarray, case: int) -> np.ndarray:
    x=x[0]
    if case==0:
        #density
        return np.where(x < 0.5, 1., 0.125)
    elif case==2:
        #pressure
        return np.where(x < 0.5, 1., 0.1)
    else:
        #velocities
        return np.zeros(x.shape)