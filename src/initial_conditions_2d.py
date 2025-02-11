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