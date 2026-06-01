import numpy as np
import initial_conditions_1d as ic1d
import initial_conditions_2d as ic2d

def step_function(xyz: np.ndarray,case: int, vx=1, vy=1, vz=1, P=1):
    if xyz.shape[0]==1:
        return ic1d.step_function(xyz,case,vx=vx,P=P)
    if xyz.shape[0]==2:
        return ic2d.step_function(xyz,case,vx=vx,vy=vy,P=P)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,
                        np.where(np.fabs(y-0.5)<0.25,
                                 np.where(np.fabs(z-0.5)<0.25,2,1),1),1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #vz
        return vz*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)

def sine_wave(xyz: np.ndarray,case: int, A=0.125, vx=1, vy=1, vz=1, P=1):
    if xyz.shape[0]==1:
        return ic1d.sine_wave(xyz,case,vx=vx,P=P)
    if xyz.shape[0]==2:
        return ic2d.sine_wave(xyz,case,vx=vx,vy=vy,P=P)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*(x+y+z)))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #vz
        return vz*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)