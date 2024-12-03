import matplotlib.pyplot as plt
import numpy as np
from simulator import Simulator
from typing import Tuple

def plot_fields(s: Simulator,
                M: np.ndarray,
                figsize: Tuple = (6,4),
                **kwargs):
    
    fig,axs = plt.subplots(1,s.nvar,figsize=(figsize[0]*s.nvar,figsize[1]))
    for var in range(s.nvar):
        plt.sca(axs[var])
        plot_field(s,M,var,**kwargs)

def plot_field(s: Simulator,
                M: np.ndarray,
                var: int,
                dim: str = "z",
                transpose: bool = True,
                regular: bool = False,
                integrate=False,
                **kwargs):
    if regular:
        M = s.regular_mesh(M)
    if transpose:
        M=s.transpose_to_fv(M)
    if s.ndim==2:
        x,y = s.regular_faces()
        plt.pcolormesh(x,y,M[var],**kwargs)
        plt.colorbar()
    elif s.ndim==3:
        x,y,z = s.regular_faces()
        if dim=="z":
            axis=0
        else:
            axis=1
            y=z
        if integrate:
            M=M[var].sum(axis=axis)/M.shape[axis+1]
        else:
            shape = M.shape[axis+1]//2
            M=M[var,shape] if dim=="z" else M[var,:,shape]
        plt.pcolormesh(x,y,M,**kwargs)
        plt.colorbar()
    else:
        x = s.regular_centers()[0]
        plt.plot(x,M[var])
    plt.title(s.variables[var],**kwargs)