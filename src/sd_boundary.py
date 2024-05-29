import numpy as np

def store_interfaces(self,M,dim) -> None:
    if dim=="x":
        if self.Z:
            #nvar,nader,Nz,Ny,Nx+1,p+1,p+1
            self.dm.MR_fp_x[...,:-1,:,:] = M[..., 0]
            self.dm.ML_fp_x[...,1: ,:,:] = M[...,-1]
        elif self.Y:
            #nvar,nader,Ny,Nx+1,p+1
            self.dm.MR_fp_x[...,:-1,:] = M[..., 0]
            self.dm.ML_fp_x[...,1: ,:] = M[...,-1]
        else:
            #nvar,nader,Nx+1
            self.dm.MR_fp_x[...,:-1] = M[..., 0]
            self.dm.ML_fp_x[...,1: ] = M[...,-1]
    elif dim=="y":
        if self.Z:
            #nvar,nader,Nz,Ny+1,Nx,p+1,p+1
            self.dm.MR_fp_y[...,:-1,:,:,:] = M[..., 0,:]
            self.dm.ML_fp_y[...,1: ,:,:,:] = M[...,-1,:]
        else:
            #nvar,nader,Ny+1,Nx,p+1
            self.dm.MR_fp_y[...,:-1,:,:] = M[..., 0,:]
            self.dm.ML_fp_y[...,1: ,:,:] = M[...,-1,:]
    elif dim=="z":
        #nvar,nader,Nz+1,Ny,Nx,p+1,p+1
        self.dm.MR_fp_z[...,:-1,:,:,:,:] = M[..., 0,:,:]
        self.dm.ML_fp_z[...,1: ,:,:,:,:] = M[...,-1,:,:]

def apply_interfaces(self,M,dim):
    if dim=="x":
        #nvar,nader,Nz,Ny,Nx+1,p+1,p+1
        if self.Z:
            M[..., 0] = self.dm.MR_fp_x[...,:-1,:,:]
            M[...,-1] = self.dm.ML_fp_x[...,1: ,:,:]
        elif self.Y:
            M[..., 0] = self.dm.MR_fp_x[...,:-1,:]
            M[...,-1] = self.dm.ML_fp_x[...,1: ,:]
        else:
            M[..., 0] = self.dm.MR_fp_x[...,:-1]
            M[...,-1] = self.dm.ML_fp_x[...,1: ]
    elif dim=="y":
        #nvar,nader,Nz,Ny+1,Nx,p+1,p+1
        if self.Z:
            M[..., 0,:] = self.dm.MR_fp_y[...,:-1,:,:,:]
            M[...,-1,:] = self.dm.ML_fp_y[...,1: ,:,:,:]
        else:
            M[..., 0,:] = self.dm.MR_fp_y[...,:-1,:,:]
            M[...,-1,:] = self.dm.ML_fp_y[...,1: ,:,:]
    elif dim=="z":
        #nvar,nader,Nz+1,Ny,Nx,p+1,p+1
        M[..., 0,:,:] = self.dm.MR_fp_z[...,:-1,:,:,:,:]
        M[...,-1,:,:] = self.dm.ML_fp_z[...,1: ,:,:,:,:]


def store_BC(self,BC_array,M,dim) -> None:
    #We store the intercell face/corner values to be used when performing
    #the Riemann Solver
    na=np.newaxis
    if self.BC[dim] == "periodic":
        BC_array[0] = slice_array(M,dim,-1,self.ndim)
        BC_array[1] = slice_array(M,dim, 0,self.ndim)
    else:
        raise("Undetermined boundary type")
                         
def apply_BC(self,dim) -> None:
    #Here we fill up the missing first column of U_L
    #and the missing last column of U_R
    if dim=="x":
        #nvar,nader,Nz,Ny,Nx+1,p+1,p+1
        if self.Z:
            self.dm.ML_fp_x[..., 0,:,:] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1,:,:] = self.dm.BC_fp_x[1]
        elif self.Y:
            self.dm.ML_fp_x[..., 0,:] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1,:] = self.dm.BC_fp_x[1]
        else:
            self.dm.ML_fp_x[..., 0] = self.dm.BC_fp_x[0]
            self.dm.MR_fp_x[...,-1] = self.dm.BC_fp_x[1]
    elif dim=="y":
        #nvar,nader,Nz,Ny+1,Nx,p+1,p+1
        if self.Z:
            self.dm.ML_fp_y[..., 0,:,:,:] = self.dm.BC_fp_y[0]
            self.dm.MR_fp_y[...,-1,:,:,:] = self.dm.BC_fp_y[1]
        else:
            self.dm.ML_fp_y[..., 0,:,:] = self.dm.BC_fp_y[0]
            self.dm.MR_fp_y[...,-1,:,:] = self.dm.BC_fp_y[1]
    elif dim=="z":
        #nvar,nader,Nz+1,Ny,Nx,p+1,p+1
        self.dm.ML_fp_z[..., 0,:,:,:,:] = self.dm.BC_fp_z[0]
        self.dm.MR_fp_z[...,-1,:,:,:,:] = self.dm.BC_fp_z[1]

def slice_array(M,dim,idx,ndim):
    if ndim==1:
        if dim=="x":
            return M[...,idx,idx]
        else:
            raise("Incorrect dimension")
    elif ndim==2:
        if dim=="x":
            return M[...,idx,:,idx]
        elif dim=="y":
            return M[...,idx,:,idx,:]
        else:
            raise("Incorrect dimension")
    elif ndim==3:
        if dim=="x":
            return M[...,idx,:,:,idx]
        elif dim=="y":
            return M[...,idx,:,:,idx,:]
        elif dim=="z":
            return M[...,idx,:,:,idx,:,:]
        else:
            raise("Incorrect dimension")