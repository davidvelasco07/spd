try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
import numpy as np
from slicing import indices2,cuts

class CommHelper():
    def __init__(self,ndim,CUDA_AWARE=True):
        self.ndim = ndim
        self.CUDA_AWARE = CUDA_AWARE
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0
        
        self.nx = int(self.size**(1./ndim))
        while (self.size%self.nx != 0):
            self.nx+=1
            
        if ndim>1:
            self.ny = int((self.size/self.nx)**(1./(ndim-1)))
            while (int((self.size/self.nx)%self.ny) != 0):
                self.ny+=1
        else:
            self.ny=1

        self.nz = int(self.size/self.nx/self.ny)
    
        self.x = int(self.rank%self.nx)
        self.y = int((self.rank/self.nx)%self.ny)
        self.z = int((self.rank/self.nx/self.ny))

        mesh = np.arange(self.size).reshape([self.nz,self.ny,self.nx])
        self.left={}
        self.right={}
        for dim in range(ndim):
            self.left[dim]  = np.roll(mesh,+1,axis=2-dim)[self.z,self.y,self.x]
            self.right[dim] = np.roll(mesh,-1,axis=2-dim)[self.z,self.y,self.x]
        #print(f"{self.rank} left {self.left} right {self.right}")
        self.Comms_fv = self.Comms(cuts)
        self.Comms_sd = self.Comms(indices2)
    
    def send_recv_replace(self,buffer,neighbour,side):
        self.comm.Sendrecv_replace(buffer,neighbour,sendtag=side,source=neighbour,recvtag=1-side)
        
    def reduce_min(self, M):
        if self.size>1:
            return self.comm.allreduce(M,op=MPI.MIN)
        else:
            return M
        
    def Comms(self, function): 
        ndim = self.ndim
        def communicate(
             dm,
             M: np.ndarray,
             BC: dict,
             idim: int,
             dim: str,
             ngh: int=0):
            rank_dim = self.__getattribute__(dim) 

            Buffers={}
            for side in [0,1]:
                Buffers[side] = M[function(-side,ndim,idim,ngh=ngh)].flatten()
    
            neighbour = self.right[idim] if rank_dim%2 else self.left[idim]
            side = rank_dim%2
            self.send_recv(dm, neighbour, BC[dim][side], Buffers[side], side)
        
            neighbour = self.left[idim] if rank_dim%2 else self.right[idim]
            side = 1-rank_dim%2
            self.send_recv(dm, neighbour, BC[dim][side], Buffers[side], side)

            self.barrier()    
        return communicate

    def send_recv(self, dm, neighbour, BC, Buffer, side):
        rank = self.rank
        if neighbour != rank:
            if not(self.CUDA_AWARE):
                Buffer = dm.xp.asnumpy(Buffer)
            self.send_recv_replace(Buffer,neighbour,side)
            Buffer = Buffer.reshape(BC.shape)
            if not(self.CUDA_AWARE):
                Buffer = dm.xp.asarray(Buffer) 
            BC[...] = Buffer
            
    def barrier(self):
        if self.size>1:
            self.comm.Barrier()