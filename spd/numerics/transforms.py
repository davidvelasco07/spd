import numpy as np

def compute_A_from_B(B,A_to_B,dim,ndim,ader=False) -> np.ndarray:
        # Axes labels:
        #   u: Conservative variables
        #   z,y,x: cells
        #   k,j,i: B pts
        #   n,m,l: A pts
        y = ("","y") [ndim>1]
        j = ("","j") [ndim>1]
        z = ("","z") [ndim>2]
        k = ("","k") [ndim>2]
        t = ("","t") [ader]
        u = f"u{t}{z}{y}x"
        if dim=="x":
            u += f"{k}{j}"
            A = np.einsum(f"fs,{u}s->{u}f",A_to_B, B)
        elif dim=="y" and ndim>1:
            u += f"{k}"
            A = np.einsum(f"fs,{u}si->{u}fi", A_to_B, B)
        elif dim=="z" and ndim>2:
            A = np.einsum(f"fs,{u}sji->{u}fji", A_to_B, B)
        else:
            raise("Wrong option for dim")
        return A
    
def compute_A_from_B_full(B,A_to_B,ndim) -> np.ndarray:
       # Axes labels:
        #   u: Conservative variables
        #   z,y,x: cells
        #   k,j,i: A
        #   n,m,l: B
        # optimize=True lets numpy/cupy contract the matrices one at a
        # time instead of naively (orders of magnitude faster on CPU).
        if ndim==3:
            A = np.einsum("kn,jm,il,uzyxnml->uzyxkji",
                         A_to_B,
                         A_to_B,
                         A_to_B, B, optimize=True)
        elif ndim==2:
            A = np.einsum("jm,il,uyxml->uyxji",
                         A_to_B,
                         A_to_B, B, optimize=True)
        else:
            A = np.einsum("il,uxl->uxi",
                         A_to_B, B)
        return A

def compute_A_from_B_full_fv(B,A_to_B,ndim) -> np.ndarray:
        """Same contraction as compute_A_from_B_full, but the output axes are
        interleaved as (element, point) per dimension so that a free reshape
        yields the FV cell-based layout, with no transpose+copy afterwards.

        Input : B (u, Nz, Ny, Nx, n, n, n)   [element-based/SD layout]
        Output: A (u, Nz*k, Ny*j, Nx*i)      [cell-based/FV layout]
        """
        if ndim==3:
            A = np.einsum("kn,jm,il,uzyxnml->uzkyjxi",
                         A_to_B,
                         A_to_B,
                         A_to_B, B, optimize=True)
            u, Nz, k, Ny, j, Nx, i = A.shape
            return A.reshape(u, Nz*k, Ny*j, Nx*i)
        elif ndim==2:
            A = np.einsum("jm,il,uyxml->uyjxi",
                         A_to_B,
                         A_to_B, B, optimize=True)
            u, Ny, j, Nx, i = A.shape
            return A.reshape(u, Ny*j, Nx*i)
        else:
            A = np.einsum("il,uxl->uxi",
                         A_to_B, B)
            u, Nx, i = A.shape
            return A.reshape(u, Nx*i)
