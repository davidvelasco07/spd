import pytest
import sys
sys.path.append("../src")
import numpy as np
import polynomials as poly
from transforms import compute_A_from_B
from transforms import compute_A_from_B_full
from sd_simulator import SD_Simulator

tolerance=1E-14
N=6

@pytest.mark.parametrize("p" , [1,3,7])
@pytest.mark.parametrize("ndim" , [1,2,3])
@pytest.mark.parametrize("Nx" , [1,N])
@pytest.mark.parametrize("Ny" , [1,N])
@pytest.mark.parametrize("Nz" , [1,N])
def test_back_and_forth(p,Nx,Ny,Nz,ndim):
    s = SD_Simulator(p=p,Nx=Nx,Ny=Ny,Nz=Nz,ndim=ndim)
    W = s.compute_cv_from_sp(s.compute_sp_from_cv(s.dm.W_cv))
    assert np.all(np.abs(W-s.dm.W_cv) < tolerance)
    W = s.compute_sp_from_fp(s.compute_fp_from_sp(s.dm.W_sp,"x"),"x")
    assert np.all(np.abs(W-s.dm.W_sp) < tolerance)

@pytest.mark.parametrize("p" , [1,3,7])
@pytest.mark.parametrize("ndim" , [2])
@pytest.mark.parametrize("Nx" , [N])
@pytest.mark.parametrize("Ny" , [N])
def test_xy_symmetry(p,Nx,Ny,ndim):
    s = SD_Simulator(p=p,Nx=Nx,Ny=Ny,ndim=ndim)
    W_x = s.transpose_to_fv(s.compute_fp_from_sp(s.dm.W_sp,"x"))
    W_y = s.transpose_to_fv(s.compute_fp_from_sp(s.dm.W_sp,"y"))
    W_y = np.transpose(W_y,(0,2,1))
    assert np.all(np.abs(W_x-W_y) < tolerance)