import pytest
import numpy as np
from spd.numerics import polynomials as poly
from spd.numerics.transforms import compute_A_from_B
from spd.numerics.transforms import compute_A_from_B_full
from spd.spectral_difference.sd_simulator import SD_Simulator
from itertools import repeat

tolerance=1E-14
N=6

@pytest.mark.parametrize("p"    , [1,3])
@pytest.mark.parametrize("ndim" , [1,2,3])
@pytest.mark.parametrize("N"    , [1,N])
def test_back_and_forth(p,N,ndim):

    s = SD_Simulator(p=p,N=tuple(repeat(N,ndim)))
    W = s.compute_cv_from_sp(s.compute_sp_from_cv(s.dm.W_cv))
    assert np.all(np.abs(W-s.dm.W_cv) < tolerance)
    W = s.compute_sp_from_fp(s.compute_fp_from_sp(s.dm.W_sp,"x"),"x")
    assert np.all(np.abs(W-s.dm.W_sp) < tolerance)

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("ndim" , [2])
@pytest.mark.parametrize("N" , [1,N])
def test_xyz_symmetry(p,N,ndim):
    s = SD_Simulator(p=p,N=tuple(repeat(N,ndim)))
    W_x = s.transpose_to_fv(s.compute_fp_from_sp(s.dm.W_sp,"x"))
    W_y = s.transpose_to_fv(s.compute_fp_from_sp(s.dm.W_sp,"y"))
    W_y = np.transpose(W_y,(0,2,1))
    assert np.all(np.abs(W_x-W_y) < tolerance)