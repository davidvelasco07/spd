import pytest
import sys
import numpy as np
sys.path.append("../src")
from sdfb_simulator import SDFB_Simulator

tolerance=1E-14
N=6

def make_sim(p, N, **kwargs):
    cfl = 0.4 if p >= 3 else 0.8
    return SDFB_Simulator(p=p, N=N, use_cupy=False, cfl_coeff=cfl, **kwargs)


@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
@pytest.mark.parametrize("FB" , [False, True])
def test_update_sd(p,N,FB):
    s = make_sim(p=p, N=N, FB=FB)
    s.perform_time_evolution(0.5)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
def test_update_fb_godunov(p,N):
    s = make_sim(p=p, N=N, FB=True, godunov=True)
    s.perform_time_evolution(0.5)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
def test_update_fb(p,N):
    s = make_sim(p=p, N=N, FB=True)
    s.perform_time_evolution(0.5)
    assert np.mean(np.abs(s.dm.W_cv[0]-s.W_init_cv[0])) > tolerance
    assert np.mean(np.abs(s.dm.W_cv[1:]-s.W_init_cv[1:])) < tolerance