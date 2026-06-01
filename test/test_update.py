import pytest
import sys
import numpy as np
from spd.sdfb_simulator import SDFB_Simulator
from spd.spectral_difference.sd_simulator import SD_Simulator
from spd.finite_volume.fv_simulator import FV_Simulator

tolerance=1E-14
N=6

# ----------------------------------------------------------------
# SDFB_Simulator: SD with and without fallback blending
# ----------------------------------------------------------------

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N),(N,N,N)])
@pytest.mark.parametrize("FB" , [True,False])
def test_sdfb_ader_update(p,N,FB):
    """Test SDFB with ADER (default) time integration."""
    s = SDFB_Simulator(p=p, N=N, use_cupy=False, FB=FB)
    s.perform_iterations(1)

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
@pytest.mark.parametrize("FB" , [True,False])
@pytest.mark.parametrize("time_integrator" , ["rk1","rk2","rk3","rk4"])
def test_sdfb_rk_update(p,N,FB,time_integrator):
    """Test SDFB with RK time integration."""
    s = SDFB_Simulator(p=p, N=N, use_cupy=False, FB=FB,
                        time_integrator=time_integrator)
    s.perform_iterations(1)

# ----------------------------------------------------------------
# Pure SD_Simulator
# ----------------------------------------------------------------

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N),(N,N,N)])
def test_sd_ader_update(p,N):
    """Test pure SD with ADER."""
    s = SD_Simulator(p=p, N=N, use_cupy=False)
    s.perform_iterations(1)

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N)])
@pytest.mark.parametrize("time_integrator" , ["rk1","rk2","rk3","rk4"])
def test_sd_rk_update(p,N,time_integrator):
    """Test pure SD with RK."""
    s = SD_Simulator(p=p, N=N, use_cupy=False,
                     time_integrator=time_integrator)
    s.perform_iterations(1)

# ----------------------------------------------------------------
# Pure FV_Simulator
# ----------------------------------------------------------------

@pytest.mark.parametrize("p" , [1])
@pytest.mark.parametrize("N" , [(N,),(N,N),(N,N,N)])
@pytest.mark.parametrize("scheme" , ["MUSCL","MUSCL-Hancock"])
def test_fv_rk_update(p,N,scheme):
    """Test FV with RK (default for FV)."""
    s = FV_Simulator(p=p, N=N, use_cupy=False, scheme=scheme)
    s.perform_iterations(1)
