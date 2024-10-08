import pytest
import sys
import numpy as np
sys.path.append("../src")
from sdader_simulator import SDADER_Simulator

tolerance=1E-14
N=6

@pytest.mark.parametrize("p" , [1,3])
@pytest.mark.parametrize("N" , [(N,),(N,N),(N,N,N)])
@pytest.mark.parametrize("update" , ["SD","FV"])
@pytest.mark.parametrize("FB" , [True,False])
def test_update(p,N,update,FB):
    s = SDADER_Simulator(p=p,N=N,use_cupy=False,
                         FB=FB,
                         update=update)
    s.perform_iterations(1)
