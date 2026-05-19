import pytest
import sys
sys.path.append("../src")
import numpy as np

from numerics import polynomials as poly
tolerance=1E-14

@pytest.mark.parametrize("p" , range(8))
def test_sp(p):
    sp = poly.solution_points(0,1,p)
    assert sp.size==(p+1)
    assert np.abs(2*sp.sum()-(p+1)) < tolerance
    assert np.abs(sp.mean()-.5) < tolerance

@pytest.mark.parametrize("p" , range(8))
def test_fp(p):
    fp = poly.flux_points(0,1,p)
    assert fp.size==(p+2)
    assert np.abs(2*fp.sum()-(p+2)) < tolerance
    assert np.abs(fp.mean()-.5) < tolerance

@pytest.mark.parametrize("p" , range(8))
def test_lm(p):
    sp = poly.solution_points(0,1,p)
    fp = poly.flux_points(0,1,p)
    for a in [sp,fp]:
        for b in [sp,fp]:
            lm = poly.lagrange_matrix(b,a)
            assert lm.shape == (b.size,a.size)
            assert np.abs(lm.sum()-(b.size)) < tolerance
            assert np.abs(lm.mean()-1/(a.size)) < tolerance