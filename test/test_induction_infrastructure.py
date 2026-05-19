"""Smoke tests for induction infrastructure wiring."""

import unittest

import numpy as np

from induction.induction_sd_scheme import InductionSD_Scheme, MHDInductionSD_Scheme
from induction.induction_fv_scheme import InductionFV_Scheme
from induction.induction_simulator import InductionSimulator
from integrators.rk_induction import InductionRKIntegrator
from finite_volume.fv_scheme import FV_Scheme


def _torus_init(mesh, var):
    if var == 0:
        return 0 * mesh[0] + 1.0
    if var in (1, 2):
        return 0 * mesh[0] + 0.05
    if var == 4:
        return 0 * mesh[0] + 1.0
    return 0 * mesh[0]


def _torus_Az(mesh, j):
    if j == 2:
        return -1.0 * mesh[0]
    return 0 * mesh[0]


class TestInductionInfra(unittest.TestCase):
    def test_induction_rk_integrator_stages(self):
        rk = InductionRKIntegrator(m=2, ndim=2)
        self.assertEqual(rk.nstages, 3)

    def test_mhd_induction_extends_induction_sd(self):
        self.assertTrue(issubclass(MHDInductionSD_Scheme, InductionSD_Scheme))

    def test_fv_induction_scheme_class(self):
        self.assertTrue(issubclass(InductionFV_Scheme, FV_Scheme))

    def test_fv_induction_one_rk_step(self):
        sim = InductionSimulator(
            scheme_fb="FV",
            init_fct=_torus_init,
            vectorpot_fct=_torus_Az,
            N=(8, 8),
            p=1,
            m=1,
            time_integrator="rk2",
            init=True,
            verbose=False,
        )
        sim.perform_iterations(1)
        self.assertGreater(sim.time, 0.0)
        self.assertTrue(np.isfinite(sim.ho_scheme.dm.By_fp).all())


if __name__ == "__main__":
    unittest.main()
