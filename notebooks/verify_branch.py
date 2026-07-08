"""Run a matrix of configurations and dump final-state checksums/arrays.

Usage: python verify_branch.py <tag>
Saves results to notebooks/verify_state_<tag>.npz; compare two tags with
verify_branch_compare.py.
"""
import sys

import numpy as np

from spd.spd_simulator import SPD_Simulator as SDFB_Simulator
from spd.initial_conditions.initial_conditions_2d import step_function
from spd.initial_conditions.initial_conditions_3d import sine_wave as sine3d


cases = {}

common = dict(verbose=False, init_fct=step_function)

# SDFB + RK3 (compute_update path)
for gpu in (False, True):
    key = f"sdfb_rk3_2d_{'gpu' if gpu else 'cpu'}"
    s = SDFB_Simulator(p=3, N=(16, 16), use_cupy=gpu, FB=True,
                       time_integrator="rk3", **common)
    s.perform_iterations(15)
    cases[key] = np.asarray(s.dm.W_cv)

# SDFB + ADER (ader_update path incl. update_solution_points=True)
for gpu in (False, True):
    key = f"sdfb_ader_2d_{'gpu' if gpu else 'cpu'}"
    s = SDFB_Simulator(p=3, N=(16, 16), use_cupy=gpu, FB=True,
                       time_integrator="ader", **common)
    s.perform_iterations(15)
    cases[key] = np.asarray(s.dm.W_cv)

# Plain SD (no fallback) + RK3, 3D
s = SDFB_Simulator(p=3, N=(8, 8, 8), use_cupy=True, FB=False,
                   time_integrator="rk3", verbose=False, init_fct=sine3d)
s.perform_iterations(10)
cases["sd_rk3_3d_gpu"] = np.asarray(s.dm.W_cv)

# SDFB 3D + RK3 on GPU
s = SDFB_Simulator(p=3, N=(8, 8, 8), use_cupy=True, FB=True,
                   time_integrator="rk3", verbose=False, init_fct=sine3d)
s.perform_iterations(10)
cases["sdfb_rk3_3d_gpu"] = np.asarray(s.dm.W_cv)

# 1D SDFB
from spd.initial_conditions.initial_conditions_1d import step_function as step1d
s = SDFB_Simulator(p=3, N=(32,), use_cupy=True, FB=True,
                   time_integrator="rk3", verbose=False,
                   init_fct=step1d)
s.perform_iterations(15)
cases["sdfb_rk3_1d_gpu"] = np.asarray(s.dm.W_cv)

# Well-balanced SDFB (exercises the equilibrium dual-layout switching)
try:
    from spd.initial_conditions.initial_conditions_2d import RTI
    for gpu in (False, True):
        s = SDFB_Simulator(p=3, N=(16, 16), use_cupy=gpu, FB=True, WB=True,
                           potential=True, time_integrator="rk3",
                           verbose=False, init_fct=RTI, eq_fct=RTI)
        s.perform_iterations(10)
        cases[f"sdfb_wb_2d_{'gpu' if gpu else 'cpu'}"] = np.asarray(s.dm.W_cv)
except Exception as e:
    print(f"WB case skipped: {type(e).__name__}: {e}")

tag = sys.argv[1]
np.savez(f"verify_state_{tag}.npz", **cases)
for k, v in cases.items():
    print(f"{k:24s} sum={v.sum():+.12e} absmax={np.abs(v).max():.12e}")
print(f"saved verify_state_{tag}.npz")
