"""Verify fused GPU kernels (MUSCL reconstruction, NAD/PAD, flux blend)
against the unfused paths, for both minmod and moncen limiters.

The unfused runs monkeypatch is_gpu_array to False in the relevant modules,
forcing the generic numpy-syntax code (which still executes on CuPy arrays
via the array-function protocol) on identical device data.
"""
import numpy as np
import cupy as cp

import spd.finite_volume.muscl as muscl
import spd.fallback.trouble_detection as td
import spd.fallback.fallback as fb
import spd.integrators.rk as rk
from spd.runtime.gpu import is_gpu_array as orig_is_gpu_array
from spd.spd_simulator import SPD_Simulator as SDFB_Simulator
from spd.initial_conditions.initial_conditions_2d import step_function


def set_fused(on):
    f = orig_is_gpu_array if on else (lambda x: False)
    muscl.is_gpu_array = f
    td.is_gpu_array = f
    fb.is_gpu_array = f
    rk.is_gpu_array = f


def run(limiter, fused, steps=20):
    set_fused(fused)
    s = SDFB_Simulator(
        p=3, N=(16, 16), use_cupy=True, FB=True,
        time_integrator="rk3", slope_limiter=limiter,
        init_fct=step_function, verbose=False,
    )
    s.perform_iterations(steps)
    n_troubled = float(s.scheme.dm.troubles.sum())
    U = cp.asnumpy(s.dm.U_cv).copy()
    set_fused(True)
    return U, n_troubled


for limiter in ("minmod", "moncen"):
    U_fused, nt_f = run(limiter, fused=True)
    U_plain, nt_p = run(limiter, fused=False)
    diff = np.max(np.abs(U_fused - U_plain))
    rel = diff / np.max(np.abs(U_plain))
    print(f"{limiter:8s} troubled cells (last step): fused={nt_f:.0f} "
          f"unfused={nt_p:.0f}  max|dU|={diff:.3e}  rel={rel:.3e}")
    assert nt_f == nt_p, "trouble counts differ"
    assert rel < 1e-12, "solutions differ"
print("OK: fused kernels match unfused paths")
