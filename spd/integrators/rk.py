import numpy as np

from .integrator import Integrator
from spd.runtime.gpu import CUPY_AVAILABLE, is_gpu_array

if CUPY_AVAILABLE:
    import cupy as cp

    # Fused RK linear combinations: one kernel per stage/final update
    # instead of one scaled add (plus temporary) per K array.
    _rk_comb_cache = {}

    def _rk_comb_kernel(n, with_U):
        """out = [U -] sum_i c_i * K_i  (minus signs when with_U)."""
        key = (n, with_U)
        if key not in _rk_comb_cache:
            ins = ", ".join(f"T K{i}, float64 c{i}" for i in range(n))
            if with_U:
                ins = "T U, " + ins
                expr = "U" + "".join(f" - c{i} * K{i}" for i in range(n))
            else:
                expr = " + ".join(f"c{i} * K{i}" for i in range(n))
            _rk_comb_cache[key] = cp.ElementwiseKernel(
                ins, "T out", f"out = {expr};",
                f"rk_comb_{n}_{int(with_U)}_k",
            )
        return _rk_comb_cache[key]


def stage_state(out, U, terms):
    """out = U - sum_i c_i*K_i for terms [(K_i, c_i), ...] (dispatcher)."""
    if is_gpu_array(out) and terms:
        args = [U]
        for K, c in terms:
            args += [K, c]
        _rk_comb_kernel(len(terms), True)(*args, out)
        return
    out[...] = U
    for K, c in terms:
        out[...] -= c * K


def weighted_sum(out, terms):
    """out = sum_i c_i*K_i for terms [(K_i, c_i), ...] (dispatcher)."""
    if is_gpu_array(out):
        args = []
        for K, c in terms:
            args += [K, c]
        _rk_comb_kernel(len(terms), False)(*args, out)
        return
    out[...] = 0.0
    for K, c in terms:
        out[...] += K * c


class RK_Integrator(Integrator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ader = False
        self.A, self.b, self.c = self._butcher_table(self.m + 1)
        self.nstages = len(self.b)

    def _butcher_table(self, order: int):
        if order <= 1:
            A = np.array([[0.0]])
            b = np.array([1.0])
            c = np.array([0.0])
        elif order == 2:
            A = np.array([[0.0, 0.0],
                          [1.0, 0.0]])
            b = np.array([0.5, 0.5])
            c = np.array([0.0, 1.0])
        elif order == 3:
            A = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.25, 0.25, 0.0]])
            b = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
            c = np.array([0.0, 1.0, 0.5])
        else:
            A = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]])
            b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
            c = np.array([0.0, 0.5, 0.5, 1.0])
        return A, b, c

    def allocate_arrays(self, target) -> None:
        """
        Allocate RK stage arrays on the target object.
        
        Parameters
        ----------
        target : SemiDiscreteScheme or Simulator
            The object that holds time_integrator_arrays().
        """
        target.dm.U_stage = target.array_sp()
        for stage in range(self.nstages):
            target.dm.__setattr__(f"K_{stage}", target.array_sp())

    def update(self, target) -> None:
        """
        Perform one multi-stage RK time step.
        
        Parameters
        ----------
        target : Simulator or SemiDiscreteScheme
            Object with get_solution(), compute_update(), update_solution(),
            and dt attribute.
        """
        dt = target.dt
        for stage in range(self.nstages):
            # Skip zero Butcher entries (exact no-ops).
            terms = [
                (target.dm.__getattribute__(f"K_{j}"), dt * self.A[stage, j])
                for j in range(stage)
                if self.A[stage, j] != 0.0
            ]
            stage_state(target.dm.U_stage, target.get_solution(), terms)
            # Compute dUdt at the current stage
            Ks = target.dm.__getattribute__(f"K_{stage}")
            Ks[...] = target.compute_update(target.dm.U_stage, ader=False,
                                         c_l=self.c[stage], dt=dt)
        terms = [
            (target.dm.__getattribute__(f"K_{stage}"), dt * self.b[stage])
            for stage in range(self.nstages)
        ]
        weighted_sum(target.dm.U_stage, terms)
        target.update_solution(target.dm.U_stage)