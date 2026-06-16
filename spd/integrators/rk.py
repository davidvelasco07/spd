import numpy as np

from .integrator import Integrator


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
            target.dm.U_stage[...] = target.get_solution()
            for j in range(stage):
                Kj = target.dm.__getattribute__(f"K_{j}")
                target.dm.U_stage[...] -= dt * self.A[stage, j] * Kj
            # Compute dUdt at the current stage
            Ks = target.dm.__getattribute__(f"K_{stage}")
            Ks[...] = target.compute_update(target.dm.U_stage, ader=False,
                                         c_l=self.c[stage], dt=dt)
        target.dm.U_stage[...] = 0.0
        for stage in range(self.nstages):
            Ks = target.dm.__getattribute__(f"K_{stage}")
            target.dm.U_stage[...] += Ks * dt * self.b[stage]
        target.update_solution(target.dm.U_stage)