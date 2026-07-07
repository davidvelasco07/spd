from typing import Callable, Tuple
import numpy as np
import os

from .runtime.data_management import GPUDataManager   # kept for lazy fallback
from .runtime.data_management import CupyLocation
from timeit import default_timer as timer
from .runtime.comms import CommHelper
from .initial_conditions import sine_wave
from . import hydro
from .MHD import mhd


class Simulator:
    """
    Main simulation driver.

    Couples a semi-discrete spatial scheme with a time integrator
    to evolve a system of conservation laws.

    The Simulator owns:
      - Mesh / grid configuration
      - Physics parameters (gamma, viscosity, etc.)
      - Equations module (hydro, mhd)
      - Data manager (CPU/GPU arrays)
      - MPI communicator
      - Time integrator (ADER or RK)
      - Semi-discrete scheme (SD, FV, or composite)
      - I/O and time loop

    Parameters
    ----------
    scheme : SemiDiscreteScheme or None
        Pre-built scheme object.  When provided the simulator skips
        the legacy "init" path and uses this scheme directly.
        When *None* (default / backward compat) subclasses create
        the scheme themselves.
    """

    def __init__(
        self,
        init_fct: Callable = None,
        eq_fct: Callable = None,
        vectorpot_fct: Callable = None,
        soe="hydro",
        p: int = 1,
        m: int = -1,
        time_integrator: str = "ader",
        N: Tuple = (32, 32),
        Nghe: int = 1,
        Nghc: int = 2,
        xlim: Tuple = (0, 1),
        ylim: Tuple = (0, 1),
        zlim: Tuple = (0, 1),
        gamma: float = 1.4,
        beta: float = 2.0 / 3,
        nu: float = 1e-4,
        chi: float = 1e-4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1e-10,
        viscosity: bool = False,
        thdiffusion: bool = False,
        potential: bool = False,
        passives: list = [],
        WB: bool = False,
        use_cupy: bool = True,
        BC: Tuple = (
            ("periodic", "periodic"),
            ("periodic", "periodic"),
            ("periodic", "periodic"),
        ),
        verbose=True,
        available_time=3600.0,
        init: bool = True,
        folder: str = "outputs/",
    ):
        self.folder = folder
        self.init = init
        self.init_fct = init_fct if init_fct is not None else sine_wave()
        self.eq_fct = eq_fct if eq_fct is not None else sine_wave()
        self.vectorpot_fct = vectorpot_fct
        self.soe = soe
        if soe == "hydro" or soe == "induction":
            self.equations = hydro
        else:
            self.equations = mhd
        if m == -1:
            # By default m=p
            m = p
        self.p = p  # Space order
        self.m = m  # Time  order
        self.time_integrator = time_integrator
        self.integrator = None
        self.scheme = None  # Set by subclass or factory
        self.execution_time = 0.0
        ndim = len(N)
        self.ndim = ndim
        assert len(BC) >= ndim
        self.BC = {}
        self.idims = {}
        self.dims = {}

        dims = ["x", "y", "z"]
        for idim in range(ndim):
            dim = dims[idim]
            self.idims[idim] = dim
            self.dims[dim] = idim
            self.BC[dim] = BC[idim]
            self.__setattr__(f"N{dim}", N[idim])

        self.Y = ndim > 1
        self.Z = ndim > 2
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0

        self.Nghe = Nghe  # Number of ghost element layers
        self.Nghc = Nghc  # Number of ghost cell layers
        self.ndim = ndim
        self.gamma = gamma
        self.beta = beta
        self.nu = nu
        self.chi = chi
        self.cfl_coeff = cfl_coeff
        self.min_c2 = min_c2
        self.viscosity = viscosity
        self.thdiffusion = thdiffusion
        self.passives = passives
        self.potential = potential
        self.WB = WB
        self.verbose = verbose
        self.comms = CommHelper(self.ndim)
        self.use_cupy = use_cupy
        # dm is now owned by the scheme; see the dm property below.
        self.outputs = []
        self.available_time = available_time

        self.nghx = Nghc
        self.nghy = (0, Nghc)[self.Y]
        self.nghz = (0, Nghc)[self.Z]

        self.lim = {}
        self.len = {}
        self.h = {}
        self.N = {}
        self.n = {}
        self.ngh = {}
        self.h_min = 1e10
        for dim in self.dims:
            self.lim[dim] = self.__getattribute__(f"{dim}lim")
            self.__setattr__(f"{dim}len", self.lim[dim][1] - self.lim[dim][0])
            self.len[dim] = self.__getattribute__(f"{dim}len")
            self.N[dim] = self.__getattribute__(f"N{dim}")
            self.__setattr__(f"d{dim}", self.len[dim] / self.N[dim])
            self.h[dim] = self.__getattribute__(f"d{dim}")
            self.ngh[dim] = self.__getattribute__(f"ngh{dim}")
            self.h_min = min(self.h_min, self.h[dim])
            self.n[dim] = self.p + 1
            self.__setattr__(f"n{dim}", self.p + 1)
        self.n_step = 0

        self.init_fields()

        # For Induction and MHD
        if self.ndim == 3:
            self.Edims = self.dims
            self.Eidims = self.idims
        else:
            self.Edims = {"z": 2}
            self.Eidims = {2: "z"}

        for dim in self.dims:
            n = self.comms.__getattribute__(f"n{dim}")
            x = self.comms.__getattribute__(f"{dim}")
            self.N[dim] = int(self.N[dim] // n)
            self.len[dim] = self.len[dim] / n
            start, end = self.lim[dim]
            start += x * self.len[dim]
            end = start + self.len[dim]
            self.lim[dim] = (start, end)
        self.rank = self.comms.rank
        self.select_integrator(time_integrator)

        self.noutput = 0


    # ----------------------------------------------------------------
    # Data manager (owned by the scheme)
    # ----------------------------------------------------------------

    @property
    def dm(self):
        """
        The GPUDataManager lives on the scheme.  When no scheme has
        been attached yet (e.g. bare Simulator in tests), a private
        fallback dm is lazily created.
        """
        scheme = self.__dict__.get('scheme')
        if scheme is not None:
            return scheme.dm
        # Lazy fallback for a bare Simulator that has no scheme
        dm = self.__dict__.get('_dm')
        if dm is None:
            dm = GPUDataManager(self.use_cupy)
            object.__setattr__(self, '_dm', dm)
        return dm

    # ----------------------------------------------------------------
    # Integrator selection
    # ----------------------------------------------------------------

    def select_integrator(self, time_integrator: str = None):
        if time_integrator is not None:
            self.time_integrator = time_integrator
        from .integrators.ader import ADER_Integrator
        from .integrators.rk import RK_Integrator
        from .integrators.rk_induction import InductionRKIntegrator

        integrator_key = (self.time_integrator or "ader").lower()
        if integrator_key == "ader":
            self.integrator = ADER_Integrator(m=self.m, ndim=self.ndim)
            self.ader = True
        elif "rk" in integrator_key:
            if len(integrator_key.split("rk")) > 1:
                self.m = int(integrator_key.split("rk")[1]) - 1
            assert self.m >= 0, "RK order must be greater than 0"
            assert self.m <= 5, "RK implementation only supports up to 5th order"
            if self.soe == "induction":
                self.integrator = InductionRKIntegrator(m=self.m, ndim=self.ndim)
            else:
                self.integrator = RK_Integrator(m=self.m, ndim=self.ndim)
            self.ader = False
        else:
            raise ValueError(f"Unknown time_integrator '{self.time_integrator}'")

    # ----------------------------------------------------------------
    # Field / variable setup
    # ----------------------------------------------------------------

    def init_fields(self):
        self.variables = [r"$\rho$"]
        self._d_ = 0
        self.vels = np.arange(3) + 1
        idim = 1
        for dim in "xyz":
            name = f"v{dim}"
            self.variables.append(name)
            self.__setattr__(f"_{name}_", idim)
            idim += 1
        self.variables.append("P")
        self._p_ = 4
        self.nvar = self._p_ + 1
        self.npassive = len(self.passives)
        for i in range(self.npassive):
            self.variables.append(self.passives[i])
        self.nvar += self.npassive
        if self.thdiffusion:
            self._t_ = self.nvar
            self.nvar += 1
            self.variables.append("T")
        else:
            self._t_ = None
        if self.soe == "induction":
            self.variables.extend([f"$B_{d}$" for d in self.dims])
            self.variables.extend([r"$B^2$", r"$S$"])
        if self.soe == "mhd":
            self.b = {}
            for dim in self.dims:
                name = f"$B_{dim}$"
                self.variables.append(name)
                self.__setattr__(f"_b{dim}_", self.nvar)
                self.b[dim] = self.nvar
                self.nvar += 1
            self.variables.extend([r"$B^2$", r"$S$"])

    # ----------------------------------------------------------------
    # Mesh / geometry
    # ----------------------------------------------------------------

    def shape(self, idim):
        return (
            (None,) * (self.ndim - idim)
            + (slice(None),)
            + (None,) * (idim)
        )

    @property
    def mesh_cv(self) -> np.ndarray:
        return self.scheme.mesh_cv

    # ----------------------------------------------------------------
    # Physics helpers
    # ----------------------------------------------------------------

    def post_init(self) -> None:
        self.scheme.post_init()

    def compute_primitives(self, U, **kwargs) -> np.ndarray:
        return self.equations.compute_primitives(
            U,
            self.vels,
            self._p_,
            self.gamma,
            _t_=self._t_,
            thdiffusion=self.thdiffusion,
            npassive=self.npassive,
            **kwargs,
        )

    def compute_conservatives(self, W, **kwargs) -> np.ndarray:
        return self.equations.compute_conservatives(
            W,
            self.vels,
            self._p_,
            self.gamma,
            _t_=self._t_,
            thdiffusion=self.thdiffusion,
            npassive=self.npassive,
            **kwargs,
        )

    def compute_fluxes(self, F, M, vels, prims) -> np.ndarray:
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        self.equations.compute_fluxes(
            W,
            vels,
            self._p_,
            self.gamma,
            F=F,
            thdiffusion=self.thdiffusion,
            npassive=self.npassive,
        )

    def compute_viscous_fluxes(
        self, M, dMs, vels, prims=False, **kwargs
    ) -> np.ndarray:
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        return self.equations.compute_viscous_fluxes(
            W,
            vels,
            dMs,
            self._p_,
            self.nu,
            self.beta,
            thdiffusion=self.thdiffusion,
            npassive=self.npassive,
            **kwargs,
        )

    def compute_thermal_fluxes(self, M, dMs, prims=False) -> np.ndarray:
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        return self.equations.compute_thermal_fluxes(W, dMs, self.chi, self._t_)

    def apply_potential(self, dUdt, U, grad_phi):
        _p_ = self._p_
        for idim in self.idims:
            vel = self.vels[idim]
            dUdt[vel, ...] += U[0] * grad_phi[idim]
            dUdt[_p_, ...] += U[vel] * grad_phi[idim]

    def riemann_solver(self, solver):
        return lambda ML, MR, vels, prims: solver(
            ML,
            MR,
            vels,
            prims,
            _p_=self._p_,
            gamma=self.gamma,
            min_c2=self.min_c2,
            npassive=self.npassive,
        )

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    @property
    def domain_size(self):
        if self.scheme is not None:
            return self.scheme.domain_size
        return np.prod([self.N[dim] for dim in self.dims])

    def regular_faces(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [
            np.linspace(lim[dim][0], lim[dim][1], N[dim] * n[dim] + 1)
            for dim in self.dims
        ]

    def regular_centers(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [
            np.linspace(lim[dim][0], lim[dim][1], N[dim] * n[dim])
            for dim in self.dims
        ]

    def crop(self, M, ngh=1) -> np.ndarray:
        return M[
            (slice(None),) + (slice(ngh, -ngh),) * self.ndim + (Ellipsis,)
        ]

    def create_dicts(self):
        if self.scheme is not None:
            self.scheme.create_dicts()

    def convert_solution(self, W=False):
        if self.scheme is not None:
            self.scheme.convert_solution(W=W)

    # ----------------------------------------------------------------
    # Simulation lifecycle
    # ----------------------------------------------------------------
    def switch_to_device(self):
        self.dm.switch_to(CupyLocation.device)
    
    def switch_to_host(self):
        self.dm.switch_to(CupyLocation.host)

    def init_sim(self):
        self.checkpoint = False
        self.switch_to_device()
        self.create_dicts()
        self.execution_time = -timer()

    def end_sim(self):
        self.execution_time += timer()
        # Convert while arrays are still on the device: the host-side
        # conversion (numpy einsum) is orders of magnitude slower.
        self.convert_solution()
        self.switch_to_host()
        self.create_dicts()
        if self.rank == 0:
            print(
                f"t={self.time}, steps taken {self.n_step}, "
                f"time taken {np.round(self.execution_time,3)}, bzcps = {np.round(self.zone_cycles/1E+9,3)}"
            )

    @property
    def elapsed_time(self):
        return self.execution_time - timer()

    @property
    def cost_per_step(self):
        cost = 0 if self.n_step == 0 else self.execution_time / self.n_step
        return cost

    @property
    def zone_cycles(self):
        #print(self.cost_per_step,self.domain_size, self.domain_size/self.cost_per_step)
        return self.domain_size/self.cost_per_step

    def perform_update(self) -> bool:
        """
        Perform a single time step: integrator advances the scheme.
        """
        self.n_step += 1
        self.integrator.update(self.scheme)
        self.scheme.post_update()
        self.time += self.dt
        return True

    def perform_iterations(self, n_step: int) -> None:
        self.init_sim()
        for i in range(n_step):
            self.compute_dt()
            self.perform_update()
        self.end_sim()

    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        self.init_sim()
        while self.time < t_end:
            if not self.n_step % 100 and self.rank == 0 and self.verbose:
                print(f"Time step #{self.n_step} (t = {np.round(self.time,3)})", end="\r")
            self.compute_dt()
            if self.time + self.dt >= t_end:
                dt = t_end - self.time
                if dt > 1e-14:
                    self.dt = dt
                    self.scheme.dt = dt
                else:
                    print(f"dt={dt}")
                    break
            self.status = self.perform_update()
            if not (self.checkpoint):
                if (
                    (self.available_time - self.elapsed_time) < 120
                ) and self.rank == 0:
                    self.checkpoint = True
                self.checkpoint = self.comms.reduce_max(self.checkpoint)
                if self.checkpoint:
                    self.output()
                    print("Checkpoint")
                    self.noutput -= 1
        self.end_sim()

    # ----------------------------------------------------------------
    # I/O
    # ----------------------------------------------------------------

    def output(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank == 0:
            os.makedirs(folder)

        self.comms.barrier()

        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size > 1:
            file += f"_{self.comms.rank}"
        np.save(file, self.dm.W_cv)
        self.outputs.append([self.time, self.noutput])
        if self.rank == 0:
            np.savetxt(folder + "/outputs.out", self.outputs)
        self.noutput += 1

    def load_output(self):
        folder = self.folder
        outputs = np.loadtxt(folder + "/outputs.out")
        rows = int(outputs.size // 2)
        self.outputs = list(outputs.reshape([rows, 2]))
        self.time, self.noutput = self.outputs[-1]
        self.noutput = int(self.noutput)
        file = f"{folder}/Output_{str(self.noutput).zfill(5)}"
        if self.comms.size > 1:
            file += f"_{self.comms.rank}"
        self.dm.W_cv[...] = np.load(file + ".npy")
        self.convert_solution(W=True)
        self.noutput += 1

    def save_checkpoint(self):
        folder = self.folder
        if not os.path.exists(folder) and self.rank == 0:
            os.makedirs(folder)

        self.comms.barrier()

        file = f"{folder}/Checkpoint"
        if self.comms.size > 1:
            file += f"_{self.comms.rank}"
        np.save(file, self.dm.W_cv)
        self.outputs.append([self.time, self.noutput])
        if self.rank == 0:
            np.savetxt(folder + "/outputs.out", self.outputs)
        self.noutput += 1
