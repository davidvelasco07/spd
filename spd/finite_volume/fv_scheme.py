"""
Finite Volume semi-discrete scheme.

Cell-centered finite volume spatial discretization using MUSCL
or MUSCL-Hancock reconstruction with slope limiters.
Computes the spatial operator L(U) = -div(F) for the method-of-lines
formulation dU/dt = L(U).
"""

import numpy as np
from itertools import repeat

from spd.schemes.scheme import SemiDiscreteScheme
from spd.riemann_solvers.riemann_solver_1D import Riemann_solver_1D as rs1d
from . import muscl
from spd.numerics.polynomials import quadrature_mean
from spd.numerics.slicing import cut, crop_fv


class FV_Scheme(SemiDiscreteScheme):
    """
    Finite Volume semi-discrete spatial scheme.

    Uses cell-averaged values with MUSCL or MUSCL-Hancock reconstruction
    to compute interface fluxes via a Riemann solver.

    Parameters
    ----------
    sim : Simulator
        Parent simulator providing shared state.
    riemann_solver_fv : str
        Name of the Riemann solver ('llf', 'hllc', 'lhllc').
    slope_limiter : str
        Name of the slope limiter ('minmod', 'moncen').
    predictor : bool
        If True, use MUSCL-Hancock (predictor-corrector).
        If False, use plain MUSCL.
    """

    def __init__(
        self,
        sim,
        riemann_solver="llf",
        equations=None,
        slope_limiter="minmod",
        scheme="MUSCL-Hancock",
        dm=None,
    ):
        super().__init__(sim)
        if equations is None:
            equations = "mhd" if sim.soe == "mhd" else "hydro"
        self.riemann_solver_name = riemann_solver
        self.riemann_solver = rs1d(riemann_solver, equations).solver
        self.slope_limiter = muscl.Slope_limiter(slope_limiter)
        self.scheme = scheme
        if scheme == "MUSCL":
            self.fluxes = muscl.MUSCL_fluxes
        elif scheme == "MUSCL-Hancock":
            self.fluxes = muscl.MUSCL_Hancock_fluxes
        else:
            raise ValueError(f"Invalid scheme: {scheme}")
        
        self.faces = {}
        self.centers = {}
        self.h_fp = {}
        self.h_cv = {}

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        """Set up FV arrays, compute initial conditions."""
        self.compute_positions()
        self.mesh_cv = self.compute_mesh_cv()
        self.post_init()
        self.compute_dt()
        self.allocate_arrays(ader=self.ader)
        self.init_Boundaries()
        if self.potential:
            self.init_potential()
        if self.WB:
            self.init_equilibrium_state()

    def compute_positions(self):
        """
        Build the FV mesh: a uniform grid with Nghc ghost cells on each side.

        Per dimension the layout is::

            |<-ngh->|<-------- N*n active cells -------->|<-ngh->|

        Reads ``N``, ``n``, ``lim``, ``len``, ``Nghc`` from the parent
        simulator via the proxy pattern and stores face / centre
        coordinates and distances on ``self.dm`` and the local dicts.
        """
        ngh = self.Nghc
        for dim in self.dims:
            idim = self.dims[dim]
            Ncv = self.N[dim] * self.n[dim]      # active cells
            h_cell = self.len[dim] / Ncv          # uniform cell width

            # --- face positions (Ncv + 2*ngh + 1 values) ----------------
            fp = np.empty(Ncv + 2 * ngh + 1)
            # Interior faces (including domain boundaries)
            fp[ngh:-ngh] = self.lim[dim][0] + np.arange(Ncv + 1) * h_cell
            # Ghost faces: uniform extension beyond the boundaries
            for i in range(ngh):
                fp[ngh - 1 - i] = fp[ngh] - (i + 1) * h_cell
                fp[-ngh + i] = fp[-ngh - 1] + (i + 1) * h_cell

            self.dm.__setattr__(f"{dim.upper()}_fp", fp)
            self.faces[dim] = fp

            # --- cell centres (Ncv + 2*ngh values) ----------------------
            cv = 0.5 * (fp[1:] + fp[:-1])
            self.dm.__setattr__(f"{dim.upper()}_cv", cv)
            self.centers[dim] = cv

            # --- distance between consecutive faces ---------------------
            h_fp = (fp[1:] - fp[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_fp", h_fp)
            self.h_fp[dim] = h_fp

            # --- distance between consecutive centres -------------------
            h_cv = (cv[1:] - cv[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_cv", h_cv)
            self.h_cv[dim] = h_cv

    def compute_mesh_cv(self) -> np.ndarray:
        """
        Physical face coordinates on a structured grid (incl. ghost cells).

        Unlike the SD version (which has per-element sub-dimensions),
        the FV mesh is flat: the last ``ndim`` axes correspond directly
        to the face positions along each spatial direction.

        The mesh has ``Nc + 2*Nghc + 1`` faces per dimension where
        ``Nc = N * n``.

        Returns
        -------
        mesh_cv : np.ndarray
            Shape ``(ndim, Nfz, Nfy, Nfx)`` (2-D → ``(ndim, Nfy, Nfx)``).
            ``quadrature_mean`` diffs along the last *d* axes to obtain
            ``Nc + 2*Nghc`` cell averages per dimension.
        """
        Nf = [len(self.faces[dim]) for dim in self.dims]
        shape = (self.ndim,) + tuple(Nf[::-1])
        mesh_cv = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            bcast = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * idim
            )
            mesh_cv[idim] = self.faces[dim][bcast]
        return mesh_cv

    def compute_mesh(self, Points) -> np.ndarray:
        """
        Physical coordinates at arbitrary reference points for every FV cell.

        Parameters
        ----------
        Points : list of np.ndarray
            Reference-space coordinates in [0, 1], one array per
            dimension (indexed by ``idim``).

        Returns
        -------
        mesh : np.ndarray
            Shape ``(ndim, Ncy+2*Nghc, …, Ncx+2*Nghc,
            len(Points[-1]), …, len(Points[0]))``.
        """
        ngh = self.Nghc
        Ns = [self.N[dim] * self.n[dim] + 2 * ngh for dim in self.dims]
        shape = (self.ndim,) + tuple(Ns[::-1])
        for points in Points[::-1]:
            shape += (points.size,)
        mesh = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            Nc = self.N[dim] * self.n[dim]
            h_cell = self.len[dim] / Nc
            N_total = Ns[idim]
            shape1 = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (self.ndim + idim)
            )
            shape2 = (
                (None,) * (2 * self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * idim
            )
            mesh[idim] = (
                self.lim[dim][0]
                + (np.arange(N_total)[shape1] + Points[idim][shape2])
                * h_cell
                - ngh * h_cell
            )
        return mesh

    def post_init(self) -> None:
        """Initialize primitive variables on the FV mesh via quadrature."""
        ngh = self.Nghc
        nvar = self.nvar
        W_gh = self.array(nvar, ngh=ngh)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(
                self.mesh_cv, self.init_fct, self.ndim, self.p, var
            )
        # Ghosted primitives live on the data manager so host/device follows
        # ``dm.switch_to`` (same as ``W_cv`` / ``U_cv``); do not keep a separate
        # scheme-only ndarray that would stay NumPy after ``init_sim``.
        self.dm.W_gh = W_gh
        self.dm.W_cv = self.active_region(self.dm.W_gh)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)

    def init_Boundaries(self) -> None:
        na = np.newaxis if self.ader else Ellipsis
        ngh = self.Nghc
        n = self.n["x"]  # same for all dims
        M = self.dm.W_gh
        #if n > 2:
        #    M = M[crop_fv(n - ngh, -(n - ngh), 0, self.ndim, n - ngh)]
        for dim in self.dims:
            idim = self.dims[dim]
            BC = self.dm.__getattribute__(f"BC_fp_{dim}")
            BC[0][...] = M[cut( None, ngh, idim)][:,na]
            BC[1][...] = M[cut(-ngh, None, idim)][:,na]

    # ----------------------------------------------------------------
    # Potential and equilibrium
    # ----------------------------------------------------------------

    def init_potential(self) -> None:
        """
        Build the cell-centred gradient of the gravitational potential.

        The potential ``phi`` is the last variable returned by ``init_fct``;
        ``dm.grad_phi`` has one component per spatial direction (indexed by
        ``idim``) over the active region, matching the layout expected by
        :meth:`Simulator.apply_potential` and the MUSCL-Hancock source term.
        """
        ngh = self.Nghc
        phi = self.array(1, ngh=ngh)
        phi[0] = quadrature_mean(
            self.mesh_cv, self.init_fct, self.ndim, self.p, -1
        )
        self.dm.grad_phi = self.array(self.ndim)
        for dim in self.dims:
            idim = self.dims[dim]
            # Central difference of the cell-averaged potential (uniform grid).
            end_plus = -(ngh - 1) if ngh > 1 else None
            phi_plus = phi[crop_fv(ngh + 1, end_plus, idim, self.ndim, ngh)]
            phi_minus = phi[crop_fv(ngh - 1, -(ngh + 1), idim, self.ndim, ngh)]
            h_cell = self.len[dim] / (self.N[dim] * self.n[dim])
            self.dm.grad_phi[idim] = (phi_plus - phi_minus)[0] / (2 * h_cell)

    def init_equilibrium_state(self) -> None:
        """
        Build the equilibrium state used by the well-balanced FV scheme.

        Stores the cell-averaged equilibrium primitives ``dm.M_eq`` (ghosted,
        read by the MUSCL reconstruction), the equilibrium conservatives
        ``dm.U_eq_cv`` (active region), and the single-valued equilibrium
        primitive state / physical flux at each face (``M_eq_fp_{dim}`` /
        ``F_eq_fp_{dim}``).  The solution ``dm.U_cv`` is then shifted to the
        perturbation ``U' = U - U_eq`` so the potential source term and the
        perturbation fluxes vanish exactly at equilibrium.
        """
        ngh = self.Nghc
        nvar = self.nvar
        M_eq = self.array_sp(ngh=ngh)
        for var in range(nvar):
            M_eq[var] = quadrature_mean(
                self.mesh_cv, self.eq_fct, self.ndim, self.p, var
            )
        self.dm.M_eq = M_eq
        self.dm.U_eq_cv = self.compute_conservatives(self.active_region(M_eq))
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels, -idim)
            S = self.compute_slopes(M_eq, idim)
            ML = self.interpolate_L(M_eq, S, idim)
            MR = self.interpolate_R(M_eq, S, idim)
            # Single-valued (continuous) equilibrium state at the faces.
            M_eq_fp = 0.5 * (ML + MR)
            self.dm.__setattr__(f"M_eq_fp_{dim}", M_eq_fp)
            F = M_eq_fp.copy()
            self.compute_physical_fluxes(F, M_eq_fp, vels)
            self.dm.__setattr__(f"F_eq_fp_{dim}", F)
        # Evolve the perturbation; W_cv keeps the full primitive state for I/O.
        self.dm.U_cv -= self.dm.U_eq_cv

    def compute_physical_fluxes(self, F, M, vels, prims=True) -> None:
        """Evaluate the analytic physical flux ``F`` of state ``M``.

        ``FV_Scheme.compute_fluxes`` is the MUSCL flux pipeline (signature
        ``(F, dt)``), which shadows the simulator-level physical-flux method.
        The well-balanced equilibrium construction needs the raw physical
        flux, so it is recomputed here directly from the equations.
        """
        W = M if prims else self.compute_primitives(M)
        self.equations.compute_fluxes(
            W,
            vels,
            self._p_,
            self.gamma,
            F=F,
            thdiffusion=self.thdiffusion,
            npassive=self.npassive,
        )

    # ----------------------------------------------------------------
    # Array allocation
    # ----------------------------------------------------------------

    def array(self, nvar, dim="", ngh=0, ader=False) -> np.ndarray:
        shape = [nvar] if not ader else [nvar, self.nader]
        N = []
        for dim2 in self.dims:
            N.append(self.N[dim2] * self.n[dim2] + (dim2 in dim) + 2 * ngh)
        return np.ndarray(shape + N[::-1])

    def _spatial_axis_for_dim(self, dim: str) -> int:
        """
        NumPy axis index for spatial direction ``dim`` on arrays whose trailing
        ``ndim`` axes follow :meth:`array` order (``…, N_z, N_y, N_x`` for
        ``dims`` ``x``, ``y``, ``z``).
        """
        rev_keys = list(reversed(list(self.dims.keys())))
        pos = rev_keys.index(dim)
        return -(len(rev_keys) - pos)

    def compute_sp_from_fp(self, M_fp, dim: str, ader: bool = False):
        """
        Average face-centred data normal to ``dim`` onto cell-centre resolution
        along that direction (uniform grid; same role as SD ``fp_to_sp`` for
        one refinement, e.g. :math:`B_x` on ``x``-faces → cell centres).

        ``M_fp`` may have any number of leading axes (e.g. ``nvar`` or
        ``(nvar, nader)`` when ``ader`` is True); only the last ``ndim``
        dimensions are spatial and must carry the ``+1`` face resolution along
        ``dim`` (length ``N_{dim} n_{dim} + 1 + 2\\,Nghc`` with ghosts), as from
        :meth:`array` with that ``dim`` in the face string.

        Parameters
        ----------
        M_fp : ndarray
            Field on a face-normal staggered layout for ``dim``.
        dim : str
            Normal direction carrying face points (``"x"``, ``"y"``, or ``"z"``).
        ader : bool
            Ignored for arithmetic; accepted for call-site compatibility with SD.

        Returns
        -------
        ndarray
            Same leading shape as ``M_fp``; spatial size along ``dim`` is one
            less than in ``M_fp`` (cell-centre / CV spacing along that axis).
        """
        _ = ader
        ax = self._spatial_axis_for_dim(dim)
        sl0 = [slice(None)] * M_fp.ndim
        sl1 = [slice(None)] * M_fp.ndim
        sl0[ax] = slice(None, -1)
        sl1[ax] = slice(1, None)
        return 0.5 * (M_fp[tuple(sl0)] + M_fp[tuple(sl1)])

    def array_sp(self, **kwargs):
        return self.array(self.nvar, **kwargs)

    def array_fp(self, dims="xyz", **kwargs):
        return self.array(self.nvar, dim=dims, **kwargs)

    def array_RS(self, dim="x", dim2=None, **kwargs) -> np.ndarray:
        return self.array(self.nvar, dim=dim, **kwargs)

    def array_BC(self, dim="x", ader=False) -> np.ndarray:
        shape = [2, self.nvar] if not ader else [2, self.nvar, self.nader]
        ngh = self.Nghc
        N = []
        for dim2 in self.dims:
            N.append(
                self.N[dim2] * self.n[dim2] + 2 * ngh
                if dim != dim2
                else ngh
            )
        return np.ndarray(shape + N[::-1])

    def allocate_arrays(self, ader=False):
        """Allocate arrays for the time integrator."""
        super().allocate_arrays(ader)
        """Allocate FV working arrays."""
        self.dm.M = self.array_sp(ngh=self.Nghc, ader=self.ader)
        self.dm.U_new = self.array_sp()
        if self.scheme == "MUSCL-Hancock":
            self.dm.dtM = self.array_sp(ngh=self.Nghc - 1, ader=self.ader)

    # ----------------------------------------------------------------
    # Region management
    # ----------------------------------------------------------------

    def active_region(self, W,):
        ngh = self.Nghc
        return W[(Ellipsis,) + (slice(ngh, -ngh),) * self.ndim]

    def fill_active_region(self, M):
        ngh = self.Nghc
        self.dm.M[
            (Ellipsis,) + tuple(repeat(slice(ngh, -ngh), self.ndim))
        ] = M

    # ----------------------------------------------------------------
    # Dictionary creation
    # ----------------------------------------------------------------

    def create_dicts(self) -> None:
        """Create the dictionaries for the FV scheme.
        It enables writting generic functions for all dimensions.
        The arrays are stored in the data manager.
        The pointers to the working arrays are stored in the scheme.
        """
        self.F_fp = {}
        self.MR_fp = {}
        self.ML_fp = {}
        self.BC_fp = {}
        for dim in self.dims:            
            self.F_fp[dim] = self.dm.__getattribute__(f"F_fp_{dim}")
            self.MR_fp[dim] = self.dm.__getattribute__(f"MR_fp_{dim}")
            self.ML_fp[dim] = self.dm.__getattribute__(f"ML_fp_{dim}")
            self.BC_fp[dim] = self.dm.__getattribute__(f"BC_fp_{dim}")

        self.working_arrays()

    def working_arrays(self) -> None:
        """Refresh the mesh-coordinate dictionaries from the data manager
        (needed after host/device switches, via create_dicts)."""
        for dim in self.dims:
            self.faces[dim] = self.dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = self.dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = self.dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = self.dm.__getattribute__(f"d{dim}_cv")

    # ----------------------------------------------------------------
    # Solution-state access
    # ----------------------------------------------------------------
    # W_cv/U_cv always resolve through the owning data manager, so layout
    # switches (which repoint dm.W_cv/dm.U_cv between the persistent SD- and
    # FV-layout buffers) need no pointer-refresh bookkeeping here.

    @property
    def state_dm(self):
        """Data manager owning the solution state."""
        return self.dm

    @property
    def W_cv(self):
        return self.state_dm.W_cv

    @W_cv.setter
    def W_cv(self, value):
        # Supports augmented assignment (self.W_cv -= dU): the in-place op
        # already modified the array; re-storing it is a no-op repoint.
        self.state_dm.W_cv = value

    @property
    def U_cv(self):
        return self.state_dm.U_cv

    @U_cv.setter
    def U_cv(self, value):
        self.state_dm.U_cv = value

    @property
    def U_eq_cv(self):
        return self.state_dm.U_eq_cv

    # ----------------------------------------------------------------
    # MUSCL reconstruction
    # ----------------------------------------------------------------

    def compute_slopes(self, M: np.ndarray, idim: int) -> np.ndarray:
        return self.slope_limiter.compute_slopes(
            M,
            self.h_cv[self.idims[idim]],
            self.h_fp[self.idims[idim]],
            idim,
        )

    def compute_gradients(self, M: np.ndarray, idim: int) -> np.ndarray:
        return self.slope_limiter.compute_gradients(
            M,
            self.h_cv[self.idims[idim]],
            self.h_fp[self.idims[idim]],
            idim,
        )

    def interpolate_R(
        self, M: np.ndarray, S: np.ndarray, idim: int
    ) -> np.ndarray:
        """Interpolate solution to right face: MR = M - S."""
        ngh = self.Nghc
        crop = lambda start, end, idim: crop_fv(
            start, end, idim, self.ndim, ngh
        )
        return M[crop(2, -1, idim)] - S[crop(1, None, idim)]

    def interpolate_L(
        self, M: np.ndarray, S: np.ndarray, idim: int
    ) -> np.ndarray:
        """Interpolate solution to left face: ML = M + S."""
        ngh = self.Nghc
        crop = lambda start, end, idim: crop_fv(
            start, end, idim, self.ndim, ngh
        )
        return M[crop(1, -2, idim)] + S[crop(None, -1, idim)]

    def compute_prediction(
        self, W: np.ndarray, dWs: np.ndarray
    ) -> None:
        """Compute MUSCL-Hancock predictor step."""
        muscl.compute_prediction(
            W,
            dWs,
            self.dm.dtM,
            self.vels,
            self.ndim,
            self.gamma,
            self._d_,
            self._p_,
            self.WB,
            self.npassive,
        )

    # ----------------------------------------------------------------
    # Riemann problem
    # ----------------------------------------------------------------

    def solve_riemann_problem(
        self, dim: str, F: np.ndarray, prims: bool
    ) -> None:
        """Solve the Riemann problem at FV cell interfaces."""
        idim = self.dims[dim]
        vels = np.roll(self.vels, -idim)
        if self.WB:
            M_eq_fp = self.dm.__getattribute__(f"M_eq_fp_{dim}")
            self.MR_fp[dim][...] += M_eq_fp
            self.ML_fp[dim][...] += M_eq_fp
        F[...] = self.riemann_solver(
            self.ML_fp[dim],
            self.MR_fp[dim],
            F,
            vels,
            self._p_,
            self.gamma,
            self.min_c2,
            prims,
            npassive=self.npassive,
            thdiffusion=self.thdiffusion,
            min_rho=self.min_rho,
        )
        if self.WB:
            F -= self.dm.__getattribute__(f"F_eq_fp_{dim}")

    # ----------------------------------------------------------------
    # Flux computation and RHS
    # ----------------------------------------------------------------

    def compute_fluxes(self, F, dt: float) -> None:
        """Compute FV fluxes: fill ghost cells, then MUSCL + Riemann."""
        self.dm.M[...] = 0
        self.fill_active_region(self.W_cv)
        self.Boundaries(self.dm.M)
        self.fluxes(self, F, dt)
        if self.viscosity or self.thdiffusion:
            self.compute_nabla_terms(F)


    def compute_nabla_terms(self,F: dict):
        ngh=self.Nghc
        dW={}
        for dim in self.dims:
            idim = self.dims[dim]
            #Make a choice of values (here left)
            M = self.ML_fp[dim]
            h = self.h_fp[dim][cut(ngh,-ngh,idim)]
            #Compute gradient in dim at cell centers
            dW[idim] = (M[cut( 1,None,idim)]-M[cut(None,-1,idim)])/h
        dW_f = {}
        for dim in self.dims:
            shift = self.dims[dim]
            vels = np.roll(self.vels,-shift)
            #Interpolate gradients(all) to faces at dim
            idims = self.idims if self.viscosity else [idim]
            for idim in idims:
                self.fill_active_region(dW[idim])
                self.Boundaries(self.dm.M,all=False)    
                S = self.compute_slopes(self.dm.M,shift)
                #Counter the previous choice of values (now right)
                dW_f[idim] = self.interpolate_R(self.dm.M,S,shift)
            #Add viscous flux
            F[dim][...] -= self.compute_viscous_fluxes(self.ML_fp[dim],dW_f,vels,prims=True)
            if self.thdiffusion:
                #Add thermal flux
                F[dim][self._p_] -= self.compute_thermal_fluxes(self.ML_fp[dim],dW_f[self.dims[dim]],prims=True)

    def compute_dudt(self, U, ader=False) -> np.ndarray:
        """Compute dU/dt from face fluxes."""
        na = np.newaxis if ader else Ellipsis
        dUdt = U.copy() * 0
        for dim in self.dims:
            ngh = self.ngh[dim]
            shift = self.dims[dim]
            h = self.h_fp[dim][cut(ngh, -ngh, shift)][:,na]
            dUdt += (
                self.F_fp[dim][cut(1, None, shift)]
                - self.F_fp[dim][cut(None, -1, shift)]
            ) / h
        if self.potential:
            self.apply_potential(
                dUdt, U, self.dm.grad_phi
            )
        return dUdt

    def apply_fluxes(self, dt):
        """Apply computed fluxes to get the new solution."""
        dUdt = self.compute_dudt(self.U_cv)
        self.dm.U_new[...] = self.U_cv - dUdt * dt

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Evaluate the spatial RHS for a given state U."""
        if self.WB:
            # U is the perturbation U'; the reconstruction works on the
            # primitive perturbation, then adds back the equilibrium.
            self.W_cv[...] = self.compute_primitives_cv(U)
        else:
            self.compute_primitives(U, W=self.W_cv)
        self.compute_fluxes(self.F_fp, self.dt)
        return self.compute_dudt(U, ader=ader)

    def switch_to_finite_volume(self):
        """Switch to finite volume representation."""
        return

    def switch_to_high_order(self):
        """Switch to high-order representation."""
        return

    # ----------------------------------------------------------------
    # Solution state management
    # ----------------------------------------------------------------

    def get_solution(self, ader=False):
        return self.U_cv[:,np.newaxis] if self.ader else self.U_cv

    def set_solution(self, U):
        self.U_cv[...] = U

    def update_solution(self, dU, ader=False):
        self.U_cv -= dU

    def convert_solution(self, W=False):
        if W:
            U = self.compute_conservatives(self.W_cv)
            if self.WB:
                # W_cv is the full primitive state; store the perturbation.
                U -= self.dm.U_eq_cv
            self.U_cv[...] = U
        else:
            self.W_cv[...] = self.compute_primitives(self._full_solution())

    def post_update(self):
        """Called after time integrator step: update primitives."""
        self.compute_primitives(self._full_solution(), W=self.W_cv)

    def _full_solution(self):
        """Full conservative state (adds back the equilibrium when WB)."""
        if self.WB:
            return self.U_cv + self.dm.U_eq_cv
        return self.U_cv

    def update(self):
        """Perform a single FV update (no integrator, Euler-forward style)."""
        self.dm.U_new[...] = self.U_cv
        self.compute_fluxes(self.F_fp, self.dt)
        self.apply_fluxes(self.dt)
        self.U_cv[...] = self.dm.U_new

    # ----------------------------------------------------------------
    # ADER interface
    # ----------------------------------------------------------------
    def ader_string(self) -> str:
        if self.ndim == 3:
            return "zyx"
        elif self.ndim == 2:
            return "yx"
        else:
            return "x"
    # ----------------------------------------------------------------
    # CFL time step
    # ----------------------------------------------------------------

    def compute_dt(self) -> None:
        W = self.W_cv
        c_s = self.equations.compute_cs(
            W[self._p_], W[self._d_], self.gamma, self.min_c2
        )
        c = c_s * self.ndim
        for vel in self.vels[: self.ndim]:
            c += np.abs(W[vel])
        c_max = np.max(c)
        h = self.h_min
        dt = h / c_max
        dt = self.comms.reduce_min(dt).item()
        if self.viscosity and self.nu > 0:
            dt = min(dt, h**2 / self.nu * 0.25)
        self.dt = self.cfl_coeff * dt
        self._sim.dt = self.dt

    # ----------------------------------------------------------------
    # Boundary handling
    # ----------------------------------------------------------------

    def store_BC(
        self, M: np.ndarray, dim: str, all: bool = True
    ) -> None:
        """Stores boundary layer values from the active region."""
        idim = self.dims[dim]
        ngh = self.Nghc
        BC = self.BC[dim]
        cuts = (
            cut(-2 * ngh, -ngh, idim),
            cut(ngh, 2 * ngh, idim),
        )
        for side in [0, 1]:
            if BC[side] == "periodic":
                self.BC_fp[dim][side] = M[cuts[side]]
            elif BC[side] == "reflective":
                if all:
                    reverse = (Ellipsis, slice(None, None, -1)) + tuple(
                        repeat(slice(None), idim)
                    )
                    self.BC_fp[dim][side] = M[cuts[1 - side]][reverse]
                    self.BC_fp[dim][side][self.vels[idim]] *= -1
            elif BC[side] == "gradfree":
                if all:
                    self.BC_fp[dim][side] = M[cuts[1 - side]]
            elif BC[side] == "ic":
                next
            elif BC[side] == "pressure":
                next
            elif BC[side] == "eq":
                if all:
                    self.BC_fp[dim][side][...] = 0
            elif BC[side] == "doublemach":
                if all:
                    self._store_doublemach_BC(M, dim, side)
            else:
                raise ("Undetermined boundary type")

    # ----------------------------------------------------------------
    # Double Mach Reflection boundary (Woodward & Colella)
    # ----------------------------------------------------------------

    def _dmr_state(self, primitive_state, nd):
        """Build a length-``nvar`` primitive vector broadcast over ``nd`` axes.

        FV boundaries operate on primitive variables.
        """
        xp = self.dm.xp
        vec = xp.zeros(self.nvar)
        vec[self._d_] = primitive_state["rho"]
        vec[self.vels[0]] = primitive_state["vx"]
        vec[self.vels[1]] = primitive_state["vy"]
        vec[self._p_] = primitive_state["P"]
        return vec.reshape((self.nvar,) + (1,) * (nd - 1))

    def _store_doublemach_BC(self, M, dim, side):
        """Fill ``BC_fp`` for the Double Mach Reflection boundary.

        x: left = inflow (post-shock state), right = outflow.
        y: lower = reflecting wall for x >= xc, post-shock state for x < xc;
           upper = the (moving, tilted) shock state for x < x_s(t), ambient
           otherwise.

        The post-shock / ambient states are imposed explicitly (rather than
        relying on frozen IC ghost values) so the BC is self-contained.
        """
        from spd.initial_conditions.initial_conditions_2d import (
            DMR_XC,
            DMR_ANGLE,
            DMR_SHOCK_SPEED,
            dmr_post_shock,
            dmr_ambient,
        )

        xp = self.dm.xp
        ngh = self.Nghc
        idim = self.dims[dim]
        BCfp = self.BC_fp[dim][side]
        nd = BCfp.ndim
        # interior ngh-layer adjacent to this boundary
        cuts = (cut(-2 * ngh, -ngh, idim), cut(ngh, 2 * ngh, idim))
        post = self._dmr_state(dmr_post_shock(self.gamma), nd)

        if dim == "x":
            # Left = inflow (post-shock); right = outflow (zero-gradient copy).
            if side == 0:
                BCfp[...] = post
            else:
                BCfp[...] = M[cuts[1 - side]]
            return

        # dim == "y"
        xc = getattr(self, "dmr_xc", DMR_XC)
        x = xp.asarray(self.centers["x"])  # full x incl. ghosts (last axis of BCfp)
        xb = x.reshape((1,) * (nd - 1) + x.shape)

        if side == 0:
            # Reflecting wall for x >= xc; post-shock inflow for x < xc.
            reverse = (Ellipsis, slice(None, None, -1)) + tuple(
                repeat(slice(None), idim)
            )
            refl = M[cuts[1 - side]][reverse].copy()
            refl[self.vels[idim]] *= -1
            BCfp[...] = xp.where(xb >= xc, refl, post)
        else:
            angle = getattr(self, "dmr_angle", DMR_ANGLE)
            speed = getattr(self, "dmr_shock_speed", DMR_SHOCK_SPEED)
            t = float(self.time)
            y = xp.asarray(self.centers["y"])[-ngh:]  # y of the upper ghost rows
            yb = y.reshape((1,) * (nd - 2) + (ngh, 1))
            x_s = speed * t / np.sin(angle) + xc + yb / np.tan(angle)
            ambient = self._dmr_state(dmr_ambient(self.gamma), nd)
            BCfp[...] = xp.where(xb < x_s, post, ambient)

    def apply_BC(self, dim: str) -> None:
        """Fills ghost cells in M_fv from stored boundary values."""
        ngh = self.Nghc
        idim = self.dims[dim]
        self.dm.M[cut(None, ngh, idim)] = self.BC_fp[dim][0]
        self.dm.M[cut(-ngh, None, idim)] = self.BC_fp[dim][1]

    def Boundaries(self, M: np.ndarray, all=True):
        """Apply boundary conditions to all dimensions."""
        for dim in self.dims:
            self.store_BC(M, dim, all)
            self.Comms(M, dim)
            self.apply_BC(dim)

    def Boundaries_scalar(self, M: np.ndarray):
        """Fill ghost cells for an auxiliary scalar field (e.g. the trouble
        indicator) **without** touching the solution ``BC_fp`` buffer.

        Periodic boundaries wrap around (with MPI halo exchange); every
        physical boundary uses a zero-gradient (copy-interior) extrapolation.

        This is deliberately separate from :meth:`Boundaries`: the solution
        ``BC_fp`` holds the (variable-dependent) boundary *state*, which is
        only meaningful for the full primitive field.  Reusing it for the
        trouble field leaks that state into the flags -- e.g. the Double Mach
        Reflection post-shock density (8) would appear as ``theta``/
        ``affected_faces`` of 8 on boundary faces and make ``correct_fluxes``
        amplify (rather than replace) the high-order flux there.
        """
        ngh = self.Nghc
        for dim in self.dims:
            idim = self.dims[dim]
            BC = self.BC[dim]
            bc = self.BC_fp_scalar[dim]
            cuts = (cut(-2 * ngh, -ngh, idim), cut(ngh, 2 * ngh, idim))
            for side in [0, 1]:
                if BC[side] == "periodic":
                    bc[side] = M[cuts[side]]
                else:
                    # zero-gradient: copy the nearest interior layer
                    bc[side] = M[cuts[1 - side]]
            self.comms.Comms_fv(self.dm, M, self.BC_fp_scalar, idim, dim, ngh)
            M[cut(None, ngh, idim)] = bc[0]
            M[cut(-ngh, None, idim)] = bc[1]

    def Comms(self, M: np.ndarray, dim: str):
        comms = self.comms
        comms.Comms_fv(self.dm, M, self.BC_fp, self.dims[dim], dim, self.Nghc)

    # ----------------------------------------------------------------
    # Mesh helpers
    # ----------------------------------------------------------------

    def interpolate_to_regular_mesh(self, W):
        return W

    def transpose_to_fv(self, W):
        return W

    @property
    def domain_size(self):
        return np.prod(
            [self.N[dim] * self.n[dim] for dim in self.dims]
        )
