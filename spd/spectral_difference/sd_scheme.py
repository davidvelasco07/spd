"""
Spectral Difference semi-discrete scheme.

High-order spatial discretization using Lagrange interpolation
between solution points and flux points within elements.
Computes the spatial operator L(U) = -div(F) for the method-of-lines
formulation dU/dt = L(U).
"""

import numpy as np

from schemes.scheme import SemiDiscreteScheme
from numerics.polynomials import (
    solution_points,
    lagrange_matrix,
    lagrangeprime_matrix,
    intfromsol_matrix,
    quadrature_mean,
)

from riemann_solvers.riemann_solver_1D import Riemann_solver_1D as rs1d
from numerics.transforms import compute_A_from_B, compute_A_from_B_full
from numerics.slicing import indices2
from spectral_difference import sd_boundary as bc
from numerics.polynomials import gauss_legendre_quadrature, flux_points, solution_points 

class SD_Scheme(SemiDiscreteScheme):
    """
    Spectral Difference semi-discrete spatial scheme.

    Discretizes space using high-order Lagrange polynomial bases
    defined on solution points and flux points within each element.

    Parameters
    ----------
    sim : Simulator
        Parent simulator providing shared state.
    riemann_solver_sd : str
        Name of the Riemann solver ('llf', 'hllc', 'lhllc').
    """

    def __init__(self, sim, riemann_solver="llf", soe="hydro"):
        super().__init__(sim)
        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, self.p)
        sp = solution_points(0.0, 1.0, self.p)
        fp = flux_points(0.0, 1.0, self.p)

        for name in ["sp", "fp"]:
            self.__setattr__(name, {})
        for dim in self.dims:
            self.__setattr__(f"{dim}_sp", sp)
            self.sp[dim] = self.__getattribute__(f"{dim}_sp")
            self.__setattr__(f"{dim}_fp", fp)
            self.fp[dim] = self.__getattribute__(f"{dim}_fp")

        self.faces = {}
        self.centers = {}
        self.h_fp = {}
        self.h_cv = {}
        self.riemann_solver = rs1d(riemann_solver, soe).solver

        # Lagrange matrices for interpolation between bases
        self.dm.sp_to_fp = lagrange_matrix(self.fp["x"], self.sp["x"])
        self.dm.fp_to_sp = lagrange_matrix(self.sp["x"], self.fp["x"])
        # Spatial derivative of the flux at sol pts from density at flux pts.
        self.dm.dfp_to_sp = lagrangeprime_matrix(self.sp["x"], self.fp["x"])
        # Mean values in control volumes from values at sol pts.
        self.dm.sp_to_cv = intfromsol_matrix(self.sp["x"], self.fp["x"])
        self.dm.fp_to_cv = intfromsol_matrix(self.fp["x"], self.fp["x"])
        self.dm.cv_to_sp = np.linalg.inv(self.dm.sp_to_cv)
        self.scheme = "FE_SD"

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        """Set up SD arrays, compute initial conditions, allocate integrator arrays."""
        self.mesh_cv = self.compute_mesh_cv()
        self.compute_positions()
        self.post_init()
        self.allocate_arrays(ader=self.ader)
        self.init_Boundaries()
        self.create_dicts()
        self.compute_dt()
        if self.potential:
            self.init_potential()
        if self.WB:
            self.init_equilibrium_state()

    def compute_positions(self):
        na = np.newaxis
        ngh = self.Nghc
        for dim in self.dims:
            idim = self.dims[dim]
            # Solution points
            sp = self.lim[dim][0] + (
                np.arange(self.N[dim])[:, na] + self.sp[dim][na, :]
            ) * self.h[dim]
            self.dm.__setattr__(
                f"{dim.upper()}_sp",
                sp.reshape(self.N[dim], self.n[dim]),
            )
            # Flux points
            fp = np.ndarray((self.N[dim] * self.n[dim] + ngh * 2 + 1))
            fp[ngh:-ngh] = self.h[dim] * np.hstack(
                (
                    np.arange(self.N[dim]).repeat(self.n[dim])
                    + np.tile(self.fp[dim][:-1], self.N[dim]),
                    self.N[dim],
                )
            )
            fp[:ngh] = -fp[(ngh + 1) : (2 * ngh + 1)][::-1]
            fp[-ngh:] = fp[-(ngh + 1)] + fp[ngh + 1 : 2 * ngh + 1]
            self.dm.__setattr__(f"{dim.upper()}_fp", fp)
            self.faces[dim] = fp
            # Cell centers
            cv = 0.5 * (fp[1:] + fp[:-1])
            self.dm.__setattr__(f"{dim.upper()}_cv", cv)
            self.centers[dim] = cv
            # Distance between faces
            h_fp = (fp[1:] - fp[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_fp", h_fp)
            self.h_fp[dim] = h_fp
            # Distance between centers
            h_cv = (cv[1:] - cv[:-1])[self.shape(idim)]
            self.dm.__setattr__(f"d{dim}_cv", h_cv)
            self.h_cv[dim] = h_cv

    def compute_mesh_cv(self) -> np.ndarray:
        """
        Compute the mesh of the control volumes.    
        """
        Nghe = self.Nghe
        Ns = [self.N[dim] + 2 * Nghe for dim in self.dims]
        shape = (self.ndim,) + tuple(Ns[::-1]) + (self.p + 2,) * self.ndim
        mesh_cv = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h = self.h[dim]
            lenght = self.len[dim] + 2 * Nghe * h
            shape1 = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (self.ndim + idim)
            )
            shape2 = (
                (None,) * (2 * self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (idim)
            )
            mesh_cv[idim] = (
                self.lim[dim][0]
                + (np.arange(N)[shape1] + self.fp[dim][shape2]) * lenght / N
                - h
            )
        return mesh_cv

    def compute_mesh(self, Points) -> np.ndarray:   
        """
        Physical coordinates at arbitrary reference points for every SD cell.
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
        Nghe = self.Nghe
        Ns = [self.N[dim] + 2 * Nghe for dim in self.dims]
        shape = (self.ndim,) + tuple(Ns[::-1])
        for points in Points[::-1]:
            shape += (points.size,)
        mesh = np.ndarray(shape)
        for dim in self.dims:
            idim = self.dims[dim]
            N = Ns[idim]
            h = self.h[dim]
            lenght = self.len[dim] + 2 * Nghe * h
            shape1 = (
                (None,) * (self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (self.ndim + idim)
            )
            shape2 = (
                (None,) * (2 * self.ndim - 1 - idim)
                + (slice(None),)
                + (None,) * (idim)
            )
            mesh[idim] = (
                self.lim[dim][0]
                + (np.arange(N)[shape1] + Points[idim][shape2]) * lenght / N
                - h
            )
        return mesh

    def post_init(self) -> None:
        nvar = self.nvar
        ngh = self.Nghe
        # This arrays contain Nghe layers of ghost elements
        W_gh = self.array_sp(ngh=ngh)
        for var in range(nvar):
            W_gh[var][...] = quadrature_mean(self.mesh_cv, self.init_fct, self.ndim, self.p, var)

        self.W_gh = W_gh
        self.W_init_cv = self.active_region(W_gh)
        self.dm.W_cv = self.W_init_cv.copy()
        self.dm.W_sp = self.compute_sp_from_cv(self.dm.W_cv)
        self.dm.U_sp = self.compute_conservatives(self.dm.W_sp)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)

    def init_Boundaries(self) -> None:
        ndim = self.ndim
        for dim in self.dims:
            idim = self.dims[dim]
            BC = self.dm.__getattribute__(f"BC_fp_{dim}")
            M_fp = self.compute_fp_from_sp(self.dm.U_sp, dim)
            if self._sim.integrator.ader:
                BC[0][...] = M_fp[:, np.newaxis][indices2(0, ndim, idim)]
                BC[1][...] = M_fp[:, np.newaxis][indices2(-1, ndim, idim)]
            else:
                BC[0][...] = M_fp[indices2(0, ndim, idim)]
                BC[1][...] = M_fp[indices2(-1, ndim, idim)]

    # ----------------------------------------------------------------
    # Array allocation
    # ----------------------------------------------------------------

    def allocate_arrays(self, ader=False):
        super().allocate_arrays(ader)
        
    def array(self, px, py, pz, ngh=0, ader=False, nvar=None) -> np.ndarray:
        if type(nvar) == type(None):
            nvar = self.nvar
        shape = [nvar, self.nader] if ader else [nvar]
        N = []
        for dim in self.dims:
            N.append(self.N[dim] + 2 * ngh)
        N = N[::-1]
        p = [px, py, pz][: self.ndim][::-1]
        return np.ndarray(shape + N + p)

    def array_sp(self, **kwargs):
        p = self.p
        return self.array(p + 1, p + 1, p + 1, **kwargs)

    def array_fp(self, dims="xyz", **kwargs):
        p = self.p
        return self.array(
            (p + 1 + ("x" in dims)),
            (p + 1 + ("y" in dims)),
            (p + 1 + ("z" in dims)),
            **kwargs,
        )

    def array_RS(self, dim="x", dim2=None, ader=False) -> np.ndarray:
        shape = [self.nvar, self.nader] if ader else [self.nvar]
        N = []
        for odim in self.dims:
            N.append(self.N[odim] + (odim == dim))
        shape += N[::-1]
        if self.ndim > 2:
            if (dim2 == "x") or (dim2 == "y" and dim == "x"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        if self.ndim > 1:
            if (dim2 == "z") or (dim2 == "y" and dim == "z"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        return np.ndarray(shape)

    def array_BC(self, dim="x", dim2=None, ader=False) -> np.ndarray:
        shape = [2, self.nvar, self.nader] if ader else [2, self.nvar]
        if self.Z:
            if dim == "x" or dim == "y":
                shape += [self.N["z"]]
        if self.Y:
            if dim == "x" or dim == "z":
                shape += [self.N["y"]]
        if dim == "y" or dim == "z":
            shape += [self.N["x"]]
        if self.Z:
            if dim2 == "x" or (dim2 == "y" and dim == "x"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        if self.Y:
            if dim2 == "z" or (dim2 == "y" and dim == "z"):
                shape += [self.p + 2]
            else:
                shape += [self.p + 1]
        return np.ndarray(shape)

    def active_region(self, W, ngh=0) -> np.ndarray:
        ngh = self.Nghe 
        return W[(Ellipsis,) + (slice(ngh, -ngh),) * self.ndim + (slice(None),)*self.ndim]

    # ----------------------------------------------------------------
    # Dictionary creation
    # ----------------------------------------------------------------

    def create_dicts(self):
        """
        Creates dictionaries for flux-point arrays so downstream methods
        can use dimension-keyed access (e.g., self.M_fp[dim]).
        """
        names = ["M_fp", "F_fp", "MR_fp", "ML_fp", "BC_fp"]
        for name in names:
            self.__setattr__(name, {})
            for dim in self.dims:
                self.__getattribute__(name)[dim] = self.dm.__getattribute__(
                    f"{name}_{dim}"
                )
        for dim in self.dims:
            self.faces[dim] = self.dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = self.dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = self.dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = self.dm.__getattribute__(f"d{dim}_cv")

    # ----------------------------------------------------------------
    # Transforms between point sets
    # ----------------------------------------------------------------

    def compute_sp_from_cv(self, M_cv) -> np.ndarray:
        return compute_A_from_B_full(M_cv, self.dm.cv_to_sp, self.ndim)

    def compute_cv_from_sp(self, M_sp) -> np.ndarray:
        return compute_A_from_B_full(M_sp, self.dm.sp_to_cv, self.ndim)

    def compute_cp_from_sp(self, M_sp) -> np.ndarray:
        return compute_A_from_B_full(M_sp, self.dm.sp_to_fp, self.ndim)

    def compute_sp_from_cp(self, M_cp) -> np.ndarray:
        return compute_A_from_B_full(M_cp, self.dm.fp_to_sp, self.ndim)

    def compute_sp_from_fp(self, M_fp, dim, **kwargs) -> np.ndarray:
        return compute_A_from_B(M_fp, self.dm.fp_to_sp, dim, self.ndim, **kwargs)

    def compute_fp_from_sp(self, M_sp, dim, **kwargs) -> np.ndarray:
        return compute_A_from_B(M_sp, self.dm.sp_to_fp, dim, self.ndim, **kwargs)

    def compute_sp_from_dfp(self, M_fp, dim, **kwargs) -> np.ndarray:
        return compute_A_from_B(M_fp, self.dm.dfp_to_sp, dim, self.ndim, **kwargs)

    def compute_sp_from_dfp_x(self, Fx, ader=True):
        return self.compute_sp_from_dfp(Fx, "x", ader=ader) / self.h["x"]

    def compute_sp_from_dfp_y(self, Fy, ader=True):
        return self.compute_sp_from_dfp(Fy, "y", ader=ader) / self.h["y"]

    def compute_sp_from_dfp_z(self, Fz, ader=True):
        return self.compute_sp_from_dfp(Fz, "z", ader=ader) / self.h["z"]

    def integrate_faces(self, M_fp, dim, ader=True):
        for other_dim in self.dims:
            if dim != other_dim:
                M_fp = compute_A_from_B(
                    M_fp, self.dm.sp_to_cv, other_dim, self.ndim, ader=ader
                )
        return M_fp

    def compute_gradient(self, M_fp, dim, ader=True):
        return self.compute_sp_from_dfp(M_fp, dim, ader=ader) / self.h[dim]

    # ----------------------------------------------------------------
    # Mesh conversions
    # ----------------------------------------------------------------

    def transpose_to_fv(self, M):
        """Transpose SD element-based array to FV cell-based layout."""
        # nvar,Nz,Ny,Nx,nz,ny,nx → nvar,Nznz,Nyny,Nxnx
        if self.ndim == 1:
            assert M.ndim == 3
            return M.reshape(M.shape[0], M.shape[1] * M.shape[2])
        elif self.ndim == 2:
            assert M.ndim == 5
            return np.transpose(M, (0, 1, 3, 2, 4)).reshape(
                M.shape[0], M.shape[1] * M.shape[3], M.shape[2] * M.shape[4]
            )
        else:
            assert M.ndim == 7
            return np.transpose(M, (0, 1, 4, 2, 5, 3, 6)).reshape(
                M.shape[0],
                M.shape[1] * M.shape[4],
                M.shape[2] * M.shape[5],
                M.shape[3] * M.shape[6],
            )

    def transpose_to_sd(self, M):
        """Transpose FV cell-based array to SD element-based layout."""
        # nvar,Nznz,Nyny,Nxnx → nvar,Nz,Ny,Nx,nz,ny,nx
        shape = []
        for dim in self.dims:
            shape += [self.n[dim], self.N[dim]]
        shape = [M.shape[0]] + shape[::-1]
        if self.ndim == 1:
            return M.reshape(shape)
        elif self.ndim == 2:
            return np.transpose(M.reshape(shape), (0, 1, 3, 2, 4))
        else:
            return np.transpose(M.reshape(shape), (0, 1, 3, 5, 2, 4, 6))

    def interpolate_to_regular_mesh(self, W):
        """Interpolate solution to a regular mesh."""
        p = self.p
        if p <= 1:
            return W
        x = np.arange(p + 2) / (p + 1)
        x = 0.5 * (x[1:] + x[:-1])
        x_sp = solution_points(0.0, 1.0, p)
        m = lagrange_matrix(x, x_sp)
        W_r = compute_A_from_B_full(W, m, self.ndim)
        return W_r

    # ----------------------------------------------------------------
    # Primitives / equilibrium helpers
    # ----------------------------------------------------------------

    def compute_primitives_cv(self, U) -> np.ndarray:
        if self.WB:
            return self.compute_primitives(
                U + self.dm.U_eq_cv
            ) - self.compute_primitives(self.dm.U_eq_cv)
        else:
            return self.compute_primitives(U)

    # ----------------------------------------------------------------
    # CFL time step
    # ----------------------------------------------------------------

    def compute_dt(self) -> None:
        W = self.dm.W_cv
        c_s = self.equations.compute_cs(
            W[self._p_], W[self._d_], self.gamma, self.min_c2
        )
        c = c_s * self.ndim
        for vel in self.vels[: self.ndim]:
            c += np.abs(W[vel])
        c_max = np.max(c)
        h = self.h_min / (self.p + 1)
        dt = h / c_max
        dt = self.comms.reduce_min(dt).item()
        if self.viscosity and self.nu > 0:
            dt = min(dt, h**2 / self.nu * 0.25)
        self.dt = self.cfl_coeff * dt
        self._sim.dt = self.dt

    # ----------------------------------------------------------------
    # Spatial operator: flux computation and RHS
    # ----------------------------------------------------------------

    def solve_faces(self, M, prims=False, ader=True) -> None:
        """Compute fluxes at faces by interpolating to flux points,
        computing physical fluxes, and solving the Riemann problem."""
        na = np.newaxis
        for key in self.idims:
            dim = self.idims[key]
            vels = np.roll(self.vels, -key)
            #Interpolate to flux points
            self.M_fp[dim][...] = self.compute_fp_from_sp(M, dim, ader=ader)
            #Add equilibrium values
            if self.WB:
                # U' -> U
                self.M_fp[dim] += self.dm.__getattribute__(f"M_eq_fp_{dim}")[:, na]
            bc.Boundaries(self, self.M_fp[dim], dim)
            self.compute_fluxes(self.F_fp[dim], self.M_fp[dim], vels, prims)
            bc.store_interfaces(self, self.M_fp[dim], dim)
            F = self.riemann_solver(
                self.ML_fp[dim],
                self.MR_fp[dim],
                self.MR_fp[dim],
                vels,
                self._p_,
                self.gamma,
                self.min_c2,
                prims,
                npassive=self.npassive,
                thdiffusion=self.thdiffusion,
                _t_=self._t_,
            )
            bc.apply_interfaces(self, F, self.F_fp[dim], dim)
            if self.WB:
                # F -> F'
                self.F_fp[dim] -= self.dm.__getattribute__(f"F_eq_fp_{dim}")[:, na]

        if self.viscosity or self.thdiffusion:
            self.add_nabla_terms()

    def add_nabla_terms(self, ader=True):
        """Add viscous and thermal diffusion fluxes."""
        dW_sp = {}
        for dim in self.dims:
            idim = self.dims[dim]
            self.M_fp[dim][...] = self.compute_primitives(self.M_fp[dim])
            bc.Boundaries_sd(self, self.M_fp[dim], dim)
            M = self.ML_fp[dim]
            bc.apply_interfaces(self, M, self.M_fp[dim], dim)
            dW_sp[idim] = self.compute_gradient(self.M_fp[dim], dim, ader=ader)
        dW_fp = {}
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels, -idim)
            idims = self.idims if self.viscosity else [idim]
            for idim in idims:
                dW_fp[idim] = self.compute_fp_from_sp(
                    dW_sp[idim], dim, ader=ader
                )
                bc.Boundaries_sd(self, dW_fp[idim], dim)
                dW = self.MR_fp[dim]
                bc.apply_interfaces(self, dW, dW_fp[idim], dim)
            self.F_fp[dim][...] -= self.compute_viscous_fluxes(
                self.M_fp[dim], dW_fp, vels, prims=True
            )
            if self.thdiffusion:
                self.F_fp[dim][self._p_] -= self.compute_thermal_fluxes(
                    self.M_fp[dim], dW_fp[self.dims[dim]], prims=True
                )

    def compute_dudt(self, U, ader=False) -> np.ndarray:
        """Compute dU/dt from flux divergence at solution points."""
        dUdt = self.compute_sp_from_dfp_x(self.dm.F_fp_x, ader=ader)
        if self.Y:
            dUdt += self.compute_sp_from_dfp_y(self.dm.F_fp_y, ader=ader)
        if self.Z:
            dUdt += self.compute_sp_from_dfp_z(self.dm.F_fp_z, ader=ader)
        if self.potential:
            grad_phi = (
                self.dm.grad_phi_sp[:, np.newaxis]
                if ader
                else self.dm.grad_phi_sp
            )
            self.apply_potential(dUdt, U, grad_phi)
        return dUdt

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Compute the spatial RHS: solve faces then compute dU/dt."""
        self.solve_faces(U, prims=prims, ader=ader)
        return self.compute_dudt(U, ader=ader)

    # ----------------------------------------------------------------
    # Solution state management
    # ----------------------------------------------------------------

    def get_solution(self, ader=False):
        return self.dm.U_sp[:, np.newaxis, ...] if ader else self.dm.U_sp

    def set_solution(self, U):
        self.dm.U_sp[...] = U

    def update_solution(self, dU):
        self.dm.U_sp -= dU
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)

    def convert_solution(self, W=False):
        if W:
            self.dm.U_cv[...] = self.compute_conservatives(self.dm.W_cv)
            self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)
        else:
            self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
            self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)

    def post_update(self):
        """Called after time integrator step: update primitives."""
        self.compute_primitives(self.dm.U_cv, W=self.dm.W_cv)

    # ----------------------------------------------------------------
    # ADER time integration interface
    # ----------------------------------------------------------------
    def ader_string(self) -> str:
        if self.ndim == 3:
            return "zyxkji"
        elif self.ndim == 2:
            return "yxji"
        else:
            return "xi"

    # ----------------------------------------------------------------
    # SD ↔ FV representation switching
    # ----------------------------------------------------------------

    def switch_to_finite_volume(self, integrate_faces=True, ader=False):
        """Convert SD element-based arrays to FV cell-based layout."""
        # Compute control volume averages
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        # Change to Finite Volume layout
        self.dm.U_cv = self.transpose_to_fv(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_fv(self.dm.W_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_fv(self.dm.U_eq_cv)
        if integrate_faces:
            for dim in self.dims:
                self.F_fp[dim][...] = self.integrate_faces(
                    self.F_fp[dim], dim, ader=ader
                )

    def switch_to_high_order(self):
        """Convert FV cell-based arrays back to SD element-based layout."""
        self.dm.U_cv = self.transpose_to_sd(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_sd(self.dm.W_cv)
        if self.WB:
            self.dm.U_eq_cv = self.transpose_to_sd(self.dm.U_eq_cv)
        # Compute solution at solution points
        self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)

    # ----------------------------------------------------------------
    # Potential and equilibrium
    # ----------------------------------------------------------------

    def init_potential(self) -> None:
        phi_cv = quadrature_mean(
            self.mesh_cv, self.init_fct, self.ndim, self.p, -1
        )
        phi_sp = self.compute_sp_from_cv(phi_cv[None])
        self.dm.grad_phi_sp = self.array_sp()[: self.ndim]
        for dim in self.dims:
            idim = self.dims[dim]
            phi_fp = self.compute_fp_from_sp(phi_sp, dim)
            self.dm.grad_phi_sp[idim] = self.crop(
                self.compute_sp_from_dfp(phi_fp, dim)
            ) / self.h[dim]

    def init_equilibrium_state(self) -> None:
        W_sp = self.compute_sp_from_cv(self.dm.W_cv)
        U_sp = self.compute_conservatives(W_sp)
        self.dm.U_eq_sp = self.crop(U_sp)
        self.dm.U_eq_cv = self.compute_cv_from_sp(self.dm.U_eq_sp)
        for dim in self.dims:
            idim = self.dims[dim]
            vels = np.roll(self.vels, -idim)
            M_fp = self.dm.__getattribute__(f"M_eq_fp_{dim}")

    # ----------------------------------------------------------------
    # Communication
    # ----------------------------------------------------------------

    def Comms_fp(self, M: np.ndarray, dim: str):
        comms = self.comms
        comms.Comms_sd(
            self.dm, M, self.BC_fp, self.dims[dim], dim, self.Nghc
        )

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def domain_size(self):
        return np.prod([self.N[dim] * (self.p + 1) for dim in self.dims])
