"""
Spectral-difference constrained transport for induction (ideal MHD substep).

Extends SD_Scheme with face-centered B, edge E, and custom ADER/RK time coupling.
"""

from __future__ import annotations

import os
import numpy as np

from numerics.polynomials import quadrature_mean
from numerics.slicing import cut, indices, indices2
from spectral_difference.sd_scheme import SD_Scheme


class InductionSD_Scheme(SD_Scheme):
    """
    SD mesh + induction transport. Initializes velocity field from init_fct (ndim
    components) and B from vector potential; stores primitives on ``dm.W_cp``.
    """

    _E_NVAR = 5  # E0, B1, B2, v1, v2 for the edge Riemann state

    def _init_B_from_vector_potential(self) -> None:
        na = np.newaxis
        A_ep = {}
        for dim, idim in zip(self.Edims, self.Eidims):
            x = self.x_sp if dim == "x" else self.x_fp
            dims = [x]
            if self.ndim > 1:
                y = self.y_sp if dim == "y" else self.y_fp
                dims.append(y)
            if self.ndim > 2:
                z = self.z_sp if dim == "z" else self.z_fp
                dims.append(z)
            mesh = self.compute_mesh(dims)
            A = self.vectorpot_fct(mesh, idim)
            A_ep[dim] = self.crop(A[na])[0]

        for dim in self.dims:
            dim1, dim2 = self.other_dims(dim)
            B_fp = self.array_fp(dims=dim)[0]
            B_fp[...] = 0
            if dim1 in self.dims:
                B_fp += self.compute_sp_from_dfp(A_ep[dim2][na], dim1)[0] / self.h[dim1]
            if dim2 in self.dims:
                B_fp -= self.compute_sp_from_dfp(A_ep[dim1][na], dim2)[0] / self.h[dim2]
            self.__setattr__(f"B{dim}_init_fp", B_fp)
            self.dm.__setattr__(f"B{dim}_fp", B_fp.copy())

        self.A_ep = A_ep

    def _sync_W_cp_from_primitives(self) -> None:
        self.dm.W_cp = self.compute_cp_from_sp(self.dm.W_sp)

    def post_init(self) -> None:
        ngh = self.Nghe
        W_gh = self.array_sp(ngh=ngh)
        for var in range(self.nvar):
            W_gh[var] = quadrature_mean(
                self.mesh_cv, self.init_fct, self.ndim, self.p, var
            )
        W_cv = self.crop(W_gh)
        W_sp = self.compute_sp_from_cv(W_cv)
        self.dm.W_cv = W_cv.copy()
        self.dm.W_sp = W_sp.copy()
        self.dm.U_sp = self.compute_conservatives(self.dm.W_sp)
        self.dm.U_cv = self.compute_conservatives(self.dm.W_cv)
        self._init_B_from_vector_potential()
        self._sync_W_cp_from_primitives()

    def allocate_arrays(self, ader=False):
        super().allocate_arrays(ader=ader)
        self._allocate_induction_ader_arrays(ader=ader)

    def _allocate_induction_ader_arrays(self, ader: bool) -> None:
        dims_map = ["yz", "zx", "xy"]
        for dim, idim in zip(self.Edims, self.Eidims):
            E_ep = self.array_fp(dims=dims_map[idim], ader=ader)
            self.dm.__setattr__(f"E{dim}_ader_ep", E_ep)
            for dim2 in self.other_dims(dim):
                self.dm.__setattr__(
                    f"E{dim}L_ep_{dim2}",
                    self._array_RS_induction(dim=dim2, dim2=dim, ader=ader),
                )
                self.dm.__setattr__(
                    f"E{dim}R_ep_{dim2}",
                    self._array_RS_induction(dim=dim2, dim2=dim, ader=ader),
                )
                self.dm.__setattr__(
                    f"BC_E{dim}_ep_{dim2}",
                    self._array_BC_induction(dim=dim2, dim2=dim, ader=ader),
                )
        for dim in self.dims:
            self.dm.__setattr__(
                f"B{dim}_ader_fp", self.array_fp(dims=dim, ader=ader)[0]
            )

    def _array_RS_induction(self, dim="x", dim2=None, ader=False):
        """Riemann buffers for 5-component edge state."""
        shape = [self._E_NVAR, self.nader] if ader else [self._E_NVAR]
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

    def _array_BC_induction(self, dim="x", dim2=None, ader=False):
        shape = [2, self._E_NVAR, self.nader] if ader else [2, self._E_NVAR]
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

    def initialize(self):
        super().initialize()
        self.init_E_Boundaries_sd()

    def induction_b_staging_array(self, dim: str):
        """Scratch buffer matching ``B{dim}_fp`` (RK stage / residual storage)."""
        return self.array_fp(dims=dim)[0]

    def compute_dt(self) -> None:
        W = self.dm.W_cp
        vel = np.abs(W[0]).copy()
        for idim in range(1, self.ndim):
            vel += np.abs(W[idim])
        c_max = np.max(vel)
        h = self.h_min / (self.p + 1)
        dt = h / c_max
        if self.nu > 0:
            dt_nu = (0.25 * self.h_min / (self.p + 1)) ** 2 / self.nu
            dt = min(dt, dt_nu)
        dt = self.comms.reduce_min(dt)
        self.dt = self.cfl_coeff * dt.item()
        self._sim.dt = self.dt

    def create_dicts(self):
        super().create_dicts()
        self.create_induction_dicts()

    def create_induction_dicts(self):
        self.E_ader_ep = {}
        self.B_ader_fp = {}
        self.B_fp = {}
        for dim in self.dims:
            self.B_ader_fp[dim] = self.dm.__getattribute__(f"B{dim}_ader_fp")
            self.B_fp[dim] = self.dm.__getattribute__(f"B{dim}_fp")
        for dim in self.Edims:
            self.E_ader_ep[dim] = self.dm.__getattribute__(f"E{dim}_ader_ep")

        self.EL_ep = {}
        self.ER_ep = {}
        for dim in self.Edims:
            self.EL_ep[dim] = {}
            self.ER_ep[dim] = {}
            for dim2 in self.other_dims(dim):
                self.EL_ep[dim][dim2] = self.dm.__getattribute__(f"E{dim}L_ep_{dim2}")
                self.ER_ep[dim][dim2] = self.dm.__getattribute__(f"E{dim}R_ep_{dim2}")

        self.BC_E_ep = {}
        for dim in self.Edims:
            self.BC_E_ep[dim] = {}
            for dim2 in self.other_dims(dim):
                self.BC_E_ep[dim][dim2] = self.dm.__getattribute__(f"BC_E{dim}_ep_{dim2}")

    def other_dims(self, dim):
        dims = ["yz", "zx", "xy"]
        if dim in self.dims:
            idim = self.dims[dim]
        else:
            idim = 2
        dim1 = dims[idim][0]
        dim2 = dims[idim][1]
        return dim1, dim2

    def ader_string(self) -> str:
        if self.ndim == 3:
            return "zyxkji"
        if self.ndim == 2:
            return "yxji"
        return "xi"

    def _edges_use_ader(self) -> bool:
        return bool(self._sim.ader)

    def ader_dBdt(self, dim):
        ad = self._edges_use_ader()
        dim1, dim2 = self.other_dims(dim)
        dBdt = (
            self.compute_sp_from_dfp(self.E_ader_ep[dim1], dim2, ader=ad)[0]
            / self.h[dim2]
            if dim1 in self.Edims
            else 0
        )
        dBdt -= (
            self.compute_sp_from_dfp(self.E_ader_ep[dim2], dim1, ader=ad)[0]
            / self.h[dim1]
            if dim2 in self.Edims
            else 0
        )
        return dBdt * self.dt

    def dBdt_dim(self, dim):
        """Spatial rhs dB/dt (no dt factor) for RK."""
        dt0 = self.dt
        if dt0 == 0:
            dt0 = 1.0
        return self.ader_dBdt(dim) / dt0

    def ader_predictor(self, prims: bool = False) -> None:
        if not self._sim.ader:
            return
        na = self.dm.xp.newaxis
        for dim in self.dims:
            self.B_ader_fp[dim][...] = self.dm.__getattribute__(f"B{dim}_fp")[na]

        for ader_iter in range(self.m + 1):
            self.solve_edges(ader_iter)
            if self.nu > 0:
                self.add_nabla_terms()
            if ader_iter < self.m:
                for dim in self.dims:
                    s = self.ader_string()
                    self.B_ader_fp[dim] = np.einsum(
                        f"np,p{s}->n{s}", self.dm.invader, self.ader_dBdt(dim)
                    )
                    self.B_ader_fp[dim][...] = self.B_fp[dim][na] - self.B_ader_fp[dim]

    def ader_update(self):
        if not self._sim.ader:
            return
        for dim in self.dims:
            s = self.ader_string()
            dBdt = np.einsum(f"t,t{s}->{s}", self.dm.w_tp, self.ader_dBdt(dim))
            self.B_fp[dim] -= dBdt

    def solve_edges(self, ader_iter):
        na = np.newaxis
        ad = self._edges_use_ader()
        for dim in self.Edims:
            dim1, dim2 = self.other_dims(dim)
            if ad:
                b1s = self.B_ader_fp[dim1][na]
                b2s = self.B_ader_fp[dim2][na]
            else:
                b1s = self.B_fp[dim1][na]
                b2s = self.B_fp[dim2][na]
            B1 = (
                self.compute_fp_from_sp(b1s, dim2, ader=ad)[0]
                if dim1 in self.dims
                else 0
            )
            B2 = (
                self.compute_fp_from_sp(b2s, dim1, ader=ad)[0]
                if dim2 in self.dims
                else 0
            )

            self.fill_E_array(self.E_ader_ep[dim], B1, B2, dim, ader=ad)

            _v1_ = 3
            for d1, d2 in [self.other_dims(dim), self.other_dims(dim)[::-1]]:
                self.E_Boundaries_sd(self.E_ader_ep[dim], dim, d1)
                E = self.E_riemann_solver(
                    self.EL_ep[dim][d1], self.ER_ep[dim][d1], _v1_
                )
                _v1_ += 1
                self.apply_edges(E, self.E_ader_ep[dim], d1)

    def compute_vels(self, dim, dim1, dim2):
        if self.ndim == 3:
            v = self.compute_sp_from_fp(self.dm.W_cp, dim)
        else:
            v = self.dm.W_cp
        v1 = v[self.dims[dim1]]
        v2 = v[self.dims[dim2]]
        return v1, v2

    def fill_E_array(self, E_ep, B1, B2, dim, ader=False):
        dim1, dim2 = self.other_dims(dim)
        v1, v2 = self.compute_vels(dim, dim1, dim2)
        E_ep[0] = v1 * B2 - v2 * B1
        E_ep[1] = B1
        E_ep[2] = B2
        if ader:
            E_ep[3] = v1[np.newaxis]
            E_ep[4] = v2[np.newaxis]
        else:
            E_ep[3] = v1
            E_ep[4] = v2

    def E_riemann_solver(self, EL, ER, _v1_):
        v = np.where(np.abs(EL[_v1_]) > np.abs(ER[_v1_]), EL[_v1_], ER[_v1_])
        return np.where(v >= 0, EL, ER)

    def compute_gradient(self, M_fp, dim):
        return self.compute_sp_from_dfp(M_fp, dim, ader=self._edges_use_ader()) / self.h[
            dim
        ]

    def add_nabla_terms(self):
        na = np.newaxis
        ad = self._edges_use_ader()
        for dim in self.Edims:
            dB_ep = {}
            dims = self.other_dims(dim)
            for dim1, dim2 in [dims, dims[::-1]]:
                bsrc = self.B_ader_fp[dim1][na] if ad else self.B_fp[dim1][na]
                B = self.compute_fp_from_sp(bsrc, dim=dim2, ader=ad)
                self.E_ader_ep[dim][1] = B

                for d in [dim1, dim2]:
                    self.E_Boundaries_sd(self.E_ader_ep[dim], dim, d)
                    self.apply_edges(self.EL_ep[dim][d][1][na], B, d)

                dB_fp = self.compute_gradient(B, dim2)
                dB_ep[dim1] = self.compute_fp_from_sp(dB_fp, dim=dim2, ader=ad)
                self.E_ader_ep[dim][1] = dB_ep[dim1]

                for d in [dim1, dim2]:
                    self.E_Boundaries_sd(self.E_ader_ep[dim], dim, d)
                    self.apply_edges(self.ER_ep[dim][d][1][na], dB_ep[dim1], d)

            self.E_ader_ep[dim][0] -= self.nu * (
                dB_ep[dim1][0] - dB_ep[dim2][0]
            )

    def init_E_Boundaries_sd(self) -> None:
        for dim, idim in zip(self.Edims, self.Eidims):
            dim1, dim2 = self.other_dims(dim)
            na = np.newaxis
            B1 = self.dm.__getattribute__(f"B{dim1}_fp")
            B1 = self.compute_fp_from_sp(B1[na], dim2)[0]
            B2 = self.dm.__getattribute__(f"B{dim2}_fp")
            B2 = self.compute_fp_from_sp(B2[na], dim1)[0]
            E_ep = self.array_fp(dims=dim1 + dim2)
            self.fill_E_array(E_ep, B1, B2, dim)
            for d2 in self.other_dims(dim):
                BC = self.dm.__getattribute__(f"BC_E{dim}_ep_{d2}")
                BC[0][...] = self.E_cut(E_ep[:, np.newaxis], 0, d2, d2)
                BC[1][...] = self.E_cut(E_ep[:, np.newaxis], -1, d2, d2)

    def E_cut(self, E, index, dim1, dim2):
        return E[indices(index, self.ndim + self.dims[dim2])][
            indices(index, self.dims[dim1])
        ]

    def store_edges(self, E: np.ndarray, dim: str, dim1: str) -> None:
        EL = self.EL_ep[dim]
        ER = self.ER_ep[dim]
        shift = self.dims[dim1] + (self.ndim - 1)
        ER[dim1][cut(None, -1, shift)] = E[indices(0, self.dims[dim1])]
        EL[dim1][cut(1, None, shift)] = E[indices(-1, self.dims[dim1])]

    def apply_edges(self, E: np.ndarray, E_ep: np.ndarray, dim1: str):
        shift = self.ndim + self.dims[dim1] - 1
        E_ep[indices(0, self.dims[dim1])] = E[cut(None, -1, shift)]
        E_ep[indices(-1, self.dims[dim1])] = E[cut(1, None, shift)]

    def store_BC(self, BC_array: np.ndarray, M: np.ndarray, dim: str):
        idim = self.dims[dim]
        BC = self.BC[dim]
        for side in [0, 1]:
            if BC[side] == "periodic":
                BC_array[side] = M[indices2(side - 1, self.ndim, idim)]
            elif BC[side] == "reflective":
                BC_array[side] = M[indices2(-side, self.ndim, idim)]
                BC_array[side, self.vels[idim]] *= -1
            elif BC[side] == "gradfree":
                BC_array[side] = M[indices2(-side, self.ndim, idim)]
            elif BC[side] in ("ic", "eq"):
                pass
            elif BC[side] == "pressure":
                M[indices2(-side, self.ndim, idim)] = BC_array[side]
            else:
                raise RuntimeError("Undetermined boundary type")

    def apply_BC(self, dim: str, dim1: str) -> None:
        shift = self.ndim + self.dims[dim1] - 1
        self.EL_ep[dim][dim1][indices(0, shift)] = self.BC_E_ep[dim][dim1][0]
        self.ER_ep[dim][dim1][indices(-1, shift)] = self.BC_E_ep[dim][dim1][1]

    def E_Boundaries_sd(self, M: np.ndarray, dim: str, dim1: str):
        self.store_BC(self.BC_E_ep[dim][dim1], M, dim1)
        self.store_edges(M, dim, dim1)
        self.Comms_ep(M, dim, dim1)
        self.apply_BC(dim, dim1)

    def Comms_ep(self, M: np.ndarray, dim: str, dim1: str):
        self.comms.Comms_sd(
            self.dm,
            M,
            self.BC_E_ep[dim],
            self.dims[dim1],
            dim1,
            self.Nghc,
        )

    def compute_B2(self):
        Bx = self.compute_sp_from_fp(self.dm.Bx_fp[np.newaxis], "x")[0]
        By = (
            self.compute_sp_from_fp(self.dm.By_fp[np.newaxis], "y")[0]
            if self.ndim > 1
            else 0
        )
        Bz = (
            self.compute_sp_from_fp(self.dm.Bz_fp[np.newaxis], "z")[0]
            if self.ndim > 2
            else 0
        )
        B2 = Bx**2 + By**2 + Bz**2
        B2 = self.compute_cv_from_sp(B2[np.newaxis])
        return B2

    def post_update(self):
        pass

    def allocate_ader_extras(self):
        """Hook for ADER_Integrator after U_ader allocation (induction uses B_ader)."""
        pass


class MHDInductionSD_Scheme(InductionSD_Scheme):
    """
    SD + induction for coupled MHD: hydro ``post_init``, 8-component edge state,
    and LLF electric-field Riemann solver.
    """

    _E_NVAR = 8

    def post_init(self) -> None:
        SD_Scheme.post_init(self)
        self._init_B_from_vector_potential()
        self._sync_W_cp_from_primitives()

    def add_nabla_terms(self, ader=True):
        SD_Scheme.add_nabla_terms(self, ader=ader)
        if self.nu > 0:
            InductionSD_Scheme.add_nabla_terms(self)

    def compute_vels(self, dim, dim1, dim2, ader=False):
        sim = self._sim
        if ader:
            Ua = self.dm.U_ader
            W = self.compute_primitives(
                self.compute_fp_from_sp(Ua, dim=dim2, ader=True)
            )
        else:
            W = self.compute_fp_from_sp(self.dm.W_sp, dim=dim1)
            W = self.compute_fp_from_sp(W, dim=dim2)
        v1 = W[sim.vels[sim.dims[dim1]]]
        v2 = W[sim.vels[sim.dims[dim2]]]
        return v1, v2, W[sim.b[dim]], W[sim._d_], W[sim._p_]

    def fill_E_array(self, E_ep, B1, B2, dim, ader=False):
        dim1, dim2 = self.other_dims(dim)
        v1, v2, B3, rho, p = self.compute_vels(dim, dim1, dim2, ader=ader)
        E_ep[0] = v1 * B2 - v2 * B1
        E_ep[1] = B1
        E_ep[2] = B2
        E_ep[3] = v1
        E_ep[4] = v2
        E_ep[5] = B3
        E_ep[6] = rho
        E_ep[7] = p

    def E_riemann_solver(self, EL, ER, _v1_):
        return self.llf_E(EL, ER, _v1_, gamma=self._sim.gamma, min_c2=self._sim.min_c2)

    def llf_E(self, E_L, E_R, vel, *args, **kwargs):
        import mhd

        B1_L = E_L[1]
        B1_R = E_R[1]
        B2_L = E_L[2]
        B2_R = E_R[2]
        B3_L = E_L[5]
        B3_R = E_R[5]
        c_L = mhd.compute_fast_vel(
            E_L[7], E_L[6], B1_L, B2_L, B3_L, **kwargs
        )
        c_R = mhd.compute_fast_vel(
            E_R[7], E_R[6], B1_R, B2_R, B3_R, **kwargs
        )
        Ss = np.maximum(np.abs(E_R[vel]), np.abs(E_L[vel])) + np.maximum(c_R, c_L)

        if vel == 3:
            Es = 0.5 * (E_R + E_L) - 0.5 * Ss * (B2_R - B2_L)
        elif vel == 4:
            Es = 0.5 * (E_R + E_L) + 0.5 * Ss * (B2_R - B2_L)
        else:
            Es = 0.5 * (E_R + E_L)
        return Es
