"""
Coupled fluid + constrained-transport finite-volume scheme for ideal MHD.

``MHD_FV_Scheme`` is the FV counterpart of :class:`MHD_SD_Scheme`: the hydro
state (cell-averaged ``U_cv`` with cell-centered B rows) advances with
MUSCL / MUSCL-Hancock fluxes of the ideal-MHD Riemann solver, while the
face-staggered magnetic field of :class:`InductionFV_Scheme` advances with a
constrained-transport update from a four-state corner electric field.

The corner E-field is resolved by :func:`four_state_E` -- the same
construction (and the same numerics) used by ``MHDFallbackScheme`` /
``InductionFallbackScheme`` for its low-order edge E levels, which borrow it
from this module: MUSCL-reconstructed corner states with RAMSES-style
positivity guards, resolved by a four-state LLF bound (fast magnetosonic
speeds) or by two one-dimensional HLLD sweeps.

The corner states are gathered from the same ghosted primitives as the
fluid fluxes (``dm.M``); with MUSCL-Hancock this is the time-centered
Hancock-predicted state, so a single forward-Euler step (``rk1``) yields the
classic second-order MUSCL-Hancock CT scheme.
"""

import numpy as np

from spd.induction.induction_fv_scheme import InductionFV_Scheme
from spd.finite_volume.fv_scheme import FV_Scheme
from spd.finite_volume import muscl
from spd.numerics.slicing import cut


def four_state_E(corner, gamma, min_c2, use_hlld=False, xp=np):
    """Resolve the CT electric field at edge/corner points from four
    corner states.

    ``corner`` is a callable ``(o1, o2) -> (v1, v2, B1, B2, B3, rho, p)``
    returning the state reconstructed towards the corner from the cell on
    side ``o1`` (transverse direction 1) and ``o2`` (transverse direction
    2), with offsets in (-1, 0).

    With the code convention ``E = v1*B2 - v2*B1``:

    - LLF (default): ``E = 1/4 sum_s E_s - S1/2 <dB2>_1 + S2/2 <dB1>_2``
      with the averaged tangential-field jumps across each transverse
      direction and ``S = max_s(|v| + c)`` a fast-magnetosonic bound.
    - HLLD: two successive one-dimensional HLLD sweeps (direction 1, then
      direction 2) on the 8-component edge state, mirroring the SD edge
      solver.
    """
    if use_hlld:
        from spd.MHD.riemann_solver import hlld_E

        def state(o1, o2):
            v1, v2, B1, B2, B3, rho, p = corner(o1, o2)
            s = xp.empty((8,) + v1.shape, dtype=v1.dtype)
            s[0] = v1 * B2 - v2 * B1
            s[1], s[2] = B1, B2
            s[3], s[4] = v1, v2
            s[5], s[6], s[7] = B3, rho, p
            return s

        # Sweep along direction 1 for each side of direction 2, then across.
        res = {
            o2: hlld_E(
                state(-1, o2), state(0, o2), 3, gamma=gamma, min_c2=min_c2
            )
            for o2 in (-1, 0)
        }
        return hlld_E(res[-1], res[0], 4, gamma=gamma, min_c2=min_c2)[0]

    E_sum = 0.0
    Sp1 = Sp2 = None
    dB2_1 = 0.0
    dB1_2 = 0.0
    for o1 in (-1, 0):
        for o2 in (-1, 0):
            v1, v2, B1, B2, B3, rho, p = corner(o1, o2)
            E_sum = E_sum + (v1 * B2 - v2 * B1)
            # Fastest magnetosonic bound: sqrt(a^2 + B^2/rho).
            c = np.sqrt((gamma * p + B1 * B1 + B2 * B2 + B3 * B3) / rho)
            s1 = np.abs(v1) + c
            s2 = np.abs(v2) + c
            Sp1 = s1 if Sp1 is None else np.maximum(Sp1, s1)
            Sp2 = s2 if Sp2 is None else np.maximum(Sp2, s2)
            jsgn1 = 1.0 if o1 == 0 else -1.0
            jsgn2 = 1.0 if o2 == 0 else -1.0
            dB2_1 = dB2_1 + 0.5 * jsgn1 * B2
            dB1_2 = dB1_2 + 0.5 * jsgn2 * B1
    return 0.25 * E_sum - 0.5 * Sp1 * dB2_1 + 0.5 * Sp2 * dB1_2


class MHD_FV_Scheme(InductionFV_Scheme):
    """
    FV scheme for ideal MHD: MUSCL(-Hancock) fluid fluxes + face-staggered
    constrained transport with a four-state corner E-field, and a fast
    magnetosonic CFL condition.
    """

    _E_NVAR = 8

    # ----------------------------------------------------------------
    # Life cycle
    # ----------------------------------------------------------------

    def initialize(self) -> None:
        if self._sim.ader:
            raise NotImplementedError(
                "MHD_FV_Scheme has no coupled ADER predictor; use a "
                "Runge-Kutta integrator (time_integrator='rk1' with "
                "fallback='MUSCL-Hancock' gives the classic second-order "
                "MUSCL-Hancock CT scheme)."
            )
        if self.ndim < 2:
            raise NotImplementedError(
                "Constrained transport is degenerate in 1D; use the "
                "SD-based MHD scheme (scheme='SD' or 'SDFB') instead."
            )
        self.compute_positions()
        self.mesh_cv = self.compute_mesh_cv()
        self.post_init()
        self.allocate_arrays(ader=self.ader)
        self._init_B_from_vector_potential_fv()
        self.init_B_fp_boundaries()
        self._sync_b_fp_halos()
        # CT-consistent cell-centered B rows, then re-derive the primitives
        # (pressure) from the projected conservative state.
        self.B_to_U()
        self.compute_primitives(self.dm.U_cv, W=self.dm.W_cv)
        self.active_region(self.dm.W_gh)[...] = self.dm.W_cv
        self._sync_W_cp_from_cv()
        self.init_Boundaries()
        self.create_dicts()
        self.compute_dt()

    # The scheme-level dt aliases the simulator's (the CT update scales by
    # dt; a private copy would go stale when the simulator clamps the step).
    @property
    def dt(self):
        return self._sim.dt

    @dt.setter
    def dt(self, value):
        self._sim.dt = value

    def compute_dt(self) -> None:
        """CFL condition on the fast magnetosonic speed (summed over dims);
        mirrors :meth:`MHD_SD_Scheme.compute_dt` on the FV cell width."""
        sim = self._sim
        W = self.W_cv
        c_max = 0
        for dim in self.dims:
            dim1, dim2 = self.other_dims(dim)
            c_max = c_max + self.equations.compute_fast_vel(
                W[sim._p_],
                W[sim._d_],
                W[sim.b[dim]],
                W[sim.b[dim1]] if dim1 in self.dims else 0,
                W[sim.b[dim2]] if dim2 in self.dims else 0,
                self.gamma,
                self.min_c2,
            )
        c_max = np.max(c_max)
        dt = self.h_min / c_max
        dt = self.comms.reduce_min(dt).item()
        if self.viscosity and self.nu > 0:
            nu = max(self.nu, self.chi)
            dt = min(dt, (0.25 * self.h_min) ** 2 / nu)
        self.dt = self.cfl_coeff * dt

    # ----------------------------------------------------------------
    # Face B <-> cell-centered B rows
    # ----------------------------------------------------------------

    def B_to_U(self):
        """Overwrite the cell-centered B rows of the conserved/primitive
        state with the average of the (divergence-free) face-staggered CT
        field."""
        sim = self._sim
        na = np.newaxis
        self._sync_b_fp_halos()
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            Bc = self.compute_sp_from_fp(B[na], dim=dim)[0]
            b = sim.b[dim]
            self.dm.W_gh[b] = Bc
            Bc_act = self.active_region(Bc)
            self.dm.W_cv[b] = Bc_act
            self.dm.U_cv[b] = Bc_act

    def set_stage_state(self, U_stage):
        """Synchronize with an RK stage: the B rows of ``U_stage`` are the
        cell averages of the (divergence-free) stage face field already
        written into ``B{dim}_fp``."""
        sim = self._sim
        na = np.newaxis
        for dim in self.dims:
            B = self.dm.__getattribute__(f"B{dim}_fp")
            Bc = self.compute_sp_from_fp(B[na], dim=dim)[0]
            U_stage[sim.b[dim]] = self.active_region(Bc)

    # ----------------------------------------------------------------
    # MUSCL-Hancock predictor with MHD source terms
    # ----------------------------------------------------------------

    def compute_prediction(self, W, dWs):
        """MUSCL-Hancock half-step prediction with the MHD source terms
        (magnetic pressure/tension in the momenta, primitive induction for
        the transverse B components)."""
        sim = self._sim
        muscl.compute_prediction_mhd(
            W,
            dWs,
            self.dm.dtM,
            self.vels,
            self.ndim,
            self.gamma,
            self._d_,
            self._p_,
            [sim.b[dim] for dim in "xyz"],
            self.WB,
            self.npassive,
        )

    # ----------------------------------------------------------------
    # Corner (edge) electric field
    # ----------------------------------------------------------------

    def _corner_gather(self, A, var, dim, dim1, dim2, o1, o2):
        """Value of row ``var`` of the ghosted array ``A`` at the cell
        displaced by (``o1``, ``o2``) from every active corner of the edge
        lattice of E-family ``dim`` (active faces along ``dim1``/``dim2``,
        active cells along ``dim``)."""
        gh = self.Nghc
        sl = []
        for d in reversed(list(self.dims)):
            if d == dim1:
                o = o1
            elif d == dim2:
                o = o2
            else:
                sl.append(slice(gh, -gh))
                continue
            Nc = self.N[d] * self.n[d]
            sl.append(slice(gh + o, gh + o + Nc + 1))
        return A[(var,) + tuple(sl)]

    def solve_edges(self, ader_iter):
        """Four-state corner E-field on the edge lattice from the ghosted
        primitives of the current flux pass (``dm.M``; Hancock-predicted,
        i.e. time-centered, for MUSCL-Hancock).  Corner states are MUSCL
        reconstructions with the fluid slope limiter and RAMSES-style
        positivity guards; the resolution (LLF fast-speed bound or two
        HLLD sweeps) follows the scheme's Riemann solver.

        ``ader_iter`` is accepted for API parity with the SD driver."""
        _ = ader_iter
        sim = self._sim
        xp = self.dm.xp
        W = self.dm.M
        gh = self.Nghc
        use_hlld = self.riemann_solver_name == "hlld"
        min_rho = getattr(self, "min_rho", 1e-12)
        min_P = getattr(self, "min_P", 1e-12)

        for dim in self.Edims:
            dim1, dim2 = self.other_dims(dim)
            if dim1 not in self.dims or dim2 not in self.dims:
                continue
            i1, i2 = self.dims[dim1], self.dims[dim2]
            S1 = xp.zeros_like(W)
            S2 = xp.zeros_like(W)
            S1[cut(1, -1, i1)] = self.compute_slopes(W, i1)
            S2[cut(1, -1, i2)] = self.compute_slopes(W, i2)
            r_v1 = sim.vels[i1]
            r_v2 = sim.vels[i2]

            def corner(o1, o2):
                gather = lambda A, var: self._corner_gather(
                    A, var, dim, dim1, dim2, o1, o2
                )
                # The corner is on the opposite side of each offset:
                # reconstruct towards it.
                sgn1 = 1.0 if o1 == -1 else -1.0
                sgn2 = 1.0 if o2 == -1 else -1.0
                g = lambda var: (
                    gather(W, var)
                    + sgn1 * gather(S1, var)
                    + sgn2 * gather(S2, var)
                )
                # Positivity guard (RAMSES-style): revert to the donor-cell
                # value where the reconstruction is at/below the floor.
                def g_pos(var, floor):
                    rec = g(var)
                    return xp.where(rec > floor, rec, gather(W, var))
                return (
                    g(r_v1), g(r_v2),
                    g(sim.b[dim1]), g(sim.b[dim2]), g(sim.b[dim]),
                    g_pos(sim._d_, min_rho),
                    g_pos(sim._p_, min_P),
                )

            E_ep = self.E_ader_ep[dim]
            E_ep[...] = 0.0
            E_act = four_state_E(
                corner, self.gamma, sim.min_c2, use_hlld=use_hlld, xp=xp
            )
            sl = [slice(None)] * (E_ep.ndim - 1)
            for d in self.dims:
                ax = self._axis_for_dim(d)
                sl[E_ep.ndim - 1 + ax] = slice(gh, -gh)
            E_ep[(0,) + tuple(sl)] = E_act

    # ----------------------------------------------------------------
    # Runge-Kutta (method-of-lines) interface
    # ----------------------------------------------------------------

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Stage RHS for U; also solves the corner E-field of the same
        stage (from the ghosted flux-pass primitives) and stores the
        face-B RHS (edge-E curl) for ``compute_B_update``."""
        dUdt = FV_Scheme.compute_update(self, U, ader=ader, prims=prims, **kwargs)
        self.solve_edges(0)
        self._K_B = {dim: self.dBdt_dim(dim) for dim in self.dims}
        return dUdt

    def compute_B_update(self):
        """Face-B RHS of the stage prepared by ``compute_update``."""
        return self._K_B
