"""
MHD fallback: child of the hydro flux-blending fallback and the induction
(constrained-transport) fallback.

The correction is a MOOD cascade (superfv-style), hardwired to the levels
[high order, MUSCL(-Hancock), first order] for both the conserved fluxes and
the edge E-field.  Per ADER time node, each detection sweep tests the *full*
candidate state: the hydro candidate ``U_new`` with its B rows replaced by
the cell average of the candidate constrained-transport update from the
current edge-E assembly, so NAD bounds act on the true new B and the PAD
pressure includes its magnetic contribution.  Cells failing the check are
demoted one cascade level and the sweep repeats (a revision changes the
neighbors' updates too), until every revisable cell is admissible or
``max_revs`` sweeps were done.  The low-order E levels are built from the
same running subcell state as the low-order fluxes -- never from the
(unlimited) ADER predictor.  All E assemblies are single-valued on the edge
lattice, so div(B) = 0 is preserved to machine precision.
"""

from spd.fallback import FallbackScheme
from spd.fallback.induction_fallback import InductionFallbackScheme
from spd.induction.induction_sd_scheme import InductionSD_Scheme
from .mhd_fv_scheme import MHD_FV_Scheme


class MHDFallbackScheme(FallbackScheme, InductionFallbackScheme):
    """
    MOOD-cascade correction for the conserved MHD state (flux levels) and
    the face-staggered magnetic field (edge E-field levels).
    """

    def __init__(self, sim, **kwargs):
        # Default NAD variables for MHD: density, pressure and the active
        # (CT-evolved) magnetic field components.
        if kwargs.get("limiting_variables") is None:
            kwargs["limiting_variables"] = [sim._d_, sim._p_] + [
                sim.b[dim] for dim in sim.dims
            ]
        super().__init__(sim, **kwargs)
        # Per-node edge-E cascade levels (0 = high order, 1 = MUSCL corners,
        # 2 = first order) and the candidate cell-averaged CT field consumed
        # by detect_troubles.
        self._E_levels = {}
        self._B_cand = None
        self._with_ct = False

    # MUSCL-Hancock half-step prediction with the MHD source terms, borrowed
    # from the MHD FV scheme (both operate on the same ghosted FV layout).
    compute_prediction = MHD_FV_Scheme.compute_prediction

    def _has_ct_edges(self):
        """True when at least one edge E-family has both transverse
        directions resolved (CT is degenerate in 1D)."""
        for dim in self.Edims:
            dim1, dim2 = self.primary.other_dims(dim)
            if dim1 in self.dims and dim2 in self.dims:
                return True
        return False

    # ----------------------------------------------------------------
    # MOOD hooks: edge-E cascade levels and per-sweep candidate B
    # ----------------------------------------------------------------

    def mood_hook_start(self, dt):
        """Level-1 edge E (MUSCL corner states) from the same ghosted
        primitives the level-1 fluxes were just built from (``dm.M``; for
        MUSCL-Hancock this is the time-centered Hancock-predicted state,
        consistent with the level-1 fluxes)."""
        if self._with_ct:
            self._E_levels[1] = self.compute_low_order_E(self.dm.M, muscl=True)

    def mood_hook_level(self, level):
        """Terminal level: first-order (unreconstructed) edge E from the
        ghosted start-of-node primitives just used for the first-order
        fluxes (``dm.M``)."""
        if self._with_ct and level == 2:
            self._E_levels[2] = self.compute_low_order_E(self.dm.M, muscl=False)

    def mood_hook_candidate(self, dt):
        """Candidate cell-averaged B from the current edge-E assembly, for
        the admissibility check of this sweep's detection."""
        if self._with_ct:
            E = self.mood_edge_E(self._E_levels, self.dm.cascade)
            self._B_cand = self.candidate_cell_B(self.ct_dB(E, dt))

    def detect_troubles(self):
        # The true candidate B is the constrained-transport one: replace the
        # hydro-updated B rows of U_new so the NAD bounds test it directly
        # and the PAD pressure includes its magnetic contribution.
        if self._B_cand is not None:
            sim = self._sim
            for dim, Bc in self._B_cand.items():
                self.dm.U_new[sim.b[dim]] = Bc
        super().detect_troubles()

    # ----------------------------------------------------------------
    # Coupled corrector
    # ----------------------------------------------------------------

    def ader_update(self):
        """Coupled corrector, per ADER time node:

        1. High-order node fluxes and edge E form cascade level 0; the MOOD
           loop builds the low-order levels, runs the detection sweeps on
           the full candidate state (hydro + candidate CT B) and demotes
           troubled cells.
        2. The hydro state is advanced with the final flux assembly; the
           face B field with the CT update from the final edge-E assembly.

        After the node loop, the CT field is re-projected onto the
        cell-centered B rows.
        """
        if self.primary is None:
            return super().ader_update()
        prim = self.primary
        self.switch_to_finite_volume(ader=True)
        w_tp = self.dm.w_tp
        self._with_ct = self._has_ct_edges() and self.use_mood

        for i_ader in range(self.nader):
            dt_i = self.dt * w_tp[i_ader]
            if self._with_ct:
                self._E_levels = {
                    0: {
                        dim: prim.E_ader_ep[dim][0][i_ader]
                        for dim in self.Edims
                        if all(
                            d in self.dims for d in prim.other_dims(dim)
                        )
                    }
                }
            self.store_high_order_fluxes(i_ader, ader=True)
            self.compute_corrected_fluxes(dt_i)
            self.U_cv -= self.compute_dudt(self.U_cv) * dt_i
            if self._with_ct:
                # CT update from the final (post-revision) edge-E assembly.
                E_fin = self.mood_edge_E(self._E_levels, self.dm.cascade)
                dB = self.ct_dB(E_fin, dt_i)
                for dim in self.dims:
                    prim.B_fp[dim] -= dB[dim]
            elif self._has_ct_edges():
                # Godunov / theta-blending path: level-1 (MUSCL corners) E
                # from the ghosted primitives of this node's MUSCL flux pass
                # (dm.M), blended with the high-order node E by theta
                # (godunov: theta = 1, i.e. the fallback E everywhere --
                # consistent with the fallback fluxes used for the update).
                E_hi = {
                    dim: prim.E_ader_ep[dim][0][i_ader]
                    for dim in self.Edims
                    if all(d in self.dims for d in prim.other_dims(dim))
                }
                E_lo = self.compute_low_order_E(self.dm.M, muscl=True)
                theta = 1.0 if self.godunov else self.dm.theta
                self.blended_ct_node_update(E_hi, E_lo, dt_i, theta)
        self._B_cand = None
        self._E_levels = {}

        if not self._has_ct_edges():
            # Degenerate CT (e.g. 1D): plain high-order update of B.
            InductionSD_Scheme.ader_update(prim)
        self.switch_to_high_order()
        # B_to_U also refreshes the cell-averaged B rows (they drive the
        # next step's CFL condition and trouble detection).
        prim.B_to_U()

    # ----------------------------------------------------------------
    # Runge-Kutta (method-of-lines) interface
    # ----------------------------------------------------------------

    def compute_update(self, U, ader=False, prims=False, **kwargs):
        """Corrected stage RHS for U and (stored for ``compute_B_update``)
        for the face B field.

        Mirrors ``FallbackScheme.compute_update`` with the CT coupling of
        ``ader_update``: the stage's high-order edge E is cascade level 0,
        the MOOD hooks build the MUSCL / first-order corner-E levels from
        the same ghosted subcell states as the low-order fluxes, and the
        detection tests the full candidate (hydro U_new with the candidate
        CT B rows).  The final assemblies give both RHS families.
        """
        if self.primary is None:
            return super().compute_update(U, ader=ader, prims=prims, **kwargs)
        prim = self.primary
        prim.solve_faces(U, ader=ader, prims=prims)
        # Edge E of the same stage (set_stage_state refreshed W_sp/B_fp).
        prim.solve_edges(0)
        if self.viscosity and self.nu > 0:
            InductionSD_Scheme.add_nabla_terms(prim)
        self._with_ct = self._has_ct_edges() and self.use_mood
        E_hi = {
            dim: prim.E_ader_ep[dim][0]
            for dim in self.Edims
            if all(d in self.dims for d in prim.other_dims(dim))
        }
        if self._with_ct:
            self._E_levels = {0: E_hi}

        prim.switch_to_finite_volume(U_sp=U)
        self.store_high_order_fluxes(0, ader=ader)
        self.compute_corrected_fluxes(self.dt)
        dUdt_fv = self.compute_dudt(self.U_cv)
        dUdt_sp = prim.compute_sp_from_cv_fv(dUdt_fv)

        if self._with_ct:
            # Face-B RHS from the final (post-revision) edge-E assembly.
            E_fin = self.mood_edge_E(self._E_levels, self.dm.cascade)
            self._K_B = self.ct_dB(E_fin, 1.0)
            self._B_cand = None
            self._E_levels = {}
        elif self._has_ct_edges():
            # Godunov / theta-blending path (consistent with correct_fluxes).
            E_lo = self.compute_low_order_E(self.dm.M, muscl=True)
            theta = 1.0 if self.godunov else self.dm.theta
            self._K_B = self.ct_dB(
                self.blended_edge_E(E_hi, E_lo, theta), 1.0
            )
        else:
            # Degenerate CT (e.g. 1D): plain high-order edge E.
            self._K_B = {dim: prim.dBdt_dim(dim) for dim in self.dims}

        prim.switch_to_high_order(update_solution_points=False)
        return dUdt_sp

    def compute_B_update(self):
        """Face-B RHS of the stage prepared by ``compute_update``."""
        return self._K_B
