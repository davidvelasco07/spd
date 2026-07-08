"""
MHD fallback: child of the hydro flux-blending fallback and the induction
(constrained-transport) fallback.

Per ADER time node, the detection tests the *full* candidate state: the
hydro candidate ``U_new`` with its B rows replaced by the cell average of
the candidate constrained-transport field (high-order CT update applied
provisionally), so NAD bounds act on the true new B and the PAD pressure
includes its magnetic contribution.  The resulting trouble indicator drives
both the hydro flux blending and the blending of the edge E-field (high
order vs four-state LLF from MUSCL-reconstructed corner states), keeping the
face-staggered B limited at shocks while divergence-free to machine
precision.
"""

import numpy as np

from spd.fallback import FallbackScheme
from spd.fallback.induction_fallback import InductionFallbackScheme
from spd.induction.induction_sd_scheme import InductionSD_Scheme


class MHDFallbackScheme(FallbackScheme, InductionFallbackScheme):
    """
    MUSCL flux blending for the conserved MHD state + theta-blended
    constrained transport for the face-staggered magnetic field.
    """

    def __init__(self, sim, **kwargs):
        # Default NAD variables for MHD: density, pressure and the active
        # (CT-evolved) magnetic field components.
        if kwargs.get("limiting_variables") is None:
            kwargs["limiting_variables"] = [sim._d_, sim._p_] + [
                sim.b[dim] for dim in sim.dims
            ]
        super().__init__(sim, **kwargs)
        # Cell-averaged candidate CT field of the current ADER node (set in
        # ader_update, consumed by detect_troubles).
        self._B_cand = None

    def _has_ct_edges(self):
        """True when at least one edge E-family has both transverse
        directions resolved (CT is degenerate in 1D)."""
        for dim in self.Edims:
            dim1, dim2 = self.primary.other_dims(dim)
            if dim1 in self.dims and dim2 in self.dims:
                return True
        return False

    def _node_low_order_E(self, i_ader):
        """Low-order edge E-field evaluated from the ADER predictor state at
        time node ``i_ader`` (projected to subcell averages, ghosted)."""
        prim = self.primary
        U_sd = prim.compute_cv_from_sp(prim.dm.U_ader[:, i_ader])
        W_fv = self._sim.compute_primitives(prim.transpose_to_fv(U_sd))
        self.fill_active_region(W_fv)
        self.Boundaries(self.dm.M)
        return self.compute_low_order_E(self.dm.M)

    def detect_troubles(self):
        # The true candidate B is the constrained-transport one: replace the
        # hydro-updated B rows of U_new so the NAD bounds test it directly
        # and the PAD pressure includes its magnetic contribution.
        if self._B_cand is not None:
            sim = self._sim
            for dim, Bc in self._B_cand.items():
                self.dm.U_new[sim.b[dim]] = Bc
        super().detect_troubles()

    def ader_update(self):
        """Coupled corrector with trouble detection, per ADER time node:

        1. Low-order edge E from the predictor state at the node; candidate
           cell-averaged B from the provisional high-order CT update.
        2. Hydro detection on the full candidate state (U_new with candidate
           B rows) + flux blending, as in the hydro fallback.
        3. Blended CT update of the face B for the node with the same theta.

        After the loop, the CT field is re-projected onto the cell-centered
        B rows.
        """
        if self.primary is None:
            return super().ader_update()
        prim = self.primary
        self.switch_to_finite_volume(ader=True)
        w_tp = self.dm.w_tp
        with_ct = self._has_ct_edges()

        for i_ader in range(self.nader):
            dt_i = self.dt * w_tp[i_ader]
            E_hi = E_lo = None
            if with_ct:
                # dm.M is scratch shared with the detection below, so gather
                # the node's low-order E first.
                E_lo = self._node_low_order_E(i_ader)
                E_hi = {
                    dim: prim.E_ader_ep[dim][0][i_ader] for dim in E_lo
                }
                self._B_cand = self.candidate_cell_B(self.ct_dB(E_hi, dt_i))
            self.store_high_order_fluxes(i_ader, ader=True)
            self.compute_corrected_fluxes(dt_i)
            self.U_cv -= self.compute_dudt(self.U_cv) * dt_i
            if with_ct:
                theta_node = 1.0 if self.godunov else self.dm.theta
                self.blended_ct_node_update(E_hi, E_lo, dt_i, theta_node)
        self._B_cand = None

        if not with_ct:
            # Degenerate CT (e.g. 1D): plain high-order update of B.
            InductionSD_Scheme.ader_update(prim)
        self.switch_to_high_order()
        prim.B_to_U()
        # Keep the cell-averaged B rows consistent with the CT face field
        # (they drive the next step's CFL condition and trouble detection).
        sim = self._sim
        b_rows = [sim.b[dim] for dim in self.dims]
        pdm = prim.dm
        pdm.U_cv[b_rows] = prim.compute_cv_from_sp(pdm.U_sp[b_rows])
