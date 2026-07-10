"""
Induction (constrained-transport) fallback.

Provides a divergence-free limiter for the face-staggered magnetic field of
an induction-capable high-order scheme:

- a four-state LLF electric field is computed on the primary's edge-point
  lattice from MUSCL-reconstructed corner states of the ghosted subcell
  primitives (same slope limiter as the hydro fallback);
- the time-integrated high-order edge E-field is blended with it using the
  trouble indicator ``theta`` pooled onto the edges;
- a single CT update from the blended (single-valued) edge E-field then
  advances the face B field.

Because the blended E-field is single-valued on the edge lattice, the CT
update preserves div(B) = 0 to machine precision — unlike scaling the
per-face dB increments, which reintroduces divergence at trouble boundaries.

This class carries no state of its own: it is designed as a base of
``MHDFallbackScheme`` (child of both the hydro ``FallbackScheme`` and this
class), where ``self.primary`` is the high-order induction-capable scheme
and ``self.dm`` holds the hydro fallback's ghosted FV arrays.
"""

import numpy as np

from spd.numerics.slicing import cut


class InductionFallbackScheme:
    """
    Edge-based induction fallback: theta-blended constrained transport.

    Expects the host class to provide (directly or by proxy):
    ``primary`` (induction-capable SD scheme), ``dm`` (with ghosted ``M``),
    ``dims``/``Edims``, ``N``/``n``, ``Nghc``, ``h``, ``dt``,
    ``compute_slopes`` and the equation indices (``vels``, ``b``, ``_d_``,
    ``_p_``).
    """

    # ----------------------------------------------------------------
    # Edge-lattice gathering
    # ----------------------------------------------------------------

    def _edge_gather_indices(self, dim):
        """Per-axis index arrays addressing, for every point of the edge
        lattice of E-family ``dim``, the ghosted-FV subcell on its "+" side
        (transverse axes) or containing it (axis along the edge).

        Shaped to broadcast to the lattice layout
        ``(N_z, N_y, N_x, p_z, p_y, p_x)`` (reversed-dims order, transverse
        point axes of length p+2, edge-parallel of length p+1).
        """
        xp = self.dm.xp
        dim1, dim2 = self.primary.other_dims(dim)
        gh = self.Nghc
        dims_r = list(self.dims)[::-1]
        ndim = len(dims_r)
        idx = {}
        for a, d in enumerate(dims_r):
            n = self.n[d]
            npts = n + 1 if d in (dim1, dim2) else n
            base = gh + (
                np.arange(self.N[d])[:, None] * n + np.arange(npts)[None, :]
            )
            shape = [1] * (2 * ndim)
            shape[a] = self.N[d]
            shape[ndim + a] = npts
            idx[d] = xp.asarray(base.reshape(shape))
        return idx

    def _edge_gather(self, M_gh, var, idx, offsets):
        """Row ``var`` of the ghosted FV array ``M_gh`` at the subcell
        displaced by ``offsets`` (dim -> 0/-1, transverse axes only) from
        each point of the edge lattice described by ``idx``."""
        ix = []
        for d in list(self.dims)[::-1]:
            off = offsets.get(d, 0)
            ix.append(idx[d] + off if off else idx[d])
        return M_gh[(var,) + tuple(ix)]

    # ----------------------------------------------------------------
    # Low-order (four-state LLF, MUSCL-reconstructed) edge electric field
    # ----------------------------------------------------------------

    def compute_low_order_E(self, W_gh, muscl=True):
        """
        Four-state CT electric field at the primary's edge points from the
        ghosted subcell primitives ``W_gh``.

        With ``muscl=True`` (cascade level 1) the four corner states are
        MUSCL reconstructions of the adjacent subcells (limited half-slopes
        along both transverse directions, same limiter as the hydro
        fallback); with ``muscl=False`` (terminal level, first order) the
        subcell values are used unreconstructed.

        The corner states are resolved by
        :func:`spd.MHD.mhd_fv_scheme.four_state_E` (borrowed from the MHD FV
        scheme, which uses the same construction for its edge E-field):
        a four-state LLF bound with fast magnetosonic speeds, or two
        successive one-dimensional HLLD sweeps (dim1 then dim2) when the
        fallback's Riemann solver is HLLD.
        """
        from spd.MHD.mhd_fv_scheme import four_state_E

        sim = self._sim
        min_rho = getattr(self, "min_rho", 1e-12)
        min_P = getattr(self, "min_P", 1e-12)
        use_hlld = getattr(self, "riemann_solver_name", "llf") == "hlld"
        E_lo = {}
        for dim in self.Edims:
            dim1, dim2 = self.primary.other_dims(dim)
            if dim1 not in self.dims or dim2 not in self.dims:
                continue
            idx = self._edge_gather_indices(dim)
            i1, i2 = self.dims[dim1], self.dims[dim2]
            # Limited half-slopes aligned with the ghosted array (the
            # outermost ghost layer keeps zero slope; it is never gathered).
            S1 = np.zeros_like(W_gh)
            S2 = np.zeros_like(W_gh)
            if muscl:
                S1[cut(1, -1, i1)] = self.compute_slopes(W_gh, i1)
                S2[cut(1, -1, i2)] = self.compute_slopes(W_gh, i2)
            r_v1 = sim.vels[i1]
            r_v2 = sim.vels[i2]

            def corner(o1, o2):
                """(v1, v2, B1, B2, B3, rho, p) reconstructed towards the
                edge point from the subcell at offsets (o1, o2)."""
                off = {dim1: o1, dim2: o2}
                # The edge point is the corner of the subcell on the
                # opposite side of each offset: reconstruct towards it.
                sgn1 = 1.0 if o1 == -1 else -1.0
                sgn2 = 1.0 if o2 == -1 else -1.0
                g = lambda var: (
                    self._edge_gather(W_gh, var, idx, off)
                    + sgn1 * self._edge_gather(S1, var, idx, off)
                    + sgn2 * self._edge_gather(S2, var, idx, off)
                )
                # Positivity guard (RAMSES-style): where the reconstructed
                # density/pressure is at/below its floor, revert that
                # variable to the donor-cell (unreconstructed) value.
                def g_pos(var, floor):
                    rec = g(var)
                    dc = self._edge_gather(W_gh, var, idx, off)
                    return np.where(rec > floor, rec, dc)
                return (
                    g(r_v1), g(r_v2),
                    g(sim.b[dim1]), g(sim.b[dim2]), g(sim.b[dim]),
                    g_pos(sim._d_, min_rho),
                    g_pos(sim._p_, min_P),
                )

            E_lo[dim] = four_state_E(
                corner,
                self.gamma,
                sim.min_c2,
                use_hlld=use_hlld,
                xp=self.dm.xp,
            )
        return E_lo

    # ----------------------------------------------------------------
    # Theta-blended constrained-transport update
    # ----------------------------------------------------------------

    def edge_pool(self, dim, field_gh):
        """Ghosted cell field (trouble indicator or cascade index) pooled to
        the edge lattice of E-family ``dim`` (max over adjacent subcells)."""
        dim1, dim2 = self.primary.other_dims(dim)
        idx = self._edge_gather_indices(dim)
        th = None
        for o1 in (-1, 0):
            for o2 in (-1, 0):
                t = self._edge_gather(field_gh, 0, idx, {dim1: o1, dim2: o2})
                th = t if th is None else np.maximum(th, t)
        return th

    # Backwards-compatible name for the theta pooling.
    edge_theta = edge_pool

    def ct_dB(self, E, dt_i):
        """Face-B increments ``dt_i * curl(E)`` from a dict of single-valued
        edge E-fields (mirrors InductionSD_Scheme.ader_dBdt term by term)."""
        prim = self.primary
        na = np.newaxis
        dB = {}
        for dim in self.dims:
            dim1, dim2 = prim.other_dims(dim)
            d = (
                prim.compute_sp_from_dfp(E[dim1][na], dim2)[0] / self.h[dim2]
                if dim1 in E
                else 0
            )
            d -= (
                prim.compute_sp_from_dfp(E[dim2][na], dim1)[0] / self.h[dim1]
                if dim2 in E
                else 0
            )
            dB[dim] = d * dt_i
        return dB

    def candidate_cell_B(self, dB_hi):
        """Cell-averaged (FV layout) B rows of the *candidate* CT update
        ``B_fp - dB_hi``, for the admissibility check of the detection."""
        prim = self.primary
        na = np.newaxis
        rows = {}
        for dim in self.dims:
            B_sp = prim.compute_sp_from_fp(
                (prim.B_fp[dim] - dB_hi[dim])[na], dim=dim
            )
            rows[dim] = prim.transpose_to_fv(prim.compute_cv_from_sp(B_sp))[0]
        return rows

    def mood_edge_E(self, E_levels, cascade_gh):
        """Per-edge assembly of the edge E-field from the MOOD cascade:
        the level of an edge is the max cascade index of its adjacent
        subcells; each edge takes the E of its level (0 = high order,
        1 = MUSCL corners, 2 = first order)."""
        E = {}
        for dim, E_hi in E_levels[0].items():
            mask = self.edge_pool(dim, cascade_gh)
            e = E_hi
            for lvl in (1, 2):
                E_l = E_levels.get(lvl)
                if E_l is not None:
                    e = np.where(mask == lvl, E_l[dim], e)
            E[dim] = e
        return E

    def blended_edge_E(self, E_hi, E_lo, theta_gh):
        """
        Theta-blended edge E-field:  E = (1 - theta_e) * E_hi + theta_e * E_lo

        ``theta_gh`` is the ghosted FV trouble indicator (``dm.theta``-like,
        leading singleton variable axis) or a scalar (godunov mode).  The
        endpoints select exactly so a non-finite E on the fully discarded
        side (e.g. HLLD fed inadmissible predictor states) does not poison
        the blend.
        """
        E_blend = {}
        for dim, E_l in E_lo.items():
            if np.isscalar(theta_gh):
                th = theta_gh
            else:
                th = self.edge_theta(dim, theta_gh)
            if np.isscalar(th) and th >= 1:
                E_blend[dim] = E_l
            elif np.isscalar(th) and th <= 0:
                E_blend[dim] = E_hi[dim]
            else:
                E_blend[dim] = np.where(
                    th >= 1,
                    E_l,
                    np.where(th <= 0, E_hi[dim],
                             E_hi[dim] + th * (E_l - E_hi[dim])),
                )
        return E_blend

    def blended_ct_node_update(self, E_hi, E_lo, dt_i, theta_gh):
        """
        Node-level CT update of the face B fields from the theta-blended
        edge E-field:  B -= dt_i * curl(E_blend).  The blended E is
        single-valued on the edge lattice, so div(B) is preserved to
        machine precision.
        """
        E_blend = self.blended_edge_E(E_hi, E_lo, theta_gh)
        dB = self.ct_dB(E_blend, dt_i)
        for dim in self.dims:
            self.primary.B_fp[dim] -= dB[dim]
