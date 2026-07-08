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

    def compute_low_order_E(self, W_gh):
        """
        Four-state LLF-CT electric field at the primary's edge points from
        the ghosted subcell primitives ``W_gh``.

        The four corner states are MUSCL reconstructions of the adjacent
        subcells (limited half-slopes along both transverse directions,
        same limiter as the hydro fallback).  For E-family ``dim`` with
        transverse directions (dim1, dim2), using the code convention
        E = v1*B2 - v2*B1:

            E_lo = 1/4 sum_s E_s - S1/2 <dB2>_1 + S2/2 <dB1>_2

        with the averaged tangential-field jumps across each transverse
        direction and S = max_s(|v| + c_fast) the local LLF speed bound.
        """
        sim = self._sim
        min_rho = getattr(self, "min_rho", 1e-12)
        min_P = getattr(self, "min_P", 1e-12)
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
            S1[cut(1, -1, i1)] = self.compute_slopes(W_gh, i1)
            S2 = np.zeros_like(W_gh)
            S2[cut(1, -1, i2)] = self.compute_slopes(W_gh, i2)
            r_v1 = sim.vels[i1]
            r_v2 = sim.vels[i2]
            rows = [r_v1, r_v2, sim.b[dim1], sim.b[dim2], sim.b[dim],
                    sim._d_, sim._p_]
            E_sum = 0.0
            Sp1 = Sp2 = None
            dB2_1 = 0.0
            dB1_2 = 0.0
            for o1 in (-1, 0):
                for o2 in (-1, 0):
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
                    v1, v2 = g(r_v1), g(r_v2)
                    B1, B2, B3 = g(sim.b[dim1]), g(sim.b[dim2]), g(sim.b[dim])
                    rho = np.maximum(g(sim._d_), min_rho)
                    p = np.maximum(g(sim._p_), min_P)
                    E_sum = E_sum + (v1 * B2 - v2 * B1)
                    # Fastest magnetosonic bound: sqrt(a^2 + B^2/rho).
                    c = np.sqrt(
                        (self.gamma * p + B1 * B1 + B2 * B2 + B3 * B3) / rho
                    )
                    s1 = np.abs(v1) + c
                    s2 = np.abs(v2) + c
                    Sp1 = s1 if Sp1 is None else np.maximum(Sp1, s1)
                    Sp2 = s2 if Sp2 is None else np.maximum(Sp2, s2)
                    jsgn1 = 1.0 if o1 == 0 else -1.0
                    jsgn2 = 1.0 if o2 == 0 else -1.0
                    dB2_1 = dB2_1 + 0.5 * jsgn1 * B2
                    dB1_2 = dB1_2 + 0.5 * jsgn2 * B1
            E_lo[dim] = 0.25 * E_sum - 0.5 * Sp1 * dB2_1 + 0.5 * Sp2 * dB1_2
        return E_lo

    # ----------------------------------------------------------------
    # Theta-blended constrained-transport update
    # ----------------------------------------------------------------

    def edge_theta(self, dim, theta_gh):
        """Trouble indicator pooled to the edge lattice of E-family ``dim``
        (max over the adjacent subcells)."""
        dim1, dim2 = self.primary.other_dims(dim)
        idx = self._edge_gather_indices(dim)
        th = None
        for o1 in (-1, 0):
            for o2 in (-1, 0):
                t = self._edge_gather(theta_gh, 0, idx, {dim1: o1, dim2: o2})
                th = t if th is None else np.maximum(th, t)
        return th

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

    def blended_ct_node_update(self, E_hi, E_lo, dt_i, theta_gh):
        """
        Node-level CT update of the face B fields from the theta-blended
        edge E-field:

            E = (1 - theta_e) * E_hi + theta_e * E_lo
            B -= dt_i * curl(E)

        ``theta_gh`` is the ghosted FV trouble indicator of this time node
        (``dm.theta``-like, leading singleton variable axis) or a scalar
        (godunov mode).  The blended E is single-valued on the edge lattice,
        so div(B) is preserved to machine precision.
        """
        E_blend = {}
        for dim, E_l in E_lo.items():
            if np.isscalar(theta_gh):
                th = theta_gh
            else:
                th = self.edge_theta(dim, theta_gh)
            E_blend[dim] = E_hi[dim] + th * (E_l - E_hi[dim])
        dB = self.ct_dB(E_blend, dt_i)
        for dim in self.dims:
            self.primary.B_fp[dim] -= dB[dim]
