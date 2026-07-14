"""Block-based (AMR-capable) MUSCL fallback scheme.

Pairs with :class:`SD_AMR_Scheme` as the primary: FV working arrays carry a
meshblock axis right after nvar ([nvar, Nb, cells...]), ghost exchange is
forest-aware (SAME / COARSER / FINER / BC), the MUSCL reconstruction uses
per-block cell spacings (scaled by 2^-level), and the blended fluxes are
made strictly conservative across block interfaces (coarse-fine restriction
+ same-level symmetrization).

RK time integrators only. Dynamic AMR is supported: ``tag_blocks``/``adapt``
delegate the forest mutation and solution remap to the primary, then rebuild
every Nb-sized FV array here.
"""

import numpy as np

from spd.schemes.scheme import SemiDiscreteScheme
from spd.fallback.fallback import FallbackScheme
from spd.finite_volume import fv_amr_boundary as fvb
from spd.numerics.slicing import cut, indices, indices2


class FallbackAMRScheme(FallbackScheme):

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        if self.primary is None:
            raise ValueError(
                "FallbackAMRScheme requires a block-based primary scheme "
                "(standalone mode is not supported)."
            )
        if self.ader:
            raise ValueError("FallbackAMRScheme supports RK integrators only.")
        if self.use_mood:
            raise NotImplementedError(
                "The MOOD cascade is not supported on the block-based AMR "
                "fallback yet; use blending=True or godunov=True."
            )
        if self.potential or self.WB or self.viscosity or self.thdiffusion:
            raise NotImplementedError(
                "potential/WB/viscosity/thdiffusion not supported on the "
                "block-based AMR fallback yet."
            )
        self._refresh_fv_block_metrics()
        self.allocate_arrays(ader=False)
        self.fb_arrays()

    def _refresh_fv_block_metrics(self) -> None:
        """Per-block FV spacings/centers with an Nb axis.

        ``h_fp``/``h_cv`` are built from the level-0 per-block grid and
        scaled by 2^-level per block so MUSCL slopes and the flux divergence
        see each block's physical cell widths. Cell centers stay at level-0
        values for every block: the smooth-extrema indicator only consumes
        ratios of divided differences, which are invariant under the uniform
        per-block rescaling (same convention as the legacy runtime).
        Stored on dm so host/device switches migrate them; ``create_dicts``
        refreshes the dict entries via ``working_arrays``.
        """
        ngh = self.Nghc
        Nb = self.forest.Nblocks
        level_factor = np.array(
            [2.0 ** (-b.level) for b in self.forest.blocks]
        ).reshape((1, Nb) + (1,) * self.ndim)
        for dim in self.dims:
            idim = self.dims[dim]
            n = self.n[dim]
            NB = self.NB[dim]
            fp_b = np.ndarray((NB * n + 2 * ngh + 1))
            fp_b[ngh:-ngh] = self.h[dim] * np.hstack((
                np.arange(NB).repeat(n)
                + np.tile(self.primary.fp[dim][:-1], NB),
                NB,
            ))
            fp_b[:ngh] = -fp_b[ngh + 1:2 * ngh + 1][::-1]
            fp_b[-ngh:] = fp_b[-(ngh + 1)] + fp_b[ngh + 1:2 * ngh + 1]
            cv_b = 0.5 * (fp_b[1:] + fp_b[:-1])
            h_fp = (fp_b[1:] - fp_b[:-1])[self.shape(idim)]
            h_cv = (cv_b[1:] - cv_b[:-1])[self.shape(idim)]
            self.dm.__setattr__(
                f"h_fp_nb_{dim}", h_fp[:, np.newaxis, ...] * level_factor
            )
            self.dm.__setattr__(
                f"h_cv_nb_{dim}", h_cv[:, np.newaxis, ...] * level_factor
            )
            self.dm.__setattr__(f"centers_nb_{dim}", cv_b)
            # Smooth-extrema alpha = vL / (0.5 * h * d2U) is scale-invariant
            # only when h and the centers (used for the divided differences)
            # share the same length scale. Centers are level-0 for every
            # block, so SED must use the UNSCALED h (the level-scaled one
            # would inflate alpha by 2^level on refined blocks and wrongly
            # relax the NAD check there).
            self.dm.__setattr__(f"h_fp_sed_nb_{dim}", h_fp)

    def working_arrays(self) -> None:
        self.sed_h_fp = {}
        for dim in self.dims:
            self.h_fp[dim] = self.dm.__getattribute__(f"h_fp_nb_{dim}")
            self.h_cv[dim] = self.dm.__getattribute__(f"h_cv_nb_{dim}")
            self.centers[dim] = self.dm.__getattribute__(f"centers_nb_{dim}")
            self.sed_h_fp[dim] = self.dm.__getattribute__(
                f"h_fp_sed_nb_{dim}")
            # Rank-global face coordinates (kept for API compatibility;
            # unused by the block-based fallback path).
            self.faces[dim] = self.primary.faces[dim]

    # ----------------------------------------------------------------
    # Array allocation (block-based layout)
    # ----------------------------------------------------------------

    def array(self, nvar, dim="", ngh=0, ader=False) -> np.ndarray:
        assert not ader, "block-based fallback arrays carry no ADER axis"
        shape = [nvar, self.forest.Nblocks]
        N = []
        for dim2 in self.dims:
            N.append(self.NB[dim2] * self.n[dim2] + (dim2 in dim) + 2 * ngh)
        return np.ndarray(shape + N[::-1])

    def array_BC(self, dim="x", ader=False) -> np.ndarray:
        assert not ader
        ngh = self.Nghc
        shape = [2, self.nvar, self.forest.Nblocks]
        N = []
        for dim2 in self.dims:
            N.append(
                self.NB[dim2] * self.n[dim2] + 2 * ngh
                if dim != dim2
                else ngh
            )
        return np.ndarray(shape + N[::-1])

    def allocate_arrays(self, ader=False):
        assert not ader, "block-based fallback is RK-only"
        # Integrator stage arrays (primary/SD block layout via array_sp
        # delegation) + M_fp/F_fp/ML_fp/MR_fp/BC_fp via the block-aware
        # array builders above.
        SemiDiscreteScheme.allocate_arrays(self, ader)
        self.dm.M = self.array(self.nvar, ngh=self.Nghc)
        self.dm.U_new = self.array(self.nvar)
        if self.scheme == "MUSCL-Hancock":
            self.dm.dtM = self.array(self.nvar, ngh=self.Nghc - 1)

    def crop(self, M, ngh=1) -> np.ndarray:
        # Block-based arrays have [nvar, Nb, ...cells...]; skip both leading
        # batch axes (the simulator-level crop only skips nvar).
        return M[
            (slice(None),) * 2 + (slice(ngh, -ngh),) * self.ndim + (Ellipsis,)
        ]

    # ----------------------------------------------------------------
    # Forest-aware boundaries
    # ----------------------------------------------------------------

    def Boundaries(self, M: np.ndarray, all=True):
        fvb.Boundaries(self, M, all=all)

    def Boundaries_scalar(self, M: np.ndarray, dims=None):
        fvb.Boundaries_scalar(self, M, dims=dims)

    def refresh_theta_ghosts(self, theta: np.ndarray) -> None:
        """Re-exchange theta after apply_blending: the blending pass writes
        each block's theta ghost cells locally, which diverges from the
        neighbor block's post-blending theta and would bias affected_faces
        at block interfaces."""
        ngh = self.Nghc
        self.dm.M[...] = 0
        self.fill_active_region(self.crop(self.dm.theta, ngh=ngh))
        self.Boundaries_scalar(self.dm.M)
        theta[...] = self.dm.M[0]

    def minimize_alpha_across_blocks(self, alpha: np.ndarray,
                                     dims=None) -> None:
        """Apply SED ``compute_min`` on a forest-ghosted alpha field.

        Packs **all** limiting-variable channels into ``dm.M`` and exchanges
        only the coupling dimensions in one ``Boundaries_scalar`` pass, then
        runs ``compute_min`` so the 3-point stencil spans block faces —
        matching the single-grid SED neighborhood on a uniform decomposition.

        Previously this issued one full (all-dim) ghost exchange per limiting
        variable; with typical 2 vars × 3 dims that was 6 orchestrations /
        SED call, each scanning every block.
        """
        from spd.fallback.trouble_detection import compute_min

        ngh = self.Nghc
        ndim = self.ndim
        nlim = alpha.shape[0]
        couple_dims = self.dims if dims is None else {
            d: self.dims[d] for d in dims
        }
        active_sl = (slice(None),) + (slice(ngh, -ngh),) * ndim
        # Pack every limiting channel; exchange only the dims we couple.
        self.dm.M[...] = 0
        self.dm.M[(slice(0, nlim),) + active_sl] = alpha
        self.Boundaries_scalar(self.dm.M, dims=list(couple_dims.keys()))
        A = self.dm.M[:nlim]
        Amin = A.copy()
        for dim in couple_dims:
            idim = couple_dims[dim]
            compute_min(A, Amin, idim)
            A, Amin = Amin, A
        alpha[...] = A[(slice(None),) + active_sl]

    # ----------------------------------------------------------------
    # High-order flux staging (block layout)
    # ----------------------------------------------------------------

    def store_high_order_fluxes(self, i_ader, ader=False):
        """Copy the primary's (face-integrated) SD fluxes into the FV
        face-flux layout. Transpose tuples include the leading (nvar, Nb)
        batch axes."""
        ndim = self.ndim
        dims_t = [(0, 1, 2, 3),
                  (0, 1, 2, 4, 3, 5),
                  (0, 1, 2, 5, 3, 6, 4, 7)]
        dims2_t = [(0, 1),
                   (0, 1, 2, 3),
                   (0, 1, 2, 4, 3, 5)]
        Nb = self.forest.Nblocks
        Nn = [self.NB[dim] * self.n[dim] for dim in self.dims][::-1]
        for dim in self.dims:
            shift = self.dims[dim]
            shape = [self.nvar, Nb] + Nn
            Fd = self.get_high_order_fluxes(dim, i_ader, ader)
            # Interior faces: n faces per element (last one dropped).
            self.F_fp[dim][cut(None, -1, shift)] = np.transpose(
                Fd[cut(None, -1, shift)], dims_t[ndim - 1]
            ).reshape(shape)
            # Domain/block-boundary face (last face of the last element).
            shape.pop(ndim - shift + 1)     # +1 for the Nb axis
            self.F_fp[dim][indices(-1, shift)] = np.transpose(
                Fd[indices2(-1, ndim, shift)], dims2_t[ndim - 1]
            ).reshape(shape)

    # ----------------------------------------------------------------
    # Corrected fluxes with cross-block conservation
    # ----------------------------------------------------------------

    def _enforce_flux_consistency(self):
        """Single-valued fluxes at block interfaces.

        The theta-blend is generally NOT equal on the two sides of a block
        face (theta varies across it), which leaks mass between blocks.
        Overwrite the coarse-side flux with the restriction of the fine-side
        fluxes, and give SAME-level pairs their shared average.
        """
        if self.forest.max_level > 0:
            for dim in self.dims:
                fvb.correct_coarse_fine_fv_flux(self, self.F_fp[dim], dim)
        for dim in self.dims:
            fvb.symmetrize_same_level_fv_flux(self, self.F_fp[dim], dim)

    def compute_corrected_fluxes(self, dt):
        # Consistent HO fluxes first: the candidate update feeding the DMP
        # check must not be biased by mismatched block-interface fluxes.
        self._enforce_flux_consistency()
        super().compute_corrected_fluxes(dt)
        # ...and again after the blend, since theta varies across interfaces.
        self._enforce_flux_consistency()

    # ----------------------------------------------------------------
    # Dynamic AMR
    # ----------------------------------------------------------------

    def tag_blocks(self, **kwargs):
        """Forward to primary; accepts refine_fn_batched / derefine_fn_batched."""
        return self.primary.tag_blocks(**kwargs)

    def adapt(self, to_refine=None, to_derefine=None) -> None:
        """Adapt the forest via the primary (which remaps the SD solution),
        then rebuild every Nb-sized FV array and metric here."""
        self.primary.adapt(to_refine=to_refine, to_derefine=to_derefine)
        self._refresh_fv_block_metrics()
        self.allocate_arrays(ader=False)
        self.fb_arrays()
        self.create_dicts()
