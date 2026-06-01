"""
Fallback / shock capturing for SD induction: trouble detection on |B|^2| and
reduced high-order magnetic update on affected faces.
"""

import numpy as np

from trouble_detection import detect_troubles_induction


class InductionFallbackScheme:
    """
    Wraps :class:`InductionSD_Scheme` with optional blending.

    When ``godunov`` is True or no troubled cells are flagged, delegates to the
    primary update. Otherwise scales the high-order ``dB`` by ``(1 - theta_face)``.
    """

    def __init__(
        self,
        sim,
        primary,
        FB: bool = True,
        tolerance: float = 0.05,
        blending: bool = True,
        godunov: bool = False,
    ):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "primary", primary)
        self.FB = FB
        self.tolerance = tolerance
        self.blending = blending
        self.godunov = godunov
        self._alloc_fb_arrays()

    @property
    def dm(self):
        return self.primary.dm

    def _alloc_fb_arrays(self):
        p = self.primary
        dm = p.dm
        dm.troubles = p.array(1)
        dm.theta = p.array(1, ngh=self._sim.Nghc)
        for dim in p.dims:
            dm.__setattr__(f"affected_faces_{dim}", p.array(1, dim=dim))

    def __getattr__(self, name):
        if name in ("_sim", "primary", "FB", "tolerance", "blending", "godunov"):
            raise AttributeError(name)
        return getattr(self.primary, name)

    def ader_predictor(self, prims: bool = False):
        self.primary.ader_predictor(prims=prims)

    def ader_update(self):
        if not self.FB or self.godunov:
            self.primary.ader_update()
            return
        detect_troubles_induction(
            self.primary,
            tolerance=self.tolerance,
            blending=self.blending,
        )
        if not np.any(self.dm.troubles):
            self.primary.ader_update()
            return
        for dim in self.dims:
            s = self.primary.ader_string()
            dB = np.einsum(
                f"t,t{s}->{s}", self.dm.w_tp, self.primary.ader_dBdt(dim)
            )
            tf = self.dm.__getattribute__(f"affected_faces_{dim}")
            self.primary.B_fp[dim][...] -= dB * (1.0 - tf)

    def post_update(self):
        if hasattr(self.primary, "post_update"):
            self.primary.post_update()
