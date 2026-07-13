"""Flux-blending fallback scheme and trouble detection."""

from .fallback import FallbackScheme
from .fallback_amr import FallbackAMRScheme
from .induction_fallback import InductionFallbackScheme
from .trouble_detection import detect_troubles, detect_troubles_induction

__all__ = [
    "FallbackScheme",
    "FallbackAMRScheme",
    "InductionFallbackScheme",
    "detect_troubles",
    "detect_troubles_induction",
]
