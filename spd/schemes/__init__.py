"""Semi-discrete spatial discretization schemes."""

from .scheme import SemiDiscreteScheme

# FallbackScheme is imported lazily to avoid circular imports
# with finite_volume.fv_scheme:
#   from schemes.fallback import FallbackScheme
