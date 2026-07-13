"""Semi-discrete spatial discretization schemes."""

from .scheme import SemiDiscreteScheme

# FallbackScheme now lives in the dedicated ``spd.fallback`` package
# (alongside trouble detection) to avoid circular imports with
# finite_volume.fv_scheme:
#   from spd.fallback import FallbackScheme
