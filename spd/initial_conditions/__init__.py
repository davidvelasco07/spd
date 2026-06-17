"""Initial condition library.

Each underlying initial-condition function has the signature
``f(xyz, case, **params)`` where ``case`` selects the sampled field
(``0`` density, ``1`` vx, ``2`` vy, ``3`` vz, ``4`` pressure, ``-1``
gravitational potential).

The simulator expects ``init_fct`` to be a two-argument callable
``init_fct(xyz, case)``.  To make the tunable parameters easy to set, the
names exported here are *factories*: calling one with keyword arguments
binds those parameters and returns a ready-to-use ``init_fct``.  For example::

    init_fct = RTI(P0=10)
    init_fct = step_function(vx=2)
    init_fct = sod()
"""

import functools

from . import initial_conditions_1d as ic1d  # noqa: F401
from . import initial_conditions_2d as ic2d  # noqa: F401
from . import initial_conditions_3d as ic3d  # noqa: F401


def parametrized(func):
    """Turn ``f(xyz, case, **params)`` into a factory ``f(**params) -> init_fct``.

    The returned ``init_fct(xyz, case)`` forwards ``xyz``/``case`` to ``func``
    with the bound parameters applied.
    """

    @functools.wraps(func)
    def factory(**params):
        def init_fct(xyz, case):
            return func(xyz, case, **params)

        init_fct.__name__ = getattr(func, "__name__", "init_fct")
        init_fct.keywords = params
        return init_fct

    return factory


# Dimension-aware factories (these dispatch to the 1d/2d/3d versions
# internally based on the shape of ``xyz``).
step_function = parametrized(ic3d.step_function)
sine_wave = parametrized(ic3d.sine_wave)

# 1d-only.
sod = parametrized(ic1d.sod_shock_tube)

# 2d-only.
RTI = parametrized(ic2d.RTI)
KH_instability = parametrized(ic2d.KH_instability)

__all__ = [
    "ic1d",
    "ic2d",
    "ic3d",
    "parametrized",
    "step_function",
    "sine_wave",
    "sod",
    "RTI",
    "KH_instability",
]
