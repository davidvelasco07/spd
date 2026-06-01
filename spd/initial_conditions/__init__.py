from . import initial_conditions_1d as ic1d  # noqa: F401
from . import initial_conditions_2d as ic2d  # noqa: F401
from . import initial_conditions_3d as ic3d  # noqa: F401

from .initial_conditions_1d import *  # noqa: F401,F403
from .initial_conditions_2d import *  # noqa: F401,F403
from .initial_conditions_3d import *  # noqa: F401,F403

def step_function(**kwargs):
    return lambda xyz, case: ic3d.step_function(xyz, case, **kwargs)

def sine_wave(**kwargs):
    return lambda xyz, case: ic3d.sine_wave(xyz, case, **kwargs)

def sod(**kwargs):
    return lambda xyz, case: ic1d.sod_shock_tube(xyz, case, **kwargs)
