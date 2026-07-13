"""
Top-level SPD entry point.

``SPD_Simulator`` dispatches on the system of equations (``soe``) to one of
the specialized simulators:

- ``"hydro"``     -> :class:`spd.hydro.hydro_simulator.HydroSimulator`
- ``"induction"`` -> :class:`spd.induction.induction_simulator.InductionSimulator`
- ``"mhd"``       -> :class:`spd.MHD.mhd_simulator.MHDSimulator`

All positional/keyword arguments are forwarded to the selected simulator, so

    sim = SPD_Simulator(soe="mhd", scheme="SDFB", ...)

is equivalent to ``MHDSimulator(scheme="SDFB", ...)``.
"""

from spd.hydro.hydro_simulator import HydroSimulator
from spd.induction.induction_simulator import InductionSimulator
from spd.MHD.mhd_simulator import MHDSimulator

_SIMULATORS = {
    "hydro": HydroSimulator,
    "induction": InductionSimulator,
    "mhd": MHDSimulator,
}


class SPD_Simulator:
    """Factory dispatching to the simulator matching ``soe``."""

    def __new__(cls, *args, soe: str = "hydro", **kwargs):
        try:
            simulator_cls = _SIMULATORS[soe]
        except KeyError:
            raise ValueError(
                f"Unknown system of equations '{soe}'; "
                f"expected one of {sorted(_SIMULATORS)}"
            ) from None
        return simulator_cls(*args, soe=soe, **kwargs)
