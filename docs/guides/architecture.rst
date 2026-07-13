Architecture Overview
=====================

The project is organized around a modular composition model:

- ``Simulator`` owns global run configuration (mesh, equations, BC, CFL, time loop).
- ``SemiDiscreteScheme`` implementations provide spatial operators.
- Integrators (ADER or RK variants) advance the selected scheme.
- Physics modules define primitives/conservatives and flux closures.
- Shared runtime and numerics packages provide communication and transforms.

Core data flow
--------------

1. A simulator subclass selects a high-order scheme (SD or FV).
2. The simulator selects a time integrator (ADER or RK family).
3. The integrator repeatedly calls scheme update hooks.
4. The scheme computes fluxes and source terms via the active physics module.
5. Optional fallback blending can replace troubled regions with robust FV fluxes.

Main modules
------------

- ``simulator.py``: base driver and integrator selection.
- ``schemes/``: abstract scheme contract and fallback implementation.
- ``spectral_difference/``: SD spatial operators and SD simulator entry points.
- ``finite_volume/``: FV spatial operators and FV simulator entry points.
- ``integrators/``: ADER, RK, and induction-specific RK integration.
- ``hydro/``, ``mhd/``, ``induction/``: equation-set specific logic.
- ``runtime/``: communication and host/device array management.
- ``numerics/``: interpolation, transforms, and slicing utilities.

Extension points
----------------

- Add a new spatial method by subclassing ``schemes.scheme.SemiDiscreteScheme``.
- Add a new time method by implementing an ``integrators`` class with
  ``allocate_arrays`` and ``update``.
- Add a new equations module by exposing compatible primitive/conservative/flux
  routines and wiring it through simulator selection logic.
