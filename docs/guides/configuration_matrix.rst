Configuration Matrix
====================

The framework combines three mostly independent axes:

- Spatial discretization: SD or FV
- Time integration: ADER or RK (including induction RK)
- Equations set: Hydro, Induction, or MHD

Common patterns
---------------

- ``spectral_difference.sd_simulator.SD_Simulator``
  - High-order SD runs with ADER (default) or RK.
- ``finite_volume.fv_simulator.FV_Simulator``
  - FV runs with MUSCL/MUSCL-Hancock options and RK/ADER integration flow.
- ``sdfb_simulator.SPD_Simulator``
  - SD primary plus FV fallback blending in troubled regions.
- ``induction.induction_simulator.InductionSimulator``
  - Induction-focused simulator selecting SD/FV induction schemes.
- ``MHD.mhd_spd_simulator.MHD_SPD_Simulator``
  - MHD-coupled simulator using the same modular architecture.

Key constructor knobs
---------------------

- ``scheme`` / ``scheme_fb``: choose SD/FV and fallback-enabled variants.
- ``time_integrator``: choose ``ader`` or ``rkN`` family.
- ``soe``: select equations path (hydro/induction/mhd where applicable).
- ``N``, ``p``, ``m``: mesh and order controls.
- ``BC``: per-dimension boundary-condition tuples.

Operational note
----------------

Fallback mode is intended to preserve robustness while keeping high-order fluxes
where smoothness criteria pass. Trouble detection and blending are implemented in
``schemes.fallback.FallbackScheme`` and ``trouble_detection`` helpers.
