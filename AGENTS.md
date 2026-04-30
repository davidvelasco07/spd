# Agent Rules for `spd`

## perform_update Array Ownership Rule (always apply)

- **Pattern**: arrays used when running `perform_update` in the simulator.
- **Hard rule**: every such array/matrix must belong to `self.dm` (or be retrieved from `self.dm`), not simulator-local ad hoc storage.
- Do not create or cache update-path arrays on simulator objects as plain NumPy if they are consumed in `perform_update` call paths.
- Any update-path array must be backend/device compatible (CuPy on GPU runs), so `numpy.ndarray` must never be mixed into CuPy math during updates.
- This includes AMR transfer operators, FV/SD face/flux helper arrays, neighbor-index maps, and correction work arrays used during stepping.

## Reject / Fix examples

- Reject: building `R_side_cv`/transfer arrays as simulator-side NumPy and passing them into update-time AMR correction math.
- Fix: store these arrays in `dm` and keep types/device placement aligned with the active backend before `perform_update` uses them.
