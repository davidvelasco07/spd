"""Shared GPU helpers for runtime and physics modules."""

import ctypes
import glob
import os

import numpy as np


def _preload_cutensor():
    """Make the pip-installed cuTENSOR shared libraries loadable.

    The ``cutensor-cu12`` wheel ships ``libcutensor*.so`` outside the default
    linker path, so load them explicitly before CuPy tries to import its
    cuTENSOR bindings.
    """
    try:
        import cutensor as _cutensor_pkg
    except ImportError:
        return
    for pkgdir in _cutensor_pkg.__path__:
        libdir = os.path.join(pkgdir, "lib")
        for pattern in ("libcutensor.so*", "libcutensorMg.so*"):
            for lib in sorted(glob.glob(os.path.join(libdir, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


try:
    import cupy as cp
    CUPY_AVAILABLE = True
    fuse = cp.fuse
except Exception:
    cp = None
    CUPY_AVAILABLE = False

    def fuse(f):
        return f


def _enable_cutensor():
    """Route CuPy einsum/tensor contractions through cuTENSOR when available.

    This typically speeds up the SD transform einsums by 3-6x.  Works
    regardless of import order or the CUPY_ACCELERATORS environment
    variable (which must otherwise be set before the first cupy import).
    """
    if not CUPY_AVAILABLE:
        return False
    _preload_cutensor()
    try:
        from cupyx import cutensor  # noqa: F401  (verifies the lib loads)
    except Exception:
        return False
    from cupy._core import _accelerator
    accels = list(_accelerator.get_routine_accelerators())
    cutensor_id = _accelerator.ACCELERATOR_CUTENSOR
    if cutensor_id not in accels:
        _accelerator.set_routine_accelerators(accels + [cutensor_id])
    return True


CUTENSOR_ENABLED = _enable_cutensor()


def is_gpu_array(arr) -> bool:
    """True if *arr* lives on the device (CuPy ndarray)."""
    if not CUPY_AVAILABLE:
        return False
    return isinstance(arr, cp.ndarray)
