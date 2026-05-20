"""Shared GPU helpers for runtime and physics modules."""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    fuse = cp.fuse
except Exception:
    cp = None
    CUPY_AVAILABLE = False

    def fuse(f):
        return f


def is_gpu_array(arr) -> bool:
    """True if *arr* lives on the device (CuPy ndarray)."""
    if not CUPY_AVAILABLE:
        return False
    return isinstance(arr, cp.ndarray)
