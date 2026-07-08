"""Can the SD->FV projection write into the interior of a ghosted buffer?

Checks correctness and speed of the 2D/3D projection contraction when the
output is a strided view (interior of a ghosted FV-layout array reinterpreted
with split element/node axes) vs a freshly allocated array.
"""
import time

import numpy as np
import cupy as cp
from cupy.lib.stride_tricks import as_strided
import spd.runtime.gpu  # noqa: F401  (enables cutensor)
from cupyx import cutensor


def interior_split_view(gh, ngh, N, n, ndim):
    """Interior of ghosted (u, N*n+2g, ...) array viewed as (u, N, n, ...)."""
    inner = gh[(Ellipsis,) + (slice(ngh, -ngh),) * ndim]
    shape = (gh.shape[0],)
    strides = (gh.strides[0],)
    for ax in range(ndim):
        shape += (N, n)
        s = inner.strides[1 + ax]
        strides += (n * s, s)
    return as_strided(inner, shape=shape, strides=strides)


def run(ndim, N, n, nvar=5, reps=20):
    B = cp.random.random((nvar,) + (N,) * ndim + (n,) * ndim)
    A2B = cp.random.random((n, n))
    ngh = 2
    gh = cp.zeros((nvar,) + (N * n + 2 * ngh,) * ndim)
    out = interior_split_view(gh, ngh, N, n, ndim)

    if ndim == 2:
        ref = cp.einsum("jm,il,uyxml->uyjxi", A2B, A2B, B, optimize=True)
        modes = (("j", "m"), ("i", "l"), ("u", "y", "x", "m", "l"),
                 ("u", "y", "j", "x", "i"))
    else:
        ref = cp.einsum("kn,jm,il,uzyxnml->uzkyjxi", A2B, A2B, A2B, B,
                        optimize=True)
        modes = (("k", "n"), ("j", "m"), ("i", "l"),
                 ("u", "z", "y", "x", "n", "m", "l"),
                 ("u", "z", "k", "y", "j", "x", "i"))

    # Pairwise contraction with the last one writing into the strided view.
    def project_into(out_view):
        if ndim == 2:
            tmp = cutensor.contraction(
                1.0, A2B, modes[1], B, modes[2], 0.0,
                cp.empty((nvar, N, N, n, n)), ("u", "y", "x", "m", "i"))
            cutensor.contraction(
                1.0, A2B, modes[0], tmp, ("u", "y", "x", "m", "i"),
                0.0, out_view, modes[3])
        else:
            t1 = cutensor.contraction(
                1.0, A2B, modes[2], B, modes[3], 0.0,
                cp.empty((nvar,) + (N,) * 3 + (n, n, n)),
                ("u", "z", "y", "x", "n", "m", "i"))
            t2 = cutensor.contraction(
                1.0, A2B, modes[1], t1, ("u", "z", "y", "x", "n", "m", "i"),
                0.0, cp.empty((nvar,) + (N,) * 3 + (n, n, n)),
                ("u", "z", "y", "x", "n", "j", "i"))
            cutensor.contraction(
                1.0, A2B, modes[0], t2, ("u", "z", "y", "x", "n", "j", "i"),
                0.0, out_view, modes[4])

    project_into(out)
    err = float(cp.max(cp.abs(out - ref.reshape(out.shape))))
    print(f"{ndim}D  max err vs einsum: {err:.3e}")

    for label, fn in (
        ("einsum (fresh alloc)", lambda: cp.einsum(
            *( ("jm,il,uyxml->uyjxi", A2B, A2B, B) if ndim == 2
               else ("kn,jm,il,uzyxnml->uzkyjxi", A2B, A2B, A2B, B) ),
            optimize=True)),
        ("cutensor -> strided view", lambda: project_into(out)),
    ):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        cp.cuda.Stream.null.synchronize()
        print(f"  {label:28s} {(time.perf_counter() - t) / reps * 1e3:7.3f} ms")


run(2, 512, 4)
run(3, 32, 4)
