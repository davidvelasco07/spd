"""Microbenchmarks for candidate GPU optimizations.

1. einsum strategies for the SD transforms (compute_A_from_B / _full)
2. transpose_to_fv: separate transpose+copy vs einsum fused into the
   cv-from-sp projection (direct FV-layout output)
3. cost of the per-step device->host sync in compute_dt
"""
import os
import time

import numpy as np
import cupy as cp

try:
    from cupyx import cutensor as cut
    HAS_CUTENSOR = True
except Exception:
    HAS_CUTENSOR = False


def timeit(f, n=30, warmup=3):
    for _ in range(warmup):
        f()
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t) / n * 1e3  # ms


def main():
    p = 3
    n = p + 1          # solution points per element per dim
    f = p + 2          # flux points
    N = 16             # elements per dim  -> 64^3 effective
    nvar, nt = 5, n    # ADER time points = n
    rng = cp.random.default_rng(0)

    print(f"config: p={p}, N={N}^3 elements ({N*n}^3 cells), nvar={nvar}, ADER nt={nt}")
    print(f"accelerators: {cp._core.get_routine_accelerators()} (1=cub, 2=cutensor)")
    print()

    # ---- 1. compute_A_from_B, dim='x': "fs,utzyxkjs->utzyxkjf" ----
    A2B = rng.random((f, n))
    B = rng.random((nvar, nt, N, N, N, n, n, n))
    out = cp.empty((nvar, nt, N, N, N, n, n, f))

    t_einsum = timeit(lambda: cp.einsum("fs,utzyxkjs->utzyxkjf", A2B, B))
    t_matmul = timeit(lambda: cp.matmul(B, A2B.T))
    t_matmul_out = timeit(lambda: cp.matmul(B, A2B.T, out=out))
    print("1a. sp->fp along x  (contract last axis)")
    print(f"    cp.einsum                : {t_einsum:8.3f} ms")
    print(f"    cp.matmul (B @ A.T)      : {t_matmul:8.3f} ms")
    print(f"    cp.matmul out=prealloc   : {t_matmul_out:8.3f} ms")

    if HAS_CUTENSOR:
        t_cut = timeit(lambda: cut.contraction(
            1.0, A2B, ("f", "s"), B, ("u", "t", "z", "y", "x", "k", "j", "s"),
            0.0, out, ("u", "t", "z", "y", "x", "k", "j", "f")))
        print(f"    cutensor.contraction out=: {t_cut:8.3f} ms")

    # ---- 1b. dim='z': "fs,utzyxsji->utzyxfji" (contract axis -3) ----
    Bz = rng.random((nvar, nt, N, N, N, n, n, n))
    outz = cp.empty((nvar, nt, N, N, N, f, n, n))
    t_einsum_z = timeit(lambda: cp.einsum("fs,utzyxsji->utzyxfji", A2B, Bz))

    def matmul_z():
        Bv = Bz.reshape(nvar, nt, N, N, N, n, n * n)
        cp.matmul(A2B, Bv, out=outz.reshape(nvar, nt, N, N, N, f, n * n))
    t_matmul_z = timeit(matmul_z)
    print("1b. sp->fp along z  (contract axis -3)")
    print(f"    cp.einsum                : {t_einsum_z:8.3f} ms")
    print(f"    cp.matmul reshaped out=  : {t_matmul_z:8.3f} ms")
    if HAS_CUTENSOR:
        t_cut_z = timeit(lambda: cut.contraction(
            1.0, A2B, ("f", "s"), Bz, ("u", "t", "z", "y", "x", "s", "j", "i"),
            0.0, outz, ("u", "t", "z", "y", "x", "f", "j", "i")))
        print(f"    cutensor.contraction out=: {t_cut_z:8.3f} ms")

    # ---- 2. cv_from_sp + transpose_to_fv (3D) ----
    U_sp = rng.random((nvar, N, N, N, n, n, n))
    sp_to_cv = rng.random((n, n))

    def current_path():
        U_cv = cp.einsum("kn,jm,il,uzyxnml->uzyxkji", sp_to_cv, sp_to_cv, sp_to_cv, U_sp)
        M = cp.transpose(U_cv, (0, 1, 4, 2, 5, 3, 6)).reshape(nvar, N * n, N * n, N * n)
        return M

    def fused_einsum():
        # einsum writes the FV-interleaved layout directly; reshape is then free
        U_fv = cp.einsum("kn,jm,il,uzyxnml->uzkyjxi", sp_to_cv, sp_to_cv, sp_to_cv, U_sp)
        return U_fv.reshape(nvar, N * n, N * n, N * n)

    def transpose_only():
        U_cv = cp.einsum("kn,jm,il,uzyxnml->uzyxkji", sp_to_cv, sp_to_cv, sp_to_cv, U_sp)
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        for _ in range(30):
            cp.ascontiguousarray(cp.transpose(U_cv, (0, 1, 4, 2, 5, 3, 6)))
        cp.cuda.Stream.null.synchronize()
        return (time.perf_counter() - t) / 30 * 1e3

    t_cur = timeit(current_path)
    t_fus = timeit(fused_einsum)
    t_tr = transpose_only()
    r1 = current_path(); r2 = fused_einsum()
    err = float(cp.abs(r1 - r2).max())
    print("2.  switch_to_finite_volume: cv_from_sp + layout change")
    print(f"    projection then transpose+copy : {t_cur:8.3f} ms")
    print(f"    fused einsum (direct FV order) : {t_fus:8.3f} ms   (max diff {err:.2e})")
    print(f"    transpose+copy alone           : {t_tr:8.3f} ms")

    # inverse direction
    M_fv = fused_einsum()
    cv_to_sp = cp.asarray(np.linalg.inv(cp.asnumpy(sp_to_cv)))

    def current_back():
        U_cv = cp.transpose(M_fv.reshape(nvar, N, n, N, n, N, n), (0, 1, 3, 5, 2, 4, 6))
        return cp.einsum("kn,jm,il,uzyxnml->uzyxkji", cv_to_sp, cv_to_sp, cv_to_sp, U_cv)

    def fused_back():
        V = M_fv.reshape(nvar, N, n, N, n, N, n)  # u z k y j x i view, free
        return cp.einsum("kn,jm,il,uznymxl->uzyxkji", cv_to_sp, cv_to_sp, cv_to_sp, V)

    t_cb = timeit(current_back)
    t_fb = timeit(fused_back)
    err2 = float(cp.abs(current_back() - fused_back()).max())
    print("    inverse (fv->sd->sp):")
    print(f"    transpose then projection      : {t_cb:8.3f} ms")
    print(f"    fused einsum on reshaped view  : {t_fb:8.3f} ms   (max diff {err2:.2e})")

    # ---- 3. compute_dt device sync ----
    c = rng.random((nvar, N * n, N * n, N * n))
    def dt_sync():
        return float(cp.max(c))  # forces D2H sync
    t_sync = timeit(dt_sync, n=50)
    print("3.  per-step reduction + D2H sync (compute_dt pattern)")
    print(f"    cp.max + .item()               : {t_sync:8.3f} ms")


if __name__ == "__main__":
    main()
