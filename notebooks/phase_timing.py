"""Per-phase GPU wall-time breakdown of an SD ADER step.

Monkeypatches key scheme methods with synchronizing timers.
Synchronization inflates total time slightly but exposes where the
GPU time is spent.
"""
import argparse
import collections
import time

import cupy as cp

from spd.spd_simulator import SPD_Simulator as SDFB_Simulator

acc = {}


def wrap(obj, name, label=None):
    label = label or name
    fn = getattr(obj, name)

    def timed(*a, **k):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        cp.cuda.Stream.null.synchronize()
        acc[label] = acc.get(label, 0.0) + (time.perf_counter() - t)
        return r

    setattr(obj, name, timed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--FB", action="store_true")
    args = ap.parse_args()

    s = SDFB_Simulator(p=args.p, N=(args.N,) * 3, use_cupy=True,
                       FB=args.FB, verbose=False)
    s.perform_iterations(2)  # warmup

    sch = s.scheme
    ho = sch if not hasattr(sch, "primary") or sch.primary is None else sch.primary
    wrap(ho, "compute_fp_from_sp")
    wrap(ho, "compute_sp_from_dfp")
    wrap(ho, "compute_cv_from_sp")
    wrap(ho, "compute_sp_from_cv")
    wrap(ho, "transpose_to_fv")
    wrap(ho, "transpose_to_sd")
    wrap(ho, "compute_fluxes")
    wrap(ho, "compute_primitives")
    wrap(ho, "compute_dt")
    rs = ho.riemann_solver
    def timed_rs(*a, **k):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        r = rs(*a, **k)
        cp.cuda.Stream.null.synchronize()
        acc["riemann_solver"] = acc.get("riemann_solver", 0.0) + (time.perf_counter() - t)
        return r
    ho.riemann_solver = timed_rs
    if hasattr(sch, "primary") and sch.primary is not None:
        wrap(sch, "store_high_order_fluxes")
        wrap(sch, "compute_corrected_fluxes")
        wrap(sch, "switch_to_finite_volume")
        wrap(sch, "switch_to_high_order")

    t0 = time.perf_counter()
    s.perform_iterations(args.steps)
    total = time.perf_counter() - t0

    ncell = (args.N * (args.p + 1)) ** 3
    print(f"\np={args.p}, {args.N}^3 elems = {round(ncell ** (1 / 3))}^3 cells, "
          f"{args.steps} steps, total {total:.3f} s -> {total / args.steps * 1e3:.1f} ms/step")
    other = total - sum(acc.values())
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v / args.steps * 1e3:9.2f} ms/step  ({v / total * 100:5.1f}%)")
    print(f"  {'(untimed / python)':28s} {other / args.steps * 1e3:9.2f} ms/step  ({other / total * 100:5.1f}%)")
    pool = cp.get_default_memory_pool()
    print(f"mem pool used {pool.used_bytes()/1e9:.2f} GB / total {pool.total_bytes()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
