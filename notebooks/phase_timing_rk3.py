"""Hierarchical per-phase GPU wall-time breakdown of an SDFB + RK3 step.

Monkeypatches scheme methods with synchronizing timers arranged in a call
stack, so nested phases are reported as children of their caller.
Synchronization inflates the total somewhat (many small kernels), but the
relative shares show where GPU time goes.
"""
import argparse
import collections
import time

import cupy as cp

from spd.spd_simulator import SPD_Simulator as SDFB_Simulator
import spd.fallback.trouble_detection as td

times = collections.defaultdict(float)
calls = collections.defaultdict(int)
stack = []


def timed_fn(fn, label):
    def timed(*a, **k):
        cp.cuda.Stream.null.synchronize()
        stack.append(label)
        key = "/".join(stack)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            cp.cuda.Stream.null.synchronize()
            times[key] += time.perf_counter() - t
            calls[key] += 1
            stack.pop()
    return timed


def wrap(obj, name, label=None):
    setattr(obj, name, timed_fn(getattr(obj, name), label or name))


def wrap_mod(mod, name):
    setattr(mod, name, timed_fn(getattr(mod, name), name))


def report(steps, total):
    print(f"total {total:.3f} s -> {total / steps * 1e3:.2f} ms/step "
          f"(inflated by sync; phases below)")
    # Build the tree: children grouped under parent keys
    children = collections.defaultdict(list)
    for key in times:
        parent = "/".join(key.split("/")[:-1])
        children[parent].append(key)

    def show(key, depth):
        name = key.split("/")[-1]
        t = times[key]
        kids = sorted(children.get(key, []), key=lambda k: -times[k])
        self_t = t - sum(times[k] for k in kids)
        print(f"  {'  ' * depth}{name:<32s} {t / steps * 1e3:9.2f} ms/step "
              f"({t / total * 100:5.1f}%)  x{calls[key] // steps}"
              + (f"   [self {self_t / steps * 1e3:.2f}]" if kids else ""))
        for k in kids:
            show(k, depth + 1)

    top = sorted(children[""], key=lambda k: -times[k])
    accounted = sum(times[k] for k in top)
    for k in top:
        show(k, 0)
    print(f"  {'(untimed / python)':<34s} {(total - accounted) / steps * 1e3:9.2f} ms/step "
          f"({(total - accounted) / total * 100:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--R", type=int, default=1024, help="cells per dimension")
    ap.add_argument("--ndim", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    N = args.R // (args.p + 1)
    s = SDFB_Simulator(p=args.p, N=(N,) * args.ndim, use_cupy=True,
                       FB=True, time_integrator="rk3", verbose=False)
    s.perform_iterations(2)  # warmup (JIT, memory pool)

    lo = s.scheme          # FallbackScheme
    ho = lo.primary        # SD scheme

    # Step top level
    wrap(ho, "compute_dt")
    wrap(lo, "compute_update", "rk_stage")
    # Inside compute_update
    wrap(ho, "solve_faces")
    wrap(ho, "compute_fp_from_sp")
    ho.riemann_solver = timed_fn(ho.riemann_solver, "riemann_solver(SD)")
    wrap(ho, "switch_to_finite_volume")
    wrap(ho, "compute_cv_from_sp_fv")
    wrap(ho, "transpose_to_fv")
    wrap(ho, "integrate_faces")
    wrap(lo, "store_high_order_fluxes")
    wrap(lo, "compute_corrected_fluxes")
    wrap(lo, "compute_dudt", "compute_dudt(FV)")
    wrap(ho, "transpose_to_sd")
    wrap(ho, "compute_sp_from_cv")
    wrap(ho, "switch_to_high_order")
    # Inside compute_corrected_fluxes
    wrap(ho, "compute_primitives_cv")
    wrap(lo, "apply_fluxes")
    wrap(lo, "detect_troubles")
    wrap(lo, "compute_fluxes", "compute_fluxes(MUSCL)")
    wrap(lo, "correct_fluxes")
    # Inside detect_troubles / compute_fluxes
    wrap(lo, "fill_active_region")
    wrap(lo, "Boundaries", "Boundaries(FV)")
    wrap(lo, "Boundaries_scalar")
    wrap(lo, "solve_riemann_problem", "riemann(MUSCL)")
    wrap_mod(td, "neighborhood_extrema")
    wrap_mod(td, "compute_smooth_extrema")
    wrap_mod(td, "apply_blending")

    s.init_sim()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        s.compute_dt()
        s.perform_update()
    cp.cuda.Stream.null.synchronize()
    total = time.perf_counter() - t0

    print(f"\nSDFB p={args.p} (order {args.p + 1}), RK3, "
          f"{args.R}^{args.ndim} cells, {args.steps} steps")
    report(args.steps, total)
    pool = cp.get_default_memory_pool()
    print(f"mem pool used {pool.used_bytes() / 1e9:.2f} GB "
          f"/ total {pool.total_bytes() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
