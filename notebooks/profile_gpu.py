"""Profile a GPU run of the SDFB simulator to find hotspots."""
import argparse
import cProfile
import pstats
import sys

import numpy as np
import cupy as cp

from spd.spd_simulator import SPD_Simulator as SDFB_Simulator


def make_sim(p, N, FB, viscosity):
    return SDFB_Simulator(
        p=p,
        N=(N, N, N),
        use_cupy=True,
        FB=FB,
        viscosity=viscosity,
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--FB", action="store_true")
    ap.add_argument("--viscosity", action="store_true")
    ap.add_argument("--sort", default="cumulative")
    args = ap.parse_args()

    s = make_sim(args.p, args.N, args.FB, args.viscosity)
    # Warmup (kernel compilation, memory pool growth)
    s.perform_iterations(2)
    cp.cuda.Stream.null.synchronize()

    pr = cProfile.Profile()
    pr.enable()
    s.perform_iterations(args.steps)
    cp.cuda.Stream.null.synchronize()
    pr.disable()

    stats = pstats.Stats(pr, stream=sys.stdout)
    stats.strip_dirs().sort_stats(args.sort).print_stats(40)

    pool = cp.get_default_memory_pool()
    print(f"mem pool used: {pool.used_bytes()/1e9:.2f} GB, total: {pool.total_bytes()/1e9:.2f} GB")
    print(f"domain size: {s.domain_size}, cost/step: {s.cost_per_step*1e3:.1f} ms")


if __name__ == "__main__":
    main()
