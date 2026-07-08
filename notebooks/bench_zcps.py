"""Measure zone-cycles per second for one configuration (one process per run).

The spd package is imported from the directory in the SPD_REPO environment
variable, so the same script can benchmark different checkouts.

Prints a single JSON line with the result.
"""
import argparse
import json
import os
import sys
import time

repo = os.environ.get("SPD_REPO")
if repo:
    sys.path.insert(0, repo)

import spd  # noqa: E402
import cupy as cp  # noqa: E402
from spd.spd_simulator import SPD_Simulator as SDFB_Simulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--R", type=int, required=True, help="cells per dimension")
    ap.add_argument("--ndim", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--sim", choices=["sd", "sdfb"], required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--integrator", default="rk3")
    args = ap.parse_args()

    assert args.R % (args.p + 1) == 0
    N = args.R // (args.p + 1)

    s = SDFB_Simulator(
        p=args.p,
        N=(N,) * args.ndim,
        use_cupy=True,
        FB=(args.sim == "sdfb"),
        time_integrator=args.integrator,
        verbose=False,
    )
    s.perform_iterations(args.warmup)

    s.init_sim()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        s.compute_dt()
        s.perform_update()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    result = {
        "repo": os.path.realpath(repo or os.path.dirname(os.path.dirname(spd.__file__))),
        "sim": args.sim,
        "p": args.p,
        "R": args.R,
        "ndim": args.ndim,
        "steps": args.steps,
        "ms_per_step": elapsed / args.steps * 1e3,
        "zcps": args.R ** args.ndim * args.steps / elapsed,
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
