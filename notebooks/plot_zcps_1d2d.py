"""Plot zone-cycles/s vs resolution in 1D and 2D: new vs previous version."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = [json.loads(l) for l in open(os.path.join(here, "zcps_results_1d2d.jsonl"))]

def get(ndim, repo_new, sim, p):
    pts = sorted(
        (d["R"], d["zcps"])
        for d in data
        if d["ndim"] == ndim and d["sim"] == sim and d["p"] == p
        and not d.get("failed")
        and (("baseline" not in d["repo"]) == repo_new)
    )
    return [r for r, _ in pts], [z for _, z in pts]

colors = {3: "tab:blue", 7: "tab:red"}
titles = {"sd": "SD (RK3)", "sdfb": "SDFB (RK3)"}

for ndim in (2, 1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, sim in zip(axes, ["sd", "sdfb"]):
        for p in (3, 7):
            order = p + 1
            for new, ls, marker in ((False, "--", "o"), (True, "-", "s")):
                R, z = get(ndim, new, sim, p)
                label = f"{titles[sim].split()[0]}{order} " + ("new" if new else "previous")
                ax.plot(R, z, ls, marker=marker, color=colors[p], label=label,
                        lw=2, ms=6, alpha=1.0 if new else 0.55)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        Rs = sorted({d["R"] for d in data if d["ndim"] == ndim})
        if ndim == 2:
            ax.set_xticks(Rs, [f"${r}^2$" for r in Rs])
        else:
            ax.set_xticks(Rs, [f"$2^{{{r.bit_length()-1}}}$" for r in Rs])
        ax.set_xlabel("resolution (cells)")
        ax.set_title(titles[sim])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("zone-cycles per second")
    fig.suptitle(f"GPU performance (A100, {ndim}D hydro, RK3): optimized vs previous version", y=1.02)
    fig.tight_layout()
    out = os.path.join(here, f"zcps_performance_{ndim}d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
