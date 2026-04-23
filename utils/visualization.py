import matplotlib.pyplot as plt
import numpy as np
from simulator import Simulator
from typing import Tuple


def plot_fields(s: Simulator,
                M: np.ndarray,
                figsize: Tuple = (6, 4),
                **kwargs):
    fig, axs = plt.subplots(1, s.nvar, figsize=(figsize[0]*s.nvar, figsize[1]))
    for var in range(s.nvar):
        plt.sca(axs[var])
        plot_field(s, M, var, **kwargs)


def plot_field(s: Simulator,
               M: np.ndarray,
               var: int,
               dim: str = "z",
               transpose: bool = True,
               regular: bool = False,
               integrate: bool = False,
               show_blocks: bool = False,
               **kwargs):
    """Plot variable ``var`` of ``M``.

    ``M`` is the per-meshblock SD layout [nvar, Nb, cells, pts]. Each block
    is drawn at its own physical extent so the plot is correct for both
    uniform multi-block grids and AMR (mixed refinement levels). The
    colormap range is computed globally across all blocks so patches share
    a scale.

    Parameters
    ----------
    show_blocks : overlay black rectangles on block boundaries.
    transpose   : retained for API compatibility. Per-block plotting always
                  reshapes to FV layout via ``s.block_to_fv``.
    """
    if regular:
        M = s.regular_mesh(M)
    Mvar = M[var]                 # [Nb, cells, pts]

    # Shared color range across blocks so patches line up.
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None:
        vmin = float(Mvar.min())
    if vmax is None:
        vmax = float(Mvar.max())

    if s.ndim == 1:
        for ib, block in enumerate(s.forest.blocks):
            flat = s.block_to_fv(Mvar[ib])        # [NxB*px]
            xlo, xhi = block.lim["x"]
            x = np.linspace(xlo, xhi, flat.size)
            plt.plot(x, flat, **kwargs)
            if show_blocks and ib + 1 < s.forest.Nblocks:
                plt.axvline(xhi, color="k", lw=0.5, alpha=0.5)
        plt.title(s.variables[var])

    elif s.ndim == 2:
        mesh = None
        for ib, block in enumerate(s.forest.blocks):
            flat = s.block_to_fv(Mvar[ib])        # [NyB*py, NxB*px]
            xlo, xhi = block.lim["x"]
            ylo, yhi = block.lim["y"]
            ny, nx = flat.shape
            x = np.linspace(xlo, xhi, nx + 1)
            y = np.linspace(ylo, yhi, ny + 1)
            mesh = plt.pcolormesh(x, y, flat, vmin=vmin, vmax=vmax, **kwargs)
            if show_blocks:
                plt.plot(
                    [xlo, xhi, xhi, xlo, xlo],
                    [ylo, ylo, yhi, yhi, ylo],
                    "k-", lw=0.5,
                )
        plt.colorbar(mesh)
        plt.title(s.variables[var])

    elif s.ndim == 3:
        # Pick a slice dim (dim kwarg) and plot blocks whose extent contains it.
        if dim == "z":
            slice_dim = "z"
            out_dims = ("x", "y")
        else:
            slice_dim = "y"
            out_dims = ("x", "z")
        slab_coord = 0.5 * (s.lim[slice_dim][0] + s.lim[slice_dim][1])
        mesh = None
        for ib, block in enumerate(s.forest.blocks):
            lo, hi = block.lim[slice_dim]
            if integrate:
                pass  # take full block
            else:
                if not (lo <= slab_coord <= hi):
                    continue
            flat = s.block_to_fv(Mvar[ib])    # [NzB*pz, NyB*py, NxB*px]
            # Pick slice along slice_dim or integrate.
            axis_map = {"z": 0, "y": 1, "x": 2}
            axis = axis_map[slice_dim]
            if integrate:
                slab = flat.sum(axis=axis) / flat.shape[axis]
            else:
                n_slice = flat.shape[axis]
                # Interpolate slab_coord to integer index along axis.
                t = (slab_coord - lo) / (hi - lo)
                idx = min(int(t * n_slice), n_slice - 1)
                slab = np.take(flat, idx, axis=axis)
            # slab has two axes; map to physical coords.
            xlo_a, xhi_a = block.lim[out_dims[0]]
            xlo_b, xhi_b = block.lim[out_dims[1]]
            n_b, n_a = slab.shape
            x = np.linspace(xlo_a, xhi_a, n_a + 1)
            y = np.linspace(xlo_b, xhi_b, n_b + 1)
            mesh = plt.pcolormesh(x, y, slab, vmin=vmin, vmax=vmax, **kwargs)
            if show_blocks:
                plt.plot(
                    [xlo_a, xhi_a, xhi_a, xlo_a, xlo_a],
                    [xlo_b, xlo_b, xhi_b, xhi_b, xlo_b],
                    "k-", lw=0.5,
                )
        if mesh is not None:
            plt.colorbar(mesh)
        plt.title(s.variables[var])


def plot_block_layout(s: Simulator, ax=None, **kwargs):
    """Draw only the meshblock boundaries (useful to inspect the forest).

    For 1D/2D; colors blocks by refinement level.
    """
    import matplotlib.patches as patches
    if ax is None:
        ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    for block in s.forest.blocks:
        color = cmap(block.level % 10)
        if s.ndim == 1:
            xlo, xhi = block.lim["x"]
            ax.axvline(xlo, color=color, **kwargs)
            ax.axvline(xhi, color=color, **kwargs)
        elif s.ndim == 2:
            xlo, xhi = block.lim["x"]
            ylo, yhi = block.lim["y"]
            rect = patches.Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo,
                fill=False, edgecolor=color, **kwargs,
            )
            ax.add_patch(rect)
            ax.text(
                0.5 * (xlo + xhi), 0.5 * (ylo + yhi), str(block.ib),
                ha="center", va="center", fontsize=8, color=color,
            )
    if s.ndim == 2:
        ax.set_xlim(s.lim["x"])
        ax.set_ylim(s.lim["y"])
        ax.set_aspect("equal")
