"""Block-based mesh forest for SMR/AMR.

Phase 0: scaffolding only. A forest currently contains a single root block
covering the (per-rank) domain. Phase 1 generalizes to Nblocks>1 uniform,
Phase 2 introduces refinement levels.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Neighbor relation tags.
SAME = "same"        # neighbor at the same refinement level
FINER = "finer"      # neighbor one level finer (has 2^(ndim-1) children on this face)
COARSER = "coarser"  # neighbor one level coarser
BC = "bc"            # domain boundary (non-periodic side handled by BC dict)

Neighbor = Tuple[Optional[int], str, Optional[int]]
# (jb, relation, child_index)
#   jb:            index of neighbor block, or None for BC
#   relation:      SAME | FINER | COARSER | BC
#   child_index:   which sibling among the 2^(ndim-1) fine neighbors we face,
#                  or None for SAME/COARSER/BC.


@dataclass
class MeshBlock:
    ib: int
    level: int                                  # refinement level (0 = coarsest)
    logical: Tuple[int, ...]                    # integer (ix, iy, iz) at `level`
    lim: Dict[str, Tuple[float, float]]         # physical bounds per dim
    h: Dict[str, float]                         # per-element physical size per dim
    neighbors: Dict[str, List[Neighbor]] = field(default_factory=dict)

    @property
    def scale(self) -> int:
        """Refinement factor relative to level 0 (2^level)."""
        return 1 << self.level


class BlockForest:
    """Forest of mesh blocks (one per rank in Phase 0)."""

    def __init__(self, ndim: int, dims: Dict[str, int],
                 NB: Dict[str, int], bc: Dict[str, Tuple[str, str]]):
        self.ndim = ndim
        self.dims = dims
        self.NB = NB            # per-block element counts per dim
        self.bc = bc            # boundary condition tags per dim
        self.blocks: List[MeshBlock] = []

    @property
    def Nblocks(self) -> int:
        return len(self.blocks)

    @classmethod
    def uniform_root(cls,
                     ndim: int,
                     dims: Dict[str, int],
                     lim: Dict[str, Tuple[float, float]],
                     N: Dict[str, int],
                     bc: Dict[str, Tuple[str, str]]) -> "BlockForest":
        """Single-block forest covering the (per-rank) domain. Periodic
        sides are self-neighbors; non-periodic sides are BC-tagged."""
        Nblocks_per_dim = {d: 1 for d in dims}
        NB = {d: N[d] for d in dims}
        return cls.uniform_grid(ndim, dims, lim, NB, Nblocks_per_dim, bc)

    @classmethod
    def uniform_grid(cls,
                     ndim: int,
                     dims: Dict[str, int],
                     lim: Dict[str, Tuple[float, float]],
                     NB: Dict[str, int],
                     Nblocks_per_dim: Dict[str, int],
                     bc: Dict[str, Tuple[str, str]]) -> "BlockForest":
        """Build a uniform grid of same-level blocks tiling the (per-rank)
        domain.

        Each block has ``NB[dim]`` elements per dim. Same-level intra-grid
        faces become ``SAME`` neighbors. At the grid edges, periodic BC wraps
        to the block on the opposite edge (``SAME``), non-periodic BC is
        tagged ``BC``.
        """
        forest = cls(ndim, dims, NB, bc)
        dim_keys = list(dims.keys())  # preserves x,y,z ordering
        Nbx = Nblocks_per_dim.get("x", 1)
        Nby = Nblocks_per_dim.get("y", 1)
        Nbz = Nblocks_per_dim.get("z", 1)
        block_len = {d: (lim[d][1] - lim[d][0]) / Nblocks_per_dim[d] for d in dims}
        h = {d: block_len[d] / NB[d] for d in dims}

        def grid_index(ix: int, iy: int, iz: int) -> int:
            return iz * Nby * Nbx + iy * Nbx + ix

        # First pass: create all blocks (without neighbors).
        ranges = {"x": range(Nbx), "y": range(Nby), "z": range(Nbz)}
        iz_range = ranges["z"] if "z" in dims else range(1)
        iy_range = ranges["y"] if "y" in dims else range(1)
        ix_range = ranges["x"]
        for iz in iz_range:
            for iy in iy_range:
                for ix in ix_range:
                    ib = grid_index(ix, iy, iz)
                    logical = tuple(
                        {"x": ix, "y": iy, "z": iz}[d] for d in dim_keys
                    )
                    b_lim = {}
                    for d in dims:
                        g = {"x": ix, "y": iy, "z": iz}[d]
                        start = lim[d][0] + g * block_len[d]
                        b_lim[d] = (start, start + block_len[d])
                    forest.blocks.append(MeshBlock(
                        ib=ib, level=0, logical=logical,
                        lim=b_lim, h=dict(h),
                    ))

        # Second pass: wire same-level neighbors.
        Nb_per = {"x": Nbx, "y": Nby, "z": Nbz}
        for iz in iz_range:
            for iy in iy_range:
                for ix in ix_range:
                    ib = grid_index(ix, iy, iz)
                    block = forest.blocks[ib]
                    coords = {"x": ix, "y": iy, "z": iz}
                    for d in dims:
                        block.neighbors[d] = []
                        for side in (0, 1):
                            c = coords[d]
                            step = -1 if side == 0 else 1
                            c_new = c + step
                            n_d = Nb_per[d]
                            if 0 <= c_new < n_d:
                                # Interior grid neighbor.
                                nc = dict(coords); nc[d] = c_new
                                jb = grid_index(nc["x"], nc["y"], nc["z"])
                                block.neighbors[d].append((jb, SAME, None))
                            elif bc[d][side] == "periodic":
                                # Wrap across the grid.
                                nc = dict(coords); nc[d] = c_new % n_d
                                jb = grid_index(nc["x"], nc["y"], nc["z"])
                                block.neighbors[d].append((jb, SAME, None))
                            else:
                                block.neighbors[d].append((None, BC, None))
        return forest
