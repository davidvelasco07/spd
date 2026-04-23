"""Block-based mesh forest for SMR/AMR.

Each block is a fixed-size logical tile (``NB[dim]`` SD elements per dim).
Blocks may live at different refinement levels in a single forest; the
physical element size scales as ``h[dim] = h0[dim] / 2**level``. Neighbor
relations account for level differences, and the forest enforces 2:1
balance (no two neighbors differ by more than one level).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Neighbor relation tags.
SAME = "same"        # neighbor at the same refinement level
FINER = "finer"      # neighbor one level finer; face has 2^(ndim-1) children
COARSER = "coarser"  # neighbor one level coarser
BC = "bc"            # domain boundary (non-periodic side handled by BC dict)

Neighbor = Tuple[Optional[int], str, Optional[int]]
# (jb, relation, sub_idx)
#   jb:        index of neighbor block, or None for BC
#   relation:  SAME | FINER | COARSER | BC
#   sub_idx:   for FINER, position of this finer block within its coarser
#              face (0 .. 2^(ndim-1)-1); for COARSER, position of this
#              block within the coarser neighbor's face. None otherwise.


_TOL = 1e-12


@dataclass
class MeshBlock:
    ib: int
    level: int                                  # refinement level (0 = coarsest)
    logical: Tuple[int, ...]                    # integer (ix, iy, iz) at `level`
    lim: Dict[str, Tuple[float, float]]         # physical bounds per dim
    h: Dict[str, float]                         # per-element physical size per dim
    # neighbors[dim][side] is a List[Neighbor]; multi-entry when the face
    # abuts a finer level (2^(ndim-1) children); otherwise a single entry.
    neighbors: Dict[str, List[List[Neighbor]]] = field(default_factory=dict)

    @property
    def scale(self) -> int:
        """Refinement factor relative to level 0 (2**level)."""
        return 1 << self.level


class BlockForest:
    """Forest of mesh blocks (possibly at multiple refinement levels)."""

    def __init__(self, ndim: int, dims: Dict[str, int],
                 NB: Dict[str, int], bc: Dict[str, Tuple[str, str]],
                 domain_lim: Dict[str, Tuple[float, float]]):
        self.ndim = ndim
        self.dims = dims
        self.NB = NB
        self.bc = bc
        self.domain_lim = domain_lim
        self.blocks: List[MeshBlock] = []

    @property
    def Nblocks(self) -> int:
        return len(self.blocks)

    @property
    def max_level(self) -> int:
        return max((b.level for b in self.blocks), default=0)

    # ------------------------------------------------------------------ ctors
    @classmethod
    def uniform_root(cls, ndim, dims, lim, N, bc) -> "BlockForest":
        """Single-block forest covering the (per-rank) domain."""
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
        """Build a uniform grid of same-level blocks tiling the domain."""
        forest = cls(ndim, dims, NB, bc, dict(lim))
        block_len = {d: (lim[d][1] - lim[d][0]) / Nblocks_per_dim[d] for d in dims}
        h = {d: block_len[d] / NB[d] for d in dims}

        dim_keys = list(dims.keys())
        Nbx = Nblocks_per_dim.get("x", 1)
        Nby = Nblocks_per_dim.get("y", 1)
        Nbz = Nblocks_per_dim.get("z", 1)

        iz_range = range(Nbz) if "z" in dims else range(1)
        iy_range = range(Nby) if "y" in dims else range(1)
        ix_range = range(Nbx)
        ib = 0
        for iz in iz_range:
            for iy in iy_range:
                for ix in ix_range:
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
                    ib += 1
        forest._rebuild_neighbors()
        return forest

    # ------------------------------------------------------ neighbor discovery
    def _rebuild_neighbors(self) -> None:
        """Recompute ``block.neighbors`` for every block.

        Uses physical extent comparison: two blocks are adjacent if their
        faces coincide in one dim and overlap in the others. Periodic BC
        wraps the face position across the domain. O(Nblocks^2); fine for
        static SMR, can be accelerated with a spatial index later.
        """
        for b in self.blocks:
            b.ib = self.blocks.index(b)    # keep ib consistent with list index
            b.neighbors = {d: [[], []] for d in self.dims}

        # Precompute block face positions to handle periodic wrap cheaply.
        def _faces_in(dim, side, lo, hi):
            face = lo if side == 0 else hi
            dlo, dhi = self.domain_lim[dim]
            candidates = [face]
            # Periodic wrap: the face at the domain edge is also "at" the
            # opposite edge for neighbor purposes.
            if self.bc[dim][side] == "periodic":
                if abs(face - dlo) < _TOL:
                    candidates.append(dhi)
                elif abs(face - dhi) < _TOL:
                    candidates.append(dlo)
            return candidates

        for ib, block in enumerate(self.blocks):
            for dim in self.dims:
                for side in (0, 1):
                    face_candidates = _faces_in(
                        dim, side, block.lim[dim][0], block.lim[dim][1]
                    )
                    found_any = False
                    for jb, other in enumerate(self.blocks):
                        # Self-matches are allowed and happen naturally when
                        # the domain is covered by a single block with
                        # periodic BC (left face wraps to own right face).
                        other_face = other.lim[dim][1] if side == 0 else other.lim[dim][0]
                        if not any(abs(other_face - f) < _TOL for f in face_candidates):
                            continue
                        # Check overlap in the other dims.
                        overlap = True
                        for od in self.dims:
                            if od == dim:
                                continue
                            if (block.lim[od][1] <= other.lim[od][0] + _TOL or
                                block.lim[od][0] >= other.lim[od][1] - _TOL):
                                overlap = False
                                break
                        if not overlap:
                            continue
                        # Level difference -> relation. Level gaps > 1 are
                        # tagged FINER/COARSER with sub_idx=None; it's the
                        # job of enforce_2to1_balance to find and refine
                        # them. Time-stepping kernels may only assume a
                        # 2:1-balanced forest.
                        dlev = other.level - block.level
                        if dlev == 0:
                            rel = SAME
                            sub = None
                        elif dlev > 0:
                            rel = FINER
                            sub = (self._sub_face_index(block, other, dim)
                                   if dlev == 1 else None)
                        else:
                            rel = COARSER
                            sub = (self._sub_face_index(other, block, dim)
                                   if dlev == -1 else None)
                        block.neighbors[dim][side].append((jb, rel, sub))
                        found_any = True
                    if not found_any:
                        # Domain boundary.
                        block.neighbors[dim][side].append((None, BC, None))

    def _sub_face_index(self, coarse: MeshBlock, fine: MeshBlock, dim: str) -> int:
        """Which sub-face of `coarse` along `dim` does `fine` cover?

        Returns a row-major index into the 2^(ndim-1) sub-faces using the
        non-`dim` dimensions in the forest's dims ordering (x, then y for 3D).
        """
        other_dims = [d for d in self.dims if d != dim]
        idx = 0
        stride = 1
        for od in other_dims:
            clo, chi = coarse.lim[od]
            flo, _ = fine.lim[od]
            half = 0.5 * (clo + chi)
            bit = 1 if flo >= half - _TOL else 0
            idx += bit * stride
            stride *= 2
        return idx

    # ---------------------------------------------------- refine / derefine
    def refine_blocks(self, ibs: List[int]) -> List[int]:
        """Refine several blocks (convenience wrapper around `refine_block`).

        Handles the index-shift gotcha: ibs are resolved to concrete block
        objects up front, so the order of refinement does not matter.
        Returns the final list indices of every new child block.
        """
        targets = [self.blocks[i] for i in ibs]
        produced: List[MeshBlock] = []
        for block in targets:
            ib = self.blocks.index(block)
            child_ibs = self.refine_block(ib)
            produced.extend(self.blocks[c] for c in child_ibs)
        return [self.blocks.index(c) for c in produced]

    def refine_block(self, ib: int) -> List[int]:
        """Replace block ``ib`` with 2**ndim children at level+1.

        Returns the new child ibs (in row-major iz,iy,ix order). Does NOT
        enforce 2:1 balance — call ``enforce_2to1_balance()`` afterwards.
        """
        parent = self.blocks[ib]
        dim_keys = list(self.dims.keys())
        child_h = {d: parent.h[d] * 0.5 for d in self.dims}
        new_level = parent.level + 1

        children: List[MeshBlock] = []
        iz_range = range(2) if "z" in self.dims else range(1)
        iy_range = range(2) if "y" in self.dims else range(1)
        ix_range = range(2)
        for iz in iz_range:
            for iy in iy_range:
                for ix in ix_range:
                    child_logical = tuple(
                        2 * parent.logical[dim_keys.index(d)] + {"x": ix, "y": iy, "z": iz}[d]
                        for d in dim_keys
                    )
                    b_lim = {}
                    for d in self.dims:
                        lo, hi = parent.lim[d]
                        mid = 0.5 * (lo + hi)
                        sel = {"x": ix, "y": iy, "z": iz}[d]
                        b_lim[d] = (lo, mid) if sel == 0 else (mid, hi)
                    children.append(MeshBlock(
                        ib=-1,                      # placeholder, fixed by _rebuild_neighbors
                        level=new_level,
                        logical=child_logical,
                        lim=b_lim,
                        h=dict(child_h),
                    ))

        # Remove the parent; append children at the end.
        self.blocks.pop(ib)
        start = len(self.blocks)
        self.blocks.extend(children)
        new_ibs = list(range(start, start + len(children)))
        self._rebuild_neighbors()
        # After _rebuild_neighbors the ib field has been updated; but the
        # list indices of the children might not match `new_ibs` because the
        # list was reindexed. Return the new list indices of the children.
        return [self.blocks.index(c) for c in children]

    def derefine_block(self, ibs: List[int]) -> int:
        """Replace 2**ndim sibling blocks with their common parent.

        Returns the ib of the new parent. Raises if the given blocks are
        not sibling children at the same level.
        """
        if len(ibs) != 2 ** self.ndim:
            raise ValueError(f"derefine expects 2**ndim={2**self.ndim} siblings; got {len(ibs)}")
        siblings = [self.blocks[i] for i in ibs]
        levels = {b.level for b in siblings}
        if len(levels) != 1 or siblings[0].level == 0:
            raise ValueError("derefine requires all siblings at a common level > 0")
        # Check they really are siblings: their logical coords at level-1
        # should all be the same.
        parent_logical = tuple(c >> 1 for c in siblings[0].logical)
        for b in siblings[1:]:
            if tuple(c >> 1 for c in b.logical) != parent_logical:
                raise ValueError("blocks are not siblings")
        # Build the parent by unioning the children's physical extents.
        parent_lim = {
            d: (min(b.lim[d][0] for b in siblings),
                max(b.lim[d][1] for b in siblings))
            for d in self.dims
        }
        parent_h = {d: 2.0 * siblings[0].h[d] for d in self.dims}
        parent = MeshBlock(
            ib=-1,
            level=siblings[0].level - 1,
            logical=parent_logical,
            lim=parent_lim,
            h=parent_h,
        )
        # Remove siblings (largest index first to avoid shift).
        for i in sorted(ibs, reverse=True):
            self.blocks.pop(i)
        self.blocks.append(parent)
        self._rebuild_neighbors()
        return self.blocks.index(parent)

    # ------------------------------------------------------------ balance
    def enforce_2to1_balance(self) -> int:
        """Cascade-refine any block whose neighbor is more than one level
        deeper. Returns the number of blocks refined."""
        total = 0
        while True:
            to_refine = set()
            for ib, block in enumerate(self.blocks):
                for dim in self.dims:
                    for side in (0, 1):
                        for (jb, rel, _sub) in block.neighbors[dim][side]:
                            if jb is None:
                                continue
                            other = self.blocks[jb]
                            if other.level - block.level > 1:
                                to_refine.add(ib)
            if not to_refine:
                return total
            # Refine one at a time (indexes shift on each refine).
            # Grab an ib whose block still exists.
            # We must refine a SPECIFIC block, not a stale index, so refine
            # the first one and re-scan.
            ib = next(iter(to_refine))
            self.refine_block(ib)
            total += 1
