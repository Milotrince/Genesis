"""Dynamic GPU geometry residency pool (Phase 3).

A ``GeometryPool`` reserves, at build time, a fixed block of uniform geometry slots appended to the solver's
global geometry/vertex/face/edge/SDF-cell arrays. The block starts empty (its device rows are zero, which is
inert: a zero quaternion rotates to a finite point and ``contype == conaffinity == 0`` yields no collision
pairs) and is filled at runtime by ``RigidSolver.set_active_object``, which uploads a processed object into a
free slot and rebinds environments to it via the same ``_bind_link_variant`` machinery heterogeneous variants
use.

Stage 1 (this file) computes and owns the static slot *layout* -- the contiguous index range each slot
occupies in every global array. Residency bookkeeping (which object is in which slot, refcounts, LRU
eviction) is layered on top in Stage 2.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeomPoolSlotRanges:
    """The reserved index range of one pool slot in each global array (half-open ``[start, end)``)."""

    geom: tuple[int, int]
    vert: tuple[int, int]
    face: tuple[int, int]
    edge: tuple[int, int]
    cell: tuple[int, int]
    free_vert: tuple[int, int]


class GeometryPool:
    """Static layout of a build-time-reserved geometry pool block.

    Parameters
    ----------
    options : GeomPoolOptions
        The per-slot budgets and slot count.
    base : dict[str, int]
        First reserved index in each global array, keyed ``geom``/``vert``/``face``/``edge``/``cell``/
        ``free_vert``. Captured by the solver right after the real entities are counted and before the pool
        block is appended.
    """

    def __init__(self, options, base: dict[str, int]):
        self.options = options
        self.n_slots = int(options.n_slots)
        self._base = dict(base)

        # Per-slot uniform budgets. A slot's free-vertex state budget equals its collision-vertex budget: each
        # collision vertex of a movable pooled geom owns exactly one world-space state row.
        self._per_slot = {
            "geom": int(options.max_geoms_per_slot),
            "vert": int(options.max_verts_per_slot),
            "face": int(options.max_faces_per_slot),
            "edge": int(options.max_edges_per_slot),
            "cell": int(options.max_cells_per_slot),
            "free_vert": int(options.max_verts_per_slot),
        }

        # Partition the reserved block into slot ranges. Uniform per-slot budgets make this a trivial
        # contiguous partition, so the runtime free-list is just a stack of slot indices (no fragmentation).
        self._slots: list[GeomPoolSlotRanges] = []
        for s in range(self.n_slots):
            ranges = {
                key: (self._base[key] + s * width, self._base[key] + (s + 1) * width)
                for key, width in self._per_slot.items()
            }
            self._slots.append(GeomPoolSlotRanges(**ranges))

    # ------------------------------------------------------------------------------------
    # Total reserved counts (added to the solver's global allocation sizes at build).
    # ------------------------------------------------------------------------------------

    def total(self, key: str) -> int:
        """Total reserved rows for one array key across all slots."""
        return self.n_slots * self._per_slot[key]

    @property
    def n_geoms(self) -> int:
        return self.total("geom")

    @property
    def n_verts(self) -> int:
        return self.total("vert")

    @property
    def n_faces(self) -> int:
        return self.total("face")

    @property
    def n_edges(self) -> int:
        return self.total("edge")

    @property
    def n_cells(self) -> int:
        return self.total("cell")

    @property
    def n_free_verts(self) -> int:
        return self.total("free_vert")

    def slot_ranges(self, slot: int) -> GeomPoolSlotRanges:
        return self._slots[slot]
