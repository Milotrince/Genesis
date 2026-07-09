"""Dynamic GPU geometry residency pool (Phase 3).

A geometry pool reserves, at build time, a fixed block of geometry slots appended to the solver's global
geometry/vertex/face/edge/SDF-cell arrays. The block starts empty (its device rows are zero, which is inert:
a zero quaternion rotates every AABB corner to a finite point and ``contype == conaffinity == 0`` yields no
collision pairs) and is filled at runtime by ``RigidSolver.set_active_object``, which uploads a processed
object into a free slot and rebinds environments to it via the same ``_bind_link_variant`` machinery
heterogeneous variants use.

The pool is declared **per entity** (``scene.add_entity(morph=base, geom_pool=GeomPoolOptions(...))``): each
poolable entity contributes one ``GeomPoolSegment`` whose slots are bound at build to that entity's base
link. This makes "which link does a slot serve" known at build (so collision-pair validity precomputes
cleanly) and keeps pool slots the same per-link geom-range kind as heterogeneous variants. ``GeometryPool``
aggregates every segment into one contiguous reserved block, because the device fields are single global
arrays; each segment owns a disjoint sub-range.
"""

from dataclasses import dataclass, field

# The global-array keys a slot reserves a uniform per-slot budget in. A slot's free-vertex state budget
# equals its collision-vertex budget: each collision vertex of a movable pooled geom owns one world state row.
# "vgeom" reserves visual-geom rows for visual pooling (0 budget = collision-only pool). Rigid visual pooling
# needs no vvert/vface device rows: the render/AABB path uses the host trimesh + the device vgeom pose.
_ARRAY_KEYS = ("geom", "vert", "face", "edge", "cell", "free_vert", "vgeom")


@dataclass(frozen=True)
class GeomPoolSlotRanges:
    """The reserved index range of one pool slot in each global array (half-open ``[start, end)``)."""

    geom: tuple[int, int]
    vert: tuple[int, int]
    face: tuple[int, int]
    edge: tuple[int, int]
    cell: tuple[int, int]
    free_vert: tuple[int, int]
    vgeom: tuple[int, int]


@dataclass
class GeomPoolSegment:
    """One poolable entity's contiguous sub-block of slots, bound to a single link.

    Owns both the static slot *layout* (index ranges) and the runtime residency table (which object occupies
    each slot, per-slot refcount, and LRU order). Residency is pure host state; the device only ever sees
    which geom range an environment's link is bound to, exactly as for heterogeneous variants.
    """

    entity_idx: int
    link_idx: int
    entity: object  # the owning RigidEntity (its collision identity is shared by all this segment's slots)
    n_slots: int
    per_slot: dict  # array key -> uniform per-slot budget
    slots: list  # list[GeomPoolSlotRanges], one per slot

    # Residency bookkeeping (Stage 2). A slot is free when resident_key is None. A slot is evictable when its
    # refcount is 0 (no env currently bound) and it is not pinned.
    resident_key: list = field(default_factory=list)  # slot -> object key | None
    resident_morph: list = field(default_factory=list)  # slot -> the resident morph (kept alive so id() stays valid)
    refcount: list = field(default_factory=list)  # slot -> #envs currently bound
    pinned: list = field(default_factory=list)  # slot -> bool (excluded from eviction)
    inertial: list = field(default_factory=list)  # slot -> LinkInertial-like of the resident object
    n_live_geoms: list = field(default_factory=list)  # slot -> #sub-geoms actually uploaded (<= geom budget)
    lru: list = field(default_factory=list)  # slot indices, least-recently-used first
    key_to_slot: dict = field(default_factory=dict)  # object key -> slot
    env_slot: dict = field(default_factory=dict)  # env index -> slot it is currently bound to (else base morph)
    vgeom_placeholders: list = field(default_factory=list)  # slot -> RigidVisGeom placeholder (visual pooling)
    # id(morph) -> (cg_infos, vg_infos) for objects declared via GeomPoolOptions.objects, processed once at
    # build. set_active_object reuses this so a declared object's (possibly nondeterministic) geometry always
    # matches the budgets derived from it. The GeomPoolOptions holds the morphs, so their ids stay valid.
    declared: dict = field(default_factory=dict)

    def __post_init__(self):
        self.resident_key = [None] * self.n_slots
        self.resident_morph = [None] * self.n_slots
        self.refcount = [0] * self.n_slots
        self.pinned = [False] * self.n_slots
        self.inertial = [None] * self.n_slots
        self.n_live_geoms = [0] * self.n_slots
        self.lru = list(range(self.n_slots))
        self.vgeom_placeholders = [None] * self.n_slots

    def total(self, key: str) -> int:
        return self.n_slots * self.per_slot[key]


class GeometryPool:
    """Aggregate of every poolable entity's segment; owns the global reserved-block layout.

    Parameters
    ----------
    base : dict[str, int]
        First free index in each global array after the real entities (where the reserved block begins).
    """

    def __init__(self, base: dict[str, int]):
        self._cursor = dict(base)
        self._segments: list[GeomPoolSegment] = []
        self._by_entity: dict[int, GeomPoolSegment] = {}

    def add_segment(
        self, entity_idx: int, link_idx: int, entity, options, per_slot=None, n_slots=None
    ) -> GeomPoolSegment:
        """Reserve a contiguous slot sub-block for one entity's pool and return its segment.

        Segments are appended in call order; each advances the shared cursor so sub-blocks stay disjoint.
        `per_slot` / `n_slots`, when given, override the options' explicit budgets (used when the caller has
        derived them from an object catalog); otherwise they come straight from `options`.
        """
        if per_slot is None:
            per_slot = {
                "geom": int(options.max_geoms_per_slot),
                "vert": int(options.max_verts_per_slot),
                "face": int(options.max_faces_per_slot),
                "edge": int(options.max_edges_per_slot),
                "cell": int(options.max_cells_per_slot),
                "free_vert": int(options.max_verts_per_slot),
                "vgeom": int(options.max_vgeoms_per_slot),
            }
        if n_slots is None:
            n_slots = int(options.n_slots)

        slots = []
        for _ in range(n_slots):
            ranges = {}
            for key in _ARRAY_KEYS:
                start = self._cursor[key]
                self._cursor[key] = start + per_slot[key]
                ranges[key] = (start, self._cursor[key])
            slots.append(GeomPoolSlotRanges(**ranges))

        segment = GeomPoolSegment(
            entity_idx=entity_idx, link_idx=link_idx, entity=entity, n_slots=n_slots, per_slot=per_slot, slots=slots
        )
        self._segments.append(segment)
        self._by_entity[entity_idx] = segment
        return segment

    def segment_for_entity(self, entity_idx: int) -> GeomPoolSegment | None:
        return self._by_entity.get(entity_idx)

    @property
    def segments(self) -> list[GeomPoolSegment]:
        return self._segments

    # Total reserved rows for one array key across every segment (added to the solver's allocation sizes).
    def total(self, key: str) -> int:
        return sum(seg.total(key) for seg in self._segments)

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

    @property
    def n_vgeoms(self) -> int:
        return self.total("vgeom")
