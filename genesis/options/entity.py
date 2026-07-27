from genesis.engine.materials.base import Material

from .morphs import Morph
from .options import Options
from .surfaces import Surface


class EntityOptions(Options):
    """
    The full configuration of one entity (or one per-environment variant of a heterogeneous entity).

    Bundles a `morph` with its `material` and `surface` so `scene.add_entity` can take a structured spec instead of
    parallel arguments. Passing several `EntityOptions` to `add_entity` builds a single heterogeneous entity, one
    variant per instance; passing one builds an ordinary entity.
    """

    morph: Morph
    material: Material | None = None
    surface: Surface | None = None
