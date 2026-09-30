"""Typed entity construction options."""

from typing import Annotated

from pydantic import ConfigDict, Field, field_validator

import genesis as gs
from genesis.engine.materials import Kinematic, Rigid
from genesis.engine.materials.base import Material

from .morphs import Morph
from .options import Options
from .surfaces import Surface


class EntityOptions(Options):
    """Common options for constructing an entity.

    Concrete subclasses select the entity family and validate compatible materials. Pass an instance to
    ``Scene.add_entity(options=...)``. The scene copies the options before resolving geometry and visual defaults.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...]
        Geometry or asset description. A sequence declares rigid geometry variants across environments.
    material : Material
        Physical properties compatible with the concrete entity family.
    surface : Surface | None, optional
        Appearance. If None, use the default surface.
    visualize_contact : bool, optional
        Whether to display contact forces. Defaults to False.
    vis_mode : str | None, optional
        Visualization mode override. If None, use the surface and material defaults.
    name : str | None, optional
        Unique name within the scene. If None, generate one from the morph and entity identity.
    """

    model_config = ConfigDict(validate_assignment=True, revalidate_instances="always")

    morph: Morph | Annotated[tuple[Morph, ...], Field(strict=False)]
    material: Material
    surface: Surface | None = None
    visualize_contact: bool = False
    vis_mode: str | None = None
    name: str | None = None


class RigidEntityOptions(EntityOptions):
    """Construct a rigid entity with compatible rigid material properties.

    Parameters follow ``EntityOptions``. The default material is ``gs.materials.Rigid()``.
    """

    material: Rigid = Field(default_factory=Rigid)

    @field_validator("material", mode="before")
    @classmethod
    def validate_material(cls, material):
        if not isinstance(material, (Rigid, dict)):
            gs.raise_exception("RigidEntityOptions requires a rigid material.")
        return material


class KinematicEntityOptions(EntityOptions):
    """Construct a visualization-only entity with kinematic material properties.

    Parameters follow ``EntityOptions``. The default material is ``gs.materials.Kinematic()``.
    """

    material: Kinematic = Field(default_factory=Kinematic)

    @field_validator("material")
    @classmethod
    def validate_material(cls, material):
        if isinstance(material, Rigid):
            gs.raise_exception(
                "KinematicEntityOptions requires a kinematic material. Use RigidEntityOptions for physics."
            )
        return material
