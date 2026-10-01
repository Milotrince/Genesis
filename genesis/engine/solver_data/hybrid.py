"""Resolved rigid/soft composition and particle associations for hybrid entities."""

from dataclasses import dataclass

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.entities.base_entity import EntityDescription
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface

from . import SolverData, SolverDescription


@dataclass(kw_only=True)
class HybridAssociationDescription:
    links_idx: np.ndarray
    geoms_idx: np.ndarray
    trans_local_to_global: np.ndarray
    quat_local_to_global: np.ndarray
    muscle_group: np.ndarray | None


@dataclass(kw_only=True)
class HybridEntityDescription(EntityDescription):
    morph: Morph
    surface: Surface
    name: str | None
    rigid_entity_idx: int
    soft_entity_idx: int
    association: HybridAssociationDescription | None


@dataclass(frozen=True, kw_only=True, eq=False)
class HybridData(SolverData):
    rigid_entity_idx: int
    soft_entity_idx: int
    info: qd.Field | None
    init_positions: qd.Field | None


@dataclass(frozen=True, kw_only=True)
class HybridDescription(SolverDescription[HybridData]):
    entity_idx: int
    entity: HybridEntityDescription
    init_positions: np.ndarray

    def allocate(self, owner) -> HybridData:
        association = self.entity.association
        info = init_positions = None
        if association is not None:
            info = qd.types.struct(
                link_idx=gs.qd_int,
                geom_idx=gs.qd_int,
                trans_local_to_global=gs.qd_vec3,
                quat_local_to_global=gs.qd_vec4,
            ).field(shape=len(association.links_idx), needs_grad=False, layout=qd.Layout.SOA)
            info.link_idx.from_numpy(association.links_idx)
            info.geom_idx.from_numpy(association.geoms_idx)
            info.trans_local_to_global.from_numpy(association.trans_local_to_global)
            info.quat_local_to_global.from_numpy(association.quat_local_to_global)
            init_positions = qd.field(dtype=gs.qd_vec3, shape=len(self.init_positions))
            init_positions.from_numpy(self.init_positions)
        return HybridData(
            owner=owner,
            idx=self.entity_idx,
            entity_idx=self.entity_idx,
            rigid_entity_idx=self.entity.rigid_entity_idx,
            soft_entity_idx=self.entity.soft_entity_idx,
            info=info,
            init_positions=init_positions,
        )
