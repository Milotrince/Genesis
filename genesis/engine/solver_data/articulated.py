"""Articulated geometry and state backed by the native bulk array layout."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from . import SolverData, SolverDescription
from .gravity import GravityData


@dataclass(frozen=True, kw_only=True, eq=False)
class ArticulatedData(GravityData):
    info: array_class.DynInfo
    state: array_class.DynState
    qpos: qd.Tensor
    qpos0: qd.Tensor
    meaninertia: qd.Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class LinksData(SolverData):
    info: array_class.LinksInfo
    state: array_class.LinksState
    link_start: int
    link_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class JointsData(SolverData):
    info: array_class.JointsInfo
    state: array_class.JointsState
    dofs_info: array_class.DofsInfo
    dofs_state: array_class.DofsState
    qpos: qd.Tensor
    qpos0: qd.Tensor
    joint_start: int
    joint_end: int
    dof_start: int
    dof_end: int
    q_start: int
    q_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class GeomData(SolverData):
    link_idx: int
    vert_start: int
    vert_end: int
    face_start: int
    face_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class VisualGeomData(GeomData):
    info: array_class.VGeomsInfo
    state: array_class.VGeomsState
    verts_info: array_class.VVertsInfo
    verts_state: array_class.VVertsState
    faces_info: array_class.VFacesInfo


@dataclass(frozen=True, kw_only=True)
class ArticulatedDescription(SolverDescription[ArticulatedData]):
    """Collision geometry and equality counts default to single-entry placeholders for kinematic-only solvers."""

    n_envs: int
    n_dofs_: int
    n_qs_: int
    n_links_: int
    n_joints_: int
    n_entities_: int
    n_vverts_: int
    n_custom_vverts_: int
    n_vfaces_: int
    n_vgeoms_: int
    n_joints_per_link: int
    n_geoms_: int = 1
    n_verts_: int = 1
    n_faces_: int = 1
    n_edges_: int = 1
    n_free_verts_: int = 1
    n_fixed_verts_: int = 1
    n_candidate_equalities_: int = 1
    is_dynamic: bool = False
    has_grad: bool
    is_batch_links_info: bool
    is_batch_dofs_info: bool
    is_batch_joints_info: bool

    def allocate(self, owner) -> ArticulatedData:
        dofs_info = array_class.get_dofs_info(self)
        dofs_state = array_class.get_dofs_state(self)
        links_info = array_class.get_links_info(self)
        links_state = array_class.get_links_state(self)
        joints_info = array_class.get_joints_info(self)
        joints_state = array_class.get_joints_state(self)

        entities_info = array_class.get_entities_info(self)
        entities_state = array_class.get_entities_state(self)

        vverts_info = array_class.get_vverts_info(self)
        vverts_state = array_class.get_vverts_state(self)
        vfaces_info = array_class.get_vfaces_info(self)

        vgeoms_info = array_class.get_vgeoms_info(self)
        vgeoms_state = array_class.get_vgeoms_state(self)

        geoms_info = array_class.get_geoms_info(self, self.is_dynamic)
        geoms_state = array_class.get_geoms_state(self, self.is_dynamic)

        verts_info = array_class.get_verts_info(self, self.is_dynamic)
        faces_info = array_class.get_faces_info(self, self.is_dynamic)
        edges_info = array_class.get_edges_info(self, self.is_dynamic)

        free_verts_state = array_class.get_free_verts_state(self, self.is_dynamic)
        fixed_verts_state = array_class.get_fixed_verts_state(self, self.is_dynamic)

        equalities_info = array_class.get_equalities_info(self, self.is_dynamic)

        info = array_class.DynInfo(
            entities=entities_info,
            links=links_info,
            joints=joints_info,
            dofs=dofs_info,
            geoms=geoms_info,
            verts=verts_info,
            faces=faces_info,
            edges=edges_info,
            vverts=vverts_info,
            vfaces=vfaces_info,
            vgeoms=vgeoms_info,
            equalities=equalities_info,
        )
        state = array_class.DynState(
            entities=entities_state,
            links=links_state,
            joints=joints_state,
            dofs=dofs_state,
            geoms=geoms_state,
            free_verts=free_verts_state,
            fixed_verts=fixed_verts_state,
            vverts=vverts_state,
            vgeoms=vgeoms_state,
        )

        return ArticulatedData(
            owner=owner,
            info=info,
            state=state,
            qpos=array_class.V(
                dtype=gs.qd_float, shape=(self.n_qs_, self.n_envs), needs_grad=self.has_grad and self.is_dynamic
            ),
            qpos0=array_class.V(dtype=gs.qd_float, shape=(self.n_qs_, self.n_envs)),
            gravity=array_class.V_VEC(3, dtype=gs.qd_float, shape=(self.n_envs,) if self.is_dynamic else ()),
            meaninertia=array_class.V(dtype=gs.qd_float, shape=(self.n_envs,) if self.is_dynamic else ()),
        )
