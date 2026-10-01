"""Resolved meshes and pose history for differentiable tools."""

from dataclasses import dataclass

import numpy as np
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.base_entity import EntityDescription, VerticesDescription
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils.mesh import compute_sdf_data, load_mesh

from . import SolverData, SolverDescription


@dataclass(kw_only=True)
class ToolEntityDescription(VerticesDescription, EntityDescription):
    morph: Morph
    surface: Surface
    name: str | None
    init_pos: np.ndarray
    init_quat: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    sdf_voxels: np.ndarray | None
    T_mesh_to_sdf: np.ndarray | None

    @classmethod
    def resolve(cls, material, morph, surface, name):
        mesh_orig = load_mesh(morph.file)
        scale = np.linalg.norm(mesh_orig.extents, ord=np.inf)
        center = np.mean(mesh_orig.bounds, axis=0)
        mesh = trimesh.Trimesh(
            vertices=(mesh_orig.vertices - center) / scale,
            faces=mesh_orig.faces,
            vertex_normals=mesh_orig.vertex_normals,
            face_normals=mesh_orig.face_normals,
        )
        T_init = gu.scale_to_T(np.array(morph.scale, dtype=gs.np_float))
        sdf_voxels = T_mesh_to_sdf = None
        if material.collision:
            sdf_data = compute_sdf_data(mesh, material.sdf_res)
            sdf_voxels = sdf_data["voxels"].astype(gs.np_float, order="C", copy=False)
            T_mesh_to_sdf = sdf_data["T_mesh_to_sdf"].astype(gs.np_float, order="C", copy=False) @ gu.inv_T(T_init)
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )
        return cls(
            material=material,
            morph=morph,
            surface=surface,
            name=name,
            init_pos=pos,
            init_quat=quat,
            init_positions=gu.transform_by_T(mesh.vertices.astype(gs.np_float, order="C", copy=False), T_init),
            normals=mesh.vertex_normals.astype(gs.np_float, order="C", copy=False),
            faces=mesh.faces.astype(gs.np_int, order="C", copy=False),
            sdf_voxels=sdf_voxels,
            T_mesh_to_sdf=T_mesh_to_sdf,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class ToolGeomData(SolverData):
    pos: qd.Field
    quat: qd.Field
    vel: qd.Field
    ang: qd.Field
    init_vertices: qd.Field
    init_vertex_normals: qd.Field
    faces: qd.Field
    vertices: qd.Field
    vertex_normals: qd.Field
    sdf_voxels: qd.Field | None
    T_mesh_to_sdf: qd.Field | None


@dataclass(frozen=True, kw_only=True, eq=False)
class ToolData(SolverData):
    geoms: tuple[ToolGeomData, ...]


@dataclass(frozen=True, kw_only=True)
class ToolDescription(SolverDescription[ToolData]):
    entities: tuple[ToolEntityDescription, ...]
    entity_idxs: tuple[int, ...]
    n_envs: int
    substeps_local: int

    def allocate(self, owner) -> ToolData:
        geoms = []
        for entity, idx in zip(self.entities, self.entity_idxs, strict=True):
            pos = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            quat = qd.Vector.field(n=4, dtype=gs.qd_float, needs_grad=True)
            vel = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            ang = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            qd.root.dense(qd.ij, (self.substeps_local + 1, self.n_envs)).place(
                pos, pos.grad, quat, quat.grad, vel, vel.grad, ang, ang.grad
            )
            n_verts = len(entity.init_positions)
            init_vertices = qd.Vector.field(n=3, dtype=gs.qd_float, shape=n_verts)
            init_vertex_normals = qd.Vector.field(n=3, dtype=gs.qd_float, shape=n_verts)
            faces = qd.field(dtype=gs.qd_int, shape=entity.faces.size)
            init_vertices.from_numpy(entity.init_positions)
            init_vertex_normals.from_numpy(entity.normals)
            faces.from_numpy(entity.faces.reshape(entity.faces.size))
            vertices = qd.Vector.field(n=3, dtype=gs.qd_float, shape=n_verts)
            vertex_normals = qd.Vector.field(n=3, dtype=gs.qd_float, shape=n_verts)
            sdf_voxels = T_mesh_to_sdf = None
            if entity.sdf_voxels is not None:
                sdf_voxels = qd.field(dtype=gs.qd_float, shape=entity.sdf_voxels.shape)
                T_mesh_to_sdf = qd.Matrix.field(n=4, m=4, dtype=gs.qd_float, shape=())
                sdf_voxels.from_numpy(entity.sdf_voxels)
                T_mesh_to_sdf.from_numpy(entity.T_mesh_to_sdf)
            geoms.append(
                ToolGeomData(
                    owner=owner,
                    idx=idx,
                    entity_idx=idx,
                    pos=pos,
                    quat=quat,
                    vel=vel,
                    ang=ang,
                    init_vertices=init_vertices,
                    init_vertex_normals=init_vertex_normals,
                    faces=faces,
                    vertices=vertices,
                    vertex_normals=vertex_normals,
                    sdf_voxels=sdf_voxels,
                    T_mesh_to_sdf=T_mesh_to_sdf,
                )
            )
        return ToolData(owner=owner, geoms=tuple(geoms))
