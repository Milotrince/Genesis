"""Particle state and rest topology for position-based dynamics (PBD)."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs

from . import SolverData, SolverDescription
from .gravity import GravityData, GravityDescription


@dataclass(frozen=True, kw_only=True, eq=False)
class PBDData(GravityData):
    particles_info: qd.Field
    particles_info_reordered: qd.Field
    particles: qd.Field
    particles_reordered: qd.Field
    particles_ng: qd.Field
    particles_ng_reordered: qd.Field
    edges_info: qd.Field
    inner_edges_info: qd.Field
    elems_info: qd.Field


@dataclass(frozen=True, kw_only=True, eq=False)
class PBDGeomData(SolverData):
    data: PBDData
    particle_start: int
    particle_end: int
    edge_start: int
    edge_end: int
    inner_edge_start: int
    inner_edge_end: int
    elem_start: int
    elem_end: int


@dataclass(frozen=True, kw_only=True)
class PBDDescription(GravityDescription, SolverDescription[PBDData]):
    n_particles: int
    n_edges: int
    n_inner_edges: int
    n_elems: int

    def allocate(self, owner) -> PBDData:
        struct_particle_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            rho_rest=gs.qd_float,
            material_type=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
            density_relaxation=gs.qd_float,
            viscosity_relaxation=gs.qd_float,
        )
        struct_particle_state = qd.types.struct(
            free=gs.qd_bool,
            pos=gs.qd_vec3,
            ipos=gs.qd_vec3,
            dpos=gs.qd_vec3,
            vel=gs.qd_vec3,
            lam=gs.qd_float,
            rho=gs.qd_float,
        )

        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )

        particles_info = struct_particle_info.field(shape=(self.n_particles,), layout=qd.Layout.SOA)
        particles_info_reordered = struct_particle_info.field(
            shape=(self.n_particles, self.n_envs), layout=qd.Layout.SOA
        )
        particles = struct_particle_state.field(shape=(self.n_particles, self.n_envs), layout=qd.Layout.SOA)
        particles_reordered = struct_particle_state.field(shape=(self.n_particles, self.n_envs), layout=qd.Layout.SOA)
        particles_ng = struct_particle_state_ng.field(shape=(self.n_particles, self.n_envs), layout=qd.Layout.SOA)
        particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self.n_particles, self.n_envs), layout=qd.Layout.SOA
        )
        struct_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            stretch_compliance=gs.qd_float,
            stretch_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
        )
        edges_info = struct_edge_info.field(shape=(max(1, self.n_edges),), layout=qd.Layout.SOA)

        struct_inner_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            bending_compliance=gs.qd_float,
            bending_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        inner_edges_info = struct_inner_edge_info.field(shape=(max(self.n_inner_edges, 1),), layout=qd.Layout.SOA)

        struct_elem_info = qd.types.struct(
            vol_rest=gs.qd_float,
            volume_compliance=gs.qd_float,
            volume_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        elems_info = struct_elem_info.field(shape=(max(self.n_elems, 1),), layout=qd.Layout.SOA)
        return PBDData(
            owner=owner,
            gravity=self.allocate_gravity(),
            particles_info=particles_info,
            particles_info_reordered=particles_info_reordered,
            particles=particles,
            particles_reordered=particles_reordered,
            particles_ng=particles_ng,
            particles_ng_reordered=particles_ng_reordered,
            edges_info=edges_info,
            inner_edges_info=inner_edges_info,
            elems_info=elems_info,
        )
