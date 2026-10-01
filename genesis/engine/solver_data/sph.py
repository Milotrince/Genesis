"""Particle state and reorder maps for smoothed particle hydrodynamics (SPH)."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs

from . import SolverData, SolverDescription


@dataclass(frozen=True, kw_only=True, eq=False)
class SPHData(SolverData):
    particles: qd.Field
    particles_ng: qd.Field
    particles_info: qd.Field
    particles_reordered: qd.Field
    particles_ng_reordered: qd.Field
    particles_info_reordered: qd.Field


@dataclass(frozen=True, kw_only=True, eq=False)
class SPHGeomData(SolverData):
    data: SPHData
    particle_start: int
    particle_end: int


@dataclass(frozen=True, kw_only=True)
class SPHDescription(SolverDescription[SPHData]):
    n_particles: int
    n_envs: int

    def allocate(self, owner) -> SPHData:
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            acc=gs.qd_vec3,
            rho=gs.qd_float,
            p=gs.qd_float,
            dfsph_factor=gs.qd_float,
            drho=gs.qd_float,
        )

        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )

        struct_particle_info = qd.types.struct(
            rho=gs.qd_float,
            mass=gs.qd_float,
            stiffness=gs.qd_float,
            exponent=gs.qd_float,
            mu=gs.qd_float,
            gamma=gs.qd_float,
        )

        particles = struct_particle_state.field(
            shape=(self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_ng = struct_particle_state_ng.field(
            shape=(self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info = struct_particle_info.field(shape=(self.n_particles,), needs_grad=False, layout=qd.Layout.SOA)
        particles_reordered = struct_particle_state.field(
            shape=(self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info_reordered = struct_particle_info.field(
            shape=(self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )

        return SPHData(
            owner=owner,
            particles=particles,
            particles_ng=particles_ng,
            particles_info=particles_info,
            particles_reordered=particles_reordered,
            particles_ng_reordered=particles_ng_reordered,
            particles_info_reordered=particles_info_reordered,
        )
