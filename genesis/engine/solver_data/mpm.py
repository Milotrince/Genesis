"""Particle history and coupling grid for the material point method (MPM)."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs

from . import SolverData, SolverDescription


@dataclass(frozen=True, kw_only=True, eq=False)
class MPMData(SolverData):
    particles: qd.Field
    particles_ng: qd.Field
    particles_info: qd.Field
    grid: qd.Field
    particle_volume_scale: float


@dataclass(frozen=True, kw_only=True, eq=False)
class MPMGeomData(SolverData):
    data: MPMData
    particle_start: int
    particle_end: int


@dataclass(frozen=True, kw_only=True)
class MPMDescription(SolverDescription[MPMData]):
    n_particles: int
    n_envs: int
    substeps_local: int
    grid_res: tuple[int, int, int]
    particle_volume_scale: float

    def allocate(self, owner) -> MPMData:
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            C=gs.qd_mat3,
            F=gs.qd_mat3,
            F_tmp=gs.qd_mat3,
            U=gs.qd_mat3,
            V=gs.qd_mat3,
            S=gs.qd_mat3,
            actu=gs.qd_float,
            Jp=gs.qd_float,
        )
        struct_particle_state_ng = qd.types.struct(active=gs.qd_bool)
        struct_particle_info = qd.types.struct(
            material_idx=gs.qd_int,
            mass=gs.qd_float,
            default_Jp=gs.qd_float,
            free=gs.qd_bool,
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )
        particles = struct_particle_state.field(
            shape=(self.substeps_local + 1, self.n_particles, self.n_envs), needs_grad=True, layout=qd.Layout.SOA
        )
        particles_ng = struct_particle_state_ng.field(
            shape=(self.substeps_local + 1, self.n_particles, self.n_envs), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info = struct_particle_info.field(shape=self.n_particles, needs_grad=False, layout=qd.Layout.SOA)
        grid_cell_state = qd.types.struct(mass=gs.qd_float, vel_in=gs.qd_vec3, vel_out=gs.qd_vec3)
        # Particle updates write frame f + 1, while grid operations only use frame f
        grid = grid_cell_state.field(
            shape=(self.substeps_local, *self.grid_res, self.n_envs), needs_grad=True, layout=qd.Layout.SOA
        )
        return MPMData(
            owner=owner,
            particles=particles,
            particles_ng=particles_ng,
            particles_info=particles_info,
            grid=grid,
            particle_volume_scale=self.particle_volume_scale,
        )
