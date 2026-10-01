import math

import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.scene import SCENE_FORMAT

from ..utils.assertions import assert_allclose


@pytest.mark.required
def test_jets(tmp_path, show_viewer, tol):
    res = 16
    orbit_tau = 0.2
    orbit_radius = 0.3
    orbit_radius_vel = 0.0

    jet_radius = 0.1

    sub_orbit_radius = 0.03
    sub_orbit_tau = 3.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        sf_options=gs.options.SFOptions(
            res=res,
            solver_iters=200,
            decay=0.025,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 2.0),
            camera_lookat=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )

    @qd.data_oriented
    class Jet:
        def __init__(
            self,
            world_center,
            jet_radius,
            orbit_radius,
            orbit_radius_vel,
            orbit_init_degree,
            orbit_tau,
            sub_orbit_radius,
            sub_orbit_tau,
        ):
            self.world_center = qd.Vector(world_center)
            self.orbit_radius = orbit_radius
            self.orbit_radius_vel = orbit_radius_vel
            self.orbit_init_radian = math.radians(orbit_init_degree)
            self.orbit_tau = orbit_tau

            self.jet_radius = jet_radius

            self.num_sub_jets = 3
            self.sub_orbit_radian_delta = 2.0 * math.pi / self.num_sub_jets
            self.sub_orbit_radius = sub_orbit_radius
            self.sub_orbit_tau = sub_orbit_tau

        @qd.func
        def get_pos(self, t: float):
            rel_pos = qd.Vector([self.orbit_radius + t * self.orbit_radius_vel, 0.0, 0.0])
            rot_mat = qd.math.rot_by_axis(qd.Vector([0.0, 1.0, 0.0]), self.orbit_init_radian + t * self.orbit_tau)[
                :3, :3
            ]
            rel_pos = rot_mat @ rel_pos
            return rel_pos

        @qd.func
        def get_factor(self, i: int, j: int, k: int, dx: float, t: float):
            rel_pos = self.get_pos(t)
            tan_dir = self.get_tan_dir(t)
            ijk = qd.Vector([i, j, k], dt=gs.qd_float) * dx
            dist = 2 * self.jet_radius
            for q in qd.static(range(self.num_sub_jets)):
                jet_pos = qd.Vector([0.0, self.sub_orbit_radius, 0.0])
                rot_mat = qd.math.rot_by_axis(tan_dir, self.sub_orbit_radian_delta * q + self.sub_orbit_tau * t)[:3, :3]
                jet_pos = (rot_mat @ jet_pos) + self.world_center + rel_pos
                dist_q = (ijk - jet_pos).norm(gs.EPS)
                if dist_q < dist:
                    dist = dist_q
            factor = 0.0
            if dist < self.jet_radius:
                factor = 1.0
            return factor

        @qd.func
        def get_inward_dir(self, t: float):
            neg_pos = -self.get_pos(t)
            return neg_pos.normalized(gs.EPS)

        @qd.func
        def get_tan_dir(self, t: float):
            inward_dir = self.get_inward_dir(t)
            tan_rot_mat = qd.math.rot_by_axis(qd.Vector([0.0, 1.0, 0.0]), 0.0)[:3, :3]
            return tan_rot_mat @ inward_dir

    jet = [
        Jet(
            world_center=[0.5, 0.5, 0.5],
            orbit_radius=orbit_radius,
            orbit_radius_vel=orbit_radius_vel,
            orbit_init_degree=orbit_init_degree,
            orbit_tau=orbit_tau,
            sub_orbit_radius=sub_orbit_radius,
            jet_radius=jet_radius,
            sub_orbit_tau=sub_orbit_tau,
        )
        for orbit_init_degree in np.linspace(0, 360, 3, endpoint=False)
    ]
    scene.sim.sf_solver.set_jets(jet)
    with pytest.raises(gs.GenesisException, match="3 Jets cannot be exported"):
        scene.export(tmp_path / f"jets{SCENE_FORMAT}")
    scene.build()
    with pytest.raises(gs.GenesisException, match="already built"):
        scene.sim.sf_solver.set_jets(jet[:1])
    scene.step()

    density = scene.sim.sf_solver.get_grid_density()
    expected_peak = 1.0 - scene.options.sf.decay * scene.sim.sf_solver.substep_dt
    orbit_angles = np.linspace(0.0, 2.0 * math.pi, len(jet), endpoint=False)
    radial_dirs = np.stack((np.cos(orbit_angles), np.zeros(len(jet)), np.sin(orbit_angles)), axis=-1)
    sub_angles = np.arange(jet[0].num_sub_jets) * jet[0].sub_orbit_radian_delta
    sub_offset = np.array((0.0, sub_orbit_radius, 0.0))
    centers = (
        np.array((0.5, 0.5, 0.5))
        + orbit_radius * radial_dirs[:, None, :]
        + np.cos(sub_angles)[None, :, None] * sub_offset
        - np.sin(sub_angles)[None, :, None] * np.cross(-radial_dirs, sub_offset)[:, None, :]
    )
    coords = np.arange(res) / res
    cells = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    has_source = (np.linalg.norm(cells[..., None, None, :] - centers, axis=-1) < jet_radius).any(axis=-1)
    assert_allclose(density, expected_peak * has_source, tol=tol)
    assert (density >= 0).all()
    velocity = scene.sim.sf_solver.get_grid_velocity()
    assert velocity.isfinite().all()
    assert (velocity.norm(dim=-1) > 0).any()
    density.zero_()
    assert_allclose(scene.sim.sf_solver.get_grid_density().amax(dim=(0, 1, 2)), expected_peak, tol=tol)
    scene.reset()
    assert_allclose(scene.sim.sf_solver.get_grid_density(), 0.0, tol=tol)
    assert_allclose(scene.sim.sf_solver.get_grid_velocity(), 0.0, tol=tol)
