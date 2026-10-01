from dataclasses import replace

import pytest

import genesis as gs
from genesis.engine.scene import SCENE_FORMAT
from genesis.engine.solver_data.articulated import JointsData, LinksData, RigidGeomData, VisualGeomData
from genesis.engine.solver_data.fem import FEMGeomData
from genesis.engine.solver_data.gravity import GravityData
from genesis.engine.solver_data.mpm import MPMGeomData
from genesis.engine.solver_data.pbd import PBDGeomData
from genesis.engine.solver_data.sph import SPHGeomData
from genesis.engine.solvers.base_solver import GravityMixin
from genesis.recorders.trajectory import TRAJECTORY_FORMAT, Trajectory
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_shared_binding(n_envs, tmp_path, show_viewer, tol):
    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.05,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.05,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.05,
        ),
        show_viewer=show_viewer,
    )
    entities = []
    for material, pos in ((gs.materials.Rigid(), (0.0, 0.0, 1.0)), (gs.materials.Kinematic(), (1.0, 0.0, 1.0))):
        entities.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    size=(0.2, 0.2, 0.2),
                ),
                material=material,
            )
        )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.25, 0.25, 0.75),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.5, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.75, 0.75, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    for i, material in enumerate(
        (
            gs.materials.PBD.Elastic(),
            gs.materials.PBD.Liquid(sampler="regular"),
            gs.materials.PBD.Particle(sampler="regular"),
        )
    ):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.25 + 0.25 * i, 0.75, 0.75),
                size=(0.1, 0.1, 0.1),
            ),
            material=material,
        )
    exported = tmp_path / f"shared_data{SCENE_FORMAT}"
    scene.export(exported)
    scene = gs.Scene.load(exported, show_viewer=show_viewer)
    entities = scene.entities[:2]
    fem_entity = scene.entities[2]
    scene.build(n_envs=n_envs)

    for data in scene.sim._solver_data.select(GravityData):
        if not isinstance(data.owner, GravityMixin) or not data.owner.is_active:
            continue
        data.owner.set_gravity((0.0, 0.0, -3.0))
        data.owner.set_gravity((0.0, 0.0, -1.0), envs_idx=0 if n_envs else None)
        data.owner.get_gravity().zero_()
        assert_allclose(data.owner.get_gravity(envs_idx=0), (0.0, 0.0, -1.0), tol=tol)
        assert_allclose(qd_to_torch(data.gravity, transpose=True)[0], (0.0, 0.0, -1.0), tol=tol)
        if n_envs:
            assert_allclose(data.owner.get_gravity(envs_idx=1), (0.0, 0.0, -3.0), tol=tol)

    for entity in entities:
        links = scene.sim._solver_data.get(LinksData, entity.solver, entity.idx)
        joints = scene.sim._solver_data.get(JointsData, entity.solver, entity.idx)
        entity.set_pos((2.0, 3.0, 4.0))
        assert_allclose(entity.get_pos(), (2.0, 3.0, 4.0), tol=tol)
        assert_allclose(
            qd_to_torch(links.state.pos, transpose=True)[:, links.link_start : links.link_end],
            (2.0, 3.0, 4.0),
            tol=tol,
        )
        assert_allclose(
            qd_to_torch(joints.dofs_state.pos, transpose=True)[:, joints.dof_start : joints.dof_start + 3],
            (2.0, 3.0, 4.0),
            tol=tol,
        )
        assert_allclose(
            qd_to_torch(joints.qpos, transpose=True)[:, joints.q_start : joints.q_start + 3],
            (2.0, 3.0, 4.0),
            tol=tol,
        )
        binding = scene.sim._solver_data.bind(joints, write="qpos")
        configuration = entity.get_qpos()
        configuration[..., :3] = 5.0
        binding.commit(configuration)
        assert_allclose(entity.get_pos(), (5.0, 5.0, 5.0), tol=tol)
        configuration[..., :3] = gs.tensor((2.0, 3.0, 4.0))
        binding.commit(configuration)
        with pytest.raises(gs.GenesisException, match="read-only"):
            scene.sim._solver_data.bind(joints).commit(entity.get_qpos())
        with pytest.raises(gs.GenesisException, match="does not support"):
            scene.sim._solver_data.bind(joints, write="mass")
        with pytest.raises(gs.GenesisException, match="different or destroyed"):
            scene.sim._solver_data.bind(replace(joints))
        with pytest.raises(gs.GenesisException, match="Unsupported coupling writes"):
            scene.sim._solver_data.bind_coupling(entity.solver._solver_data, writes=("qpos",))
        qpos = entity.get_qpos()
        qpos.zero_()
        assert_allclose(entity.get_pos(), (2.0, 3.0, 4.0), tol=tol)
        assert_allclose(entity.get_vverts().mean(dim=-2), (2.0, 3.0, 4.0), tol=tol)
        geom = scene.sim._solver_data.get(VisualGeomData, entity.solver)
        assert_allclose(qd_to_torch(geom.state.pos, transpose=True)[:, geom.idx], (2.0, 3.0, 4.0), tol=tol)

    geom = scene.sim._solver_data.get(RigidGeomData, entities[0].solver)
    assert_allclose(entities[0].geoms[0].get_pos(), (2.0, 3.0, 4.0), tol=tol)
    assert_allclose(qd_to_torch(geom.state.pos, transpose=True)[:, geom.idx], (2.0, 3.0, 4.0), tol=tol)
    verts = entities[0].geoms[0].get_verts()
    assert_allclose(
        qd_to_torch(geom.verts_state.pos, transpose=True)[:, geom.verts_state_start : geom.verts_state_end],
        verts,
        tol=tol,
    )

    scene.reset()
    for entity, pos in zip(entities, ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))):
        assert_allclose(entity.get_pos(), pos, tol=tol)

    fem_geom = scene.sim._solver_data.get(FEMGeomData, fem_entity.solver, fem_entity.idx)
    state = fem_entity.get_state()
    assert_allclose((state.pos.amin(dim=-2) + state.pos.amax(dim=-2)) / 2, (0.25, 0.25, 0.75), tol=tol)
    assert_allclose(
        qd_to_torch(fem_geom.data.elements_v.pos, col_mask=0, keepdim=False, transpose=True)[
            :, fem_geom.vert_start : fem_geom.vert_end
        ],
        state.pos,
        tol=tol,
    )

    mpm_entity = scene.entities[3]
    mpm_geom = scene.sim._solver_data.get(MPMGeomData, mpm_entity.solver, mpm_entity.idx)
    assert_allclose(mpm_entity.get_particles_pos().mean(dim=-2), (0.5, 0.5, 0.5), tol=tol)
    assert_allclose(
        qd_to_torch(mpm_geom.data.particles.pos, col_mask=0, keepdim=False, transpose=True)[
            :, mpm_geom.particle_start : mpm_geom.particle_end
        ],
        mpm_entity.get_particles_pos(),
        tol=tol,
    )

    sph_entity = scene.entities[4]
    sph_geom = scene.sim._solver_data.get(SPHGeomData, sph_entity.solver, sph_entity.idx)
    assert_allclose(sph_entity.get_particles_pos().mean(dim=-2), (0.75, 0.75, 0.5), tol=tol)
    assert_allclose(
        qd_to_torch(sph_geom.data.particles.pos, transpose=True)[:, sph_geom.particle_start : sph_geom.particle_end],
        sph_entity.get_particles_pos(),
        tol=tol,
    )

    for i, entity in enumerate(scene.entities[5:]):
        geom = scene.sim._solver_data.get(PBDGeomData, entity.solver, entity.idx)
        pos = entity.get_particles_pos()
        assert_allclose((pos.amin(dim=-2) + pos.amax(dim=-2)) / 2, (0.25 + 0.25 * i, 0.75, 0.75), tol=tol)
        entity.set_particles_pos((0.5, 0.5, 0.5))
        assert_allclose(
            qd_to_torch(geom.data.particles.pos, transpose=True)[:, geom.particle_start : geom.particle_end],
            (0.5, 0.5, 0.5),
            tol=tol,
        )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_plugin_lifecycle(n_envs, solver_plugin, tmp_path, show_viewer, tol):
    primary, secondary = solver_plugin.primary, solver_plugin.secondary
    with pytest.raises(TypeError, match="concrete SolverOptions subclass"):

        class InvalidSolver(primary.Solver, options_cls=gs.options.RigidOptions):
            pass

    with pytest.raises(ValueError, match="already registered"):

        class DuplicateSolver(primary.Solver, options_cls=primary.Options):
            pass

    with pytest.raises(gs.GenesisException, match="overlaps"):
        gs.Scene(additional_solver_options=(primary.OverlappingOptions(entity_name="first", speed=1.0),))
    with pytest.raises(gs.GenesisException, match="Import a solver registered"):
        gs.Scene(additional_solver_options=(primary.UnregisteredOptions(entity_name="first", speed=1.0),))
    with pytest.raises(gs.GenesisException, match="Duplicate additional solver"):
        gs.Scene(additional_solver_options=(primary.Options(entity_name="first", speed=1.0),) * 2)
    with pytest.raises(gs.GenesisException, match="does not support differentiable"):
        gs.Scene(
            sim_options=gs.options.SimOptions(
                requires_grad=True,
            ),
            additional_solver_options=(primary.Options(entity_name="first", speed=1.0),),
        )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        additional_solver_options=(
            primary.Options(entity_name="first", speed=1.0),
            secondary.Options(entity_name="second", speed=2.0),
        ),
    )
    for name, y in (("first", -0.2), ("second", 0.2)):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, y, 0.0),
                size=(0.1, 0.1, 0.1),
            ),
            material=gs.materials.Kinematic(),
            name=name,
        )
    exported = tmp_path / f"plugin{SCENE_FORMAT}"
    scene.export(exported)
    scene = gs.Scene.load(exported, show_viewer=show_viewer)
    path = tmp_path / f"plugin{TRAJECTORY_FORMAT}"
    scene.start_recording(gs.recorders.TrajectoryFile(filename=str(path), exact=True))
    scene.build(n_envs=n_envs)
    scene.step()
    scene.step()
    scene.stop_recording()
    checkpoint = scene.__getstate__()
    scene.step()
    scene.__setstate__(checkpoint)
    for options_cls, name, speed in ((primary.Options, "first", 1.0), (secondary.Options, "second", 2.0)):
        entity = scene.get_entity(name)
        plugin = scene.get_solver(options_cls)
        assert_allclose(entity.get_pos()[..., 0], speed * scene.dt * 2, tol=tol)
        assert_allclose(plugin.get_positions()[:, 0], entity.get_pos(), tol=tol)
        plugin.get_positions().zero_()
        assert_allclose(plugin.get_positions()[:, 0], entity.get_pos(), tol=tol)
    trajectory = Trajectory(path, scene=scene)
    trajectory.seek(1)
    for options_cls, name, speed in ((primary.Options, "first", 1.0), (secondary.Options, "second", 2.0)):
        entity = scene.get_entity(name)
        assert_allclose(entity.get_pos()[..., 0], speed * scene.dt, tol=tol)
        assert_allclose(scene.get_solver(options_cls).get_positions()[:, 0], entity.get_pos(), tol=tol)
    scene.reset(envs_idx=0 if n_envs else None)
    for options_cls, speed in ((primary.Options, 1.0), (secondary.Options, 2.0)):
        assert_allclose(
            scene.get_solver(options_cls).get_positions()[:, 0, 0],
            [0.0, speed * scene.dt] if n_envs else [0.0],
            tol=tol,
        )
    scene.reset()
    for options_cls in (primary.Options, secondary.Options):
        assert_allclose(scene.get_solver(options_cls).get_positions()[..., 0], 0.0, tol=tol)
