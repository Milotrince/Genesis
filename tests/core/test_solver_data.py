import pytest

import genesis as gs
from genesis.engine.scene import SCENE_FORMAT
from genesis.engine.solver_data.articulated import JointsData, LinksData, RigidGeomData, VisualGeomData
from genesis.engine.solver_data.fem import FEMGeomData
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_shared_binding(n_envs, tmp_path, show_viewer, tol):
    scene = gs.Scene(
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
            pos=(3.0, 0.0, 1.0),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    exported = tmp_path / f"shared_data{SCENE_FORMAT}"
    scene.export(exported)
    scene = gs.Scene.load(exported, show_viewer=show_viewer)
    entities = scene.entities[:2]
    fem_entity = scene.entities[2]
    scene.build(n_envs=n_envs)

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
    assert_allclose((state.pos.amin(dim=-2) + state.pos.amax(dim=-2)) / 2, (3.0, 0.0, 1.0), tol=tol)
    assert_allclose(
        qd_to_torch(fem_geom.data.elements_v.pos, col_mask=0, keepdim=False, transpose=True)[
            :, fem_geom.vert_start : fem_geom.vert_end
        ],
        state.pos,
        tol=tol,
    )
