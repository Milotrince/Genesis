import pytest

import genesis as gs
from genesis.engine.solver_data.articulated import JointsData, LinksData, VisualGeomData
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_articulated_binding(n_envs, show_viewer, tol):
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

    scene.reset()
    for entity, pos in zip(entities, ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))):
        assert_allclose(entity.get_pos(), pos, tol=tol)
