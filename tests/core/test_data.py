import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_read_only_tensor():
    values = torch.arange(6, dtype=gs.tc_float, device=gs.device).reshape(2, 3)
    tensor = gs.data.ReadOnlyTensor(values)
    views = (
        tensor,
        tensor[:, 1:],
        tensor.view(-1),
        tensor.detach(),
        tensor.data,
        tensor.to(values.device),
        *tensor.unbind(),
    )
    for view in views:
        with pytest.raises(gs.GenesisException, match="read-only"):
            view.fill_(0)
        with pytest.raises(gs.GenesisException, match="read-only"):
            torch.add(view, 1, out=view)
    with pytest.raises(gs.GenesisException, match="read-only"):
        torch._foreach_add_([tensor], 1)
    with pytest.raises(gs.GenesisException, match="read-only"):
        tensor[0] = 0
    with pytest.raises(gs.GenesisException, match="read-only"):
        tensor.data = torch.zeros_like(values)
    with pytest.raises(gs.GenesisException, match="Clone"):
        tensor.as_subclass(torch.Tensor)
    with pytest.raises(gs.GenesisException, match="clone"):
        tensor.data_ptr()
    with pytest.raises(gs.GenesisException, match="clone"):
        tensor.untyped_storage()
    with pytest.raises(gs.GenesisException, match="clone"):
        torch.from_dlpack(tensor)

    assert_equal(values, [[0, 1, 2], [3, 4, 5]])
    values.add_(1)
    assert_equal(tensor, values)
    assert_equal(views[1], values[:, 1:])
    clone = tensor.clone()
    clone.zero_()
    assert_equal(tensor, [[1, 2, 3], [4, 5, 6]])
    assert_equal(tensor + 1, values + 1)
    destination = torch.empty_like(values)
    destination.copy_(tensor)
    assert_equal(destination, values)
    torch.add(tensor, 1, out=destination)
    assert_equal(destination, values + 1)
    snapshot = tensor.cpu().numpy()
    snapshot.fill(0)
    assert_equal(tensor, [[1, 2, 3], [4, 5, 6]])
    assert_equal(np.asarray(tensor.cpu()), [[1, 2, 3], [4, 5, 6]])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_data_reference(n_envs, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(13.0, -14.0, 11.0),
            camera_lookat=(4.0, 2.0, 2.0),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
            ),
        ),
    )
    fixed = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(4.0, 0.0, 0.0),
            size=(0.2, 0.2, 0.2),
            fixed=True,
        ),
    )
    visual = scene.add_entity(
        options=gs.options.KinematicEntityOptions(
            morph=gs.morphs.Box(
                pos=(8.0, 0.0, 0.0),
                size=(0.2, 0.2, 0.2),
            ),
        ),
    )
    with pytest.raises(gs.GenesisException, match="built"):
        scene.data
    scene.build(n_envs=n_envs)
    reference = entity.base_link.data.pos
    geom = entity.geoms[0]
    joint = entity.joints[0]
    assert geom.data in scene.data
    assert entity.base_link.data in entity.data
    assert joint.data.child is entity.base_link
    assert joint.data.parent is None
    assert len(scene.data) == len({record.uid for record in scene.data})
    assert_equal(joint.data.qpos.read(), entity.get_qpos())
    assert_equal(joint.data.dofs_vel.read(), entity.get_dofs_velocity())
    assert_equal(geom.data.vertices.read(), geom.init_verts)
    assert_equal(geom.data.triangles.read() - geom.data.vertex_start, geom.init_faces)
    snapshot = reference.read()
    view = reference.read(copy=False)
    entity.set_pos((1.0, 2.0, 3.0))
    assert_equal(reference.read(), [[[1.0, 2.0, 3.0]]])
    assert_equal(view, [[[1.0, 2.0, 3.0]]])
    assert_equal(snapshot, 0.0)
    assert_equal(geom.data.pos.read()[..., 0, :], geom.get_pos(relative=False))
    assert_allclose(
        geom.get_verts(),
        gu.transform_by_trans_quat(geom.data.vertices.read(), geom.data.pos.read(), geom.data.quat.read()),
        tol=gs.EPS,
    )
    assert_allclose(
        fixed.geoms[0].get_verts(),
        fixed.geoms[0].data.vertices.read() + fixed.geoms[0].get_pos(relative=False)[..., None, :],
        tol=gs.EPS,
    )
    assert_allclose(geom.data.world_vertices.read(), geom.get_verts(), tol=gs.EPS)
    assert_allclose(fixed.geoms[0].data.world_vertices.read(), fixed.geoms[0].get_verts(), tol=gs.EPS)
    visual.set_pos((9.0, 0.0, 0.0), skip_forward=True)
    assert_equal(visual.vgeoms[0].data.pos.read(), [[[9.0, 0.0, 0.0]]])
    entity.set_pos((2.0, 3.0, 4.0), skip_forward=True)
    assert_equal(geom.data.pos.read(), [[[2.0, 3.0, 4.0]]])
    with pytest.raises(gs.GenesisException, match="read-only"):
        view.zero_()
    scene.step()
    assert_equal(reference.read()[..., 0, :], entity.base_link.get_pos(relative=False))
    checkpoint_pos = reference.read()
    checkpoint = scene.__getstate__()
    entity.set_pos((5.0, 6.0, 7.0), envs_idx=1 if n_envs else None)
    assert_equal(reference.read()[..., 0, :], entity.base_link.get_pos(relative=False))
    scene.__setstate__(checkpoint)
    assert_equal(reference.read(), checkpoint_pos)
    if n_envs:
        entity.set_pos((5.0, 6.0, 7.0), envs_idx=1)
        scene.reset(envs_idx=[0])
        assert_equal(reference.read(), [[[0.0, 0.0, 0.0]], [[5.0, 6.0, 7.0]]])
    scene.reset()
    assert_equal(reference.read(), snapshot)
    scene.destroy()
    with pytest.raises(gs.GenesisException, match="built"):
        reference.read()
    with pytest.raises(gs.GenesisException, match="destroyed"):
        view.clone()
