import numpy as np
import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_equal


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
def test_data_reference(n_envs):
    scene = gs.Scene(
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
        ),
    )
    scene.build(n_envs=n_envs)
    reference = gs.data.DataReference(
        entity.solver, "links.pos", entity.solver.dyn_state.links.pos, (slice(None), slice(0, 1))
    )
    snapshot = reference.read()
    view = reference.read(copy=False)
    entity.set_pos((1.0, 2.0, 3.0))
    assert_equal(reference.read(), [[[1.0, 2.0, 3.0]]])
    assert_equal(view, [[[1.0, 2.0, 3.0]]])
    assert_equal(snapshot, 0.0)
    with pytest.raises(gs.GenesisException, match="read-only"):
        view.zero_()
    scene.reset()
    assert_equal(reference.read(), snapshot)
    scene.destroy()
    with pytest.raises(gs.GenesisException, match="built"):
        reference.read()
    with pytest.raises(gs.GenesisException, match="destroyed"):
        view.clone()
