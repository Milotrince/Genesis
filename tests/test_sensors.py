import torch

import genesis as gs
from genesis.options.sensors import IMUOptions
from utils import assert_allclose


def test_imu_sensor(show_viewer):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.1),
        ),
    )

    imu = scene.add_sensor(IMUOptions(link_idx=box.base_link_idx))

    scene.build()

    for _ in range(100):
        scene.step()

    acc, ang = imu.read()
    assert_allclose(acc, torch.tensor([0.0, 0.0, -GRAVITY]), tol=1e-7)
    assert_allclose(ang, torch.tensor([0.0, 0.0, 0.0]), tol=1e-7)
