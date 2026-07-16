"""
Per-environment geometry scale
===============================

Each parallel environment simulates the same four objects - a sphere, a box, a duck and a bunny - but at a
different, randomly resampled size, using per-environment runtime geometry scale
(`RigidOptions(enable_geom_scaling=True)` + `entity.set_scale(...)`). This is the backbone for size domain
randomization: one `add_entity` per object, one solver, N differently-sized copies.

Anisotropic scale is supported, so each axis is scaled independently: a sphere becomes an ellipsoid, a box a
cuboid, a mesh a stretched mesh. The objects float in a zero-gravity row so the focus is the size variation
itself; the meshes are convexified so their (support-based) collision would match what is drawn.

Controls: R randomize sizes, ESC quit.

Usage:
    python geom_scale.py            # 9 environments, GPU
    python geom_scale.py -n 16
    python geom_scale.py --cpu
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_envs", type=int, default=9)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            gravity=(0.0, 0.0, 0.0),  # objects hold their spawn pose so the demo is purely about size
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.5, 1.5),
            camera_lookat=(0.75, 0.0, 0.3),
            enable_default_keybinds=False,
        ),
        show_viewer=True,
    )

    # Four objects in a row, each a separate entity resized independently per environment. Meshes are convexified
    # so their scaled collision (support-based narrowphase) matches what is drawn; their real shape still renders.
    objects = [
        scene.add_entity(
            gs.morphs.Sphere(
                radius=0.1,
                pos=(0.0, 0.0, 0.3),
            ),
        ),
        scene.add_entity(
            gs.morphs.Box(
                size=(0.18, 0.18, 0.18),
                pos=(0.5, 0.0, 0.3),
            ),
        ),
        scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/duck.obj",
                scale=0.05,
                pos=(1.0, 0.0, 0.3),
                convexify=True,
            ),
        ),
        scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/bunny.obj",
                scale=0.2,
                pos=(1.5, 0.0, 0.3),
                convexify=True,
            ),
        ),
    ]

    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    rng = np.random.default_rng(args.seed)

    def randomize_size():
        for obj in objects:
            obj.set_scale(rng.uniform(0.5, 1.6, size=(n_envs, 3)))
        gs.logger.info("Randomized per-environment object sizes.")

    # Keybind callbacks fire on the viewer thread, which must not launch GPU kernels concurrently with the main
    # loop's scene.step(); the callback only flips a flag that the loop drains between steps.
    is_running = True
    is_resize_requested = False

    def request_resize():
        nonlocal is_resize_requested
        is_resize_requested = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("randomize_size", Key.R, KeyAction.PRESS, callback=request_resize),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    # Start with a varied scene interactively; skip the resize under pytest so the example test just builds and steps.
    if "PYTEST_VERSION" not in os.environ:
        randomize_size()

    print("\nGeometry-scale controls:")
    print("R   - randomize each environment's object sizes")
    print("ESC - quit\n")

    while is_running and scene.viewer.is_alive():
        if is_resize_requested:
            is_resize_requested = False
            randomize_size()
        scene.step()
        if "PYTEST_VERSION" in os.environ:
            break


if __name__ == "__main__":
    main()
