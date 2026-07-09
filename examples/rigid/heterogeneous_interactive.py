"""
Interactive Geometry Pool
=========================

Each parallel environment can show a different object, swapped in at runtime from a dynamic GPU geometry pool
(`add_entity(geom_pool=...)` + `entity.set_active_object(...)`), and resized per environment with per-env geom
scaling (`entity.set_scale(...)`). Unlike a fixed heterogeneous morph list, pool objects are processed and
uploaded on demand, so a large catalog of shapes (primitives, convex-decomposed and nonconvex meshes) shares a
small reserved slot budget.

Usage:
    python heterogeneous_interactive.py           # 9 environments, GPU
    python heterogeneous_interactive.py -n 16     # 16 environments
    python heterogeneous_interactive.py --cpu     # run on CPU

Controls:
    R           - randomize each environment's object (box / sphere / cylinder / duck / bunny / dragon)
    T           - randomize each environment's size
    left-drag   - grab and drag an object (MouseInteraction plugin)
    ESC         - quit
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

SPAWN_POS = (0.0, 0.0, 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_envs", type=int, default=9)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_geom_scaling=True),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, -4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # The catalog of objects the pool can hold. Meshes are normalized to ~0.2 m and kept nonconvex (single
    # geom, colliding via the vertex-vs-SDF narrowphase, rendering their real shape) with decimation to keep
    # the vertex count - hence contact/SDF-scan cost - low. These morph objects persist so their pool residency
    # keys stay stable. (Convex decomposition of a pooled mesh is not used here: its many small pieces yield an
    # ill-conditioned composed inertial - a known pool limitation.)
    objects = [
        gs.morphs.Box(size=(0.18, 0.18, 0.18), pos=SPAWN_POS),
        gs.morphs.Sphere(radius=0.1, pos=SPAWN_POS),
        gs.morphs.Cylinder(radius=0.08, height=0.2, pos=SPAWN_POS),
        gs.morphs.Mesh(file="meshes/duck.obj", scale=0.05, pos=SPAWN_POS, convexify=False, decimate=True),
        gs.morphs.Mesh(file="meshes/bunny.obj", scale=0.2, pos=SPAWN_POS, convexify=False, decimate=True),
        gs.morphs.Mesh(file="meshes/dragon.obj", scale=0.2, pos=SPAWN_POS, convexify=False, decimate=True),
    ]
    # Passing the catalog as the geometry pool auto-sizes it: the per-slot budgets and slot count are derived
    # by processing each object once at build (each object's geometry is cached for a fast set_active_object).
    obj = scene.add_entity(
        gs.morphs.Box(size=(0.15, 0.15, 0.15), pos=SPAWN_POS),
        geom_pool=objects,
    )

    # Drag objects around with the mouse (left-click and drag) to probe the swapped-in collision geometry.
    scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin(color=(0.1, 0.6, 0.8, 0.6)))

    scene.build(n_envs=n_envs, env_spacing=(0.7, 0.7))

    rng = np.random.default_rng(args.seed)
    spawn_pos = np.tile(np.array(SPAWN_POS, dtype=np.float32), (n_envs, 1))

    def apply_randomize_objects():
        """Bind each environment to a random pooled object, grouping envs that pick the same object."""
        choices = rng.integers(0, len(objects), size=n_envs)
        for object_idx in np.unique(choices):
            envs_idx = np.where(choices == object_idx)[0]
            obj.set_active_object(objects[object_idx], envs_idx=envs_idx.tolist())
        obj.set_pos(spawn_pos, zero_velocity=True)
        gs.logger.info(f"Objects: {[type(objects[c]).__name__ for c in choices]}")

    def apply_randomize_size():
        """Give each environment a random per-axis (x, y, z) scale, stretching and squashing objects.

        The geometry pool applies per-axis scale directly (unlike the base set_scale path, it does not
        require radial primitives to stay isotropic), so spheres and cylinders become ellipsoids here.
        """
        sizes = rng.uniform(0.6, 1.6, size=(n_envs, 3)).astype(np.float32)
        obj.set_scale(sizes)
        obj.set_pos(spawn_pos, zero_velocity=True)
        gs.logger.info(f"Sizes:\n{sizes.round(2)}")

    # The viewer runs in its own thread, so keybind callbacks fire there - not on this stepping thread. They
    # must not touch the GPU directly: set_active_object / set_scale launch kernels that would race the
    # scene.step() kernels below, corrupting device state (swapped-in objects silently fail to upload, and
    # the process can crash). So a callback only raises a request flag; the main loop drains it and runs the
    # actual work between steps, keeping every kernel launch on this one thread.
    requested = {"objects": False, "size": False}
    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind(
            "randomize_objects",
            Key.R,
            KeyAction.PRESS,
            callback=lambda: requested.__setitem__("objects", True),
            allow_overload=False,
        ),
        Keybind("randomize_size", Key.T, KeyAction.PRESS, callback=lambda: requested.__setitem__("size", True)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    # Start with a varied scene when run interactively; skip the (mesh-processing) load under pytest so the
    # example test just builds and steps the base entity quickly.
    if "PYTEST_VERSION" not in os.environ:
        apply_randomize_objects()

    print("\nGeometry-pool controls:")
    print("R         - randomize each environment's object")
    print("T         - randomize each environment's size")
    print("left-drag - grab and drag an object")
    print("ESC       - quit\n")

    while is_running and scene.viewer.is_alive():
        if requested["objects"]:
            requested["objects"] = False
            apply_randomize_objects()
        if requested["size"]:
            requested["size"] = False
            apply_randomize_size()
        scene.step()
        if "PYTEST_VERSION" in os.environ:
            break


if __name__ == "__main__":
    main()
