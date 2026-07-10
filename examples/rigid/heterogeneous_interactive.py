"""
Interactive heterogeneous environments
======================================

Two ways a single `add_entity` can give each parallel environment a different body, both interactive:

- Default (geometry pool): each environment shows a different object swapped in at runtime from a dynamic GPU
  geometry pool (`add_entity(geom_pool=...)` + `entity.set_active_object(...)`), resized per environment with
  per-env geom scaling (`entity.set_scale(...)`). A large catalog of shapes shares a small reserved slot
  budget.
- `--articulated`: each environment simulates a different articulated robot - a Franka with and without its
  gripper, a KUKA iiwa, and a simple 2-link arm - all from a single `add_entity(morph=[variant, ...])`. The
  variants differ in kinematic topology (link/joint/DOF counts); a smaller robot leaves its extra link/DOF
  slots inert. This is the backbone for cross-embodiment RL and morphology domain randomization.

Usage:
    python heterogeneous_interactive.py                 # geometry pool, 9 environments, GPU
    python heterogeneous_interactive.py --articulated   # ragged articulated topology
    python heterogeneous_interactive.py -n 16
    python heterogeneous_interactive.py --cpu

Controls (pool):        R randomize objects, T randomize size, left-drag grab, ESC quit
Controls (articulated): R randomize robots, T randomize size, Y randomize joint pose, left-drag grab, ESC quit
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

SPAWN_POS = (0.0, 0.0, 0.5)


def run_pool(scene, args, n_envs):
    """Geometry-pool demo: swap and rescale each environment's object at runtime."""
    # The catalog of objects the pool can hold. Meshes are normalized to ~0.2 m and kept nonconvex (single
    # geom, colliding via the vertex-vs-SDF narrowphase, rendering their real shape) with decimation to keep
    # the vertex count - hence contact/SDF-scan cost - low. These morph objects persist so their pool residency
    # keys stay stable. (Convex decomposition of a pooled mesh is not used here: its many small pieces yield an
    # ill-conditioned composed inertial - a known pool limitation.)
    objects = [
        gs.morphs.Box(
            size=(0.18, 0.18, 0.18),
            pos=SPAWN_POS,
        ),
        gs.morphs.Sphere(
            radius=0.1,
            pos=SPAWN_POS,
        ),
        gs.morphs.Cylinder(
            radius=0.08,
            height=0.2,
            pos=SPAWN_POS,
        ),
        gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.05,
            pos=SPAWN_POS,
            convexify=False,
            decimate=True,
        ),
        gs.morphs.Mesh(
            file="meshes/bunny.obj",
            scale=0.2,
            pos=SPAWN_POS,
            convexify=False,
            decimate=True,
        ),
        gs.morphs.Mesh(
            file="meshes/dragon.obj",
            scale=0.2,
            pos=SPAWN_POS,
            convexify=False,
            decimate=True,
        ),
    ]
    # Passing the catalog as the geometry pool auto-sizes it: the per-slot budgets and slot count are derived
    # by processing each object once at build (each object's geometry is cached for a fast set_active_object).
    obj = scene.add_entity(
        gs.morphs.Box(
            size=(0.15, 0.15, 0.15),
            pos=SPAWN_POS,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 1.0, 1.0),
        ),
        geom_pool=objects,
    )

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            color=(0.1, 0.6, 0.8, 0.6),
            use_force=True,
        ),
    )

    scene.build(n_envs=n_envs, env_spacing=(0.7, 0.7))

    rng = np.random.default_rng(args.seed)
    spawn_pos = np.tile(np.array(SPAWN_POS, dtype=np.float32), (n_envs, 1))
    # The object currently bound to each environment, so size randomization can respect its shape.
    current_choices = np.zeros(n_envs, dtype=int)

    def apply_randomize_objects():
        """Bind each environment to a random pooled object, grouping envs that pick the same object."""
        choices = rng.integers(0, len(objects), size=n_envs)
        for object_idx in np.unique(choices):
            envs_idx = np.where(choices == object_idx)[0]
            obj.set_active_object(objects[object_idx], envs_idx=envs_idx.tolist())
        current_choices[:] = choices
        obj.set_pos(spawn_pos, zero_velocity=True)
        gs.logger.info(f"Objects: {[type(objects[c]).__name__ for c in choices]}")

    def apply_randomize_size():
        """Give each environment a random per-axis (x, y, z) scale, stretching and squashing objects.

        Radial primitives collide analytically, so their radial plane must stay circular (a cylinder cannot be
        an ellipse): spheres are kept fully isotropic and cylinders keep sx == sy (free height). Boxes and
        meshes stretch on every axis.
        """
        sizes = rng.uniform(0.6, 1.6, size=(n_envs, 3)).astype(np.float32)
        for env_idx, object_idx in enumerate(current_choices):
            morph = objects[object_idx]
            if isinstance(morph, gs.morphs.Sphere):
                sizes[env_idx] = sizes[env_idx, 0]
            elif isinstance(morph, gs.morphs.Cylinder):
                sizes[env_idx, 1] = sizes[env_idx, 0]
        obj.set_scale(sizes)
        obj.set_pos(spawn_pos, zero_velocity=True)
        gs.logger.info(f"Sizes:\n{sizes.round(2)}")

    # The viewer runs in its own thread, so keybind callbacks fire there - not on this stepping thread. They
    # must not touch the GPU directly: set_active_object / set_scale launch kernels that would race the
    # scene.step() kernels below, corrupting device state (swapped-in objects silently fail to upload, and
    # the process can crash). So a callback only raises a request flag; the main loop drains it and runs the
    # actual work between steps, keeping every kernel launch on this one thread.
    requested = {"objects": False, "size": False}
    is_running = [True]

    scene.viewer.register_keybinds(
        Keybind(
            "randomize_objects",
            Key.R,
            KeyAction.PRESS,
            callback=lambda: requested.__setitem__("objects", True),
            allow_overload=False,
        ),
        Keybind("randomize_size", Key.T, KeyAction.PRESS, callback=lambda: requested.__setitem__("size", True)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=lambda: is_running.__setitem__(0, False)),
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

    while is_running[0] and scene.viewer.is_alive():
        if requested["objects"]:
            requested["objects"] = False
            apply_randomize_objects()
        if requested["size"]:
            requested["size"] = False
            apply_randomize_size()
        scene.step()
        if "PYTEST_VERSION" in os.environ:
            break


def run_articulated(scene, args, n_envs):
    """Ragged-topology demo: each environment is a different robot, swapped/rescaled/posed at runtime."""
    scene.add_entity(gs.morphs.Plane())
    # A mix of genuinely different robots per environment - a Franka with and without its gripper, a KUKA iiwa,
    # and a simple 2-link arm. The first morph defines the skeleton and must be the largest (most links, widest
    # joint per slot): the Franka-with-gripper is a superset of the others, whose missing links/DOFs are the
    # trailing slots they leave inert. batch_fixed_verts lets the fixed base be rescaled per environment.
    variants = (
        gs.morphs.URDF(file="urdf/panda_bullet/panda.urdf", fixed=True, batch_fixed_verts=True),
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True, batch_fixed_verts=True),
        gs.morphs.URDF(file="urdf/kuka_iiwa/model.urdf", fixed=True, batch_fixed_verts=True),
        gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf", fixed=True, batch_fixed_verts=True),
    )
    robot = scene.add_entity(morph=variants)

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            color=(0.1, 0.6, 0.8, 0.6),
            use_force=True,
        ),
    )

    scene.build(n_envs=n_envs, env_spacing=(1.2, 1.2))
    gs.logger.info(
        f"Ragged robot fleet: superset {robot.n_links} links / {robot.n_dofs} DOFs; smaller variants leave "
        f"their extra slots inert."
    )

    # Position control holds each arm at a joint target against gravity; Y re-samples the target.
    robot.set_dofs_kp(np.full(robot.n_dofs, 200.0, dtype=np.float32))
    robot.set_dofs_kv(np.full(robot.n_dofs, 20.0, dtype=np.float32))
    limits = robot.get_dofs_limit()  # (lower, upper), each (n_dofs,)
    lower = np.nan_to_num(tensor_to_array(limits[0]), neginf=-np.pi)
    upper = np.nan_to_num(tensor_to_array(limits[1]), posinf=np.pi)
    rng = np.random.default_rng(args.seed)

    def apply_randomize_objects():
        """Swap which robot each environment simulates (runtime topology rebind via set_active_variant)."""
        robot.set_active_variant(rng.integers(0, len(variants), size=n_envs), envs_idx=list(range(n_envs)))
        robot.control_dofs_position(rng.uniform(lower, upper, size=(n_envs, robot.n_dofs)).astype(np.float32))

    def apply_randomize_size():
        """Give each environment a random isotropic scale (per-env geom scaling of the whole robot)."""
        scales = rng.uniform(0.6, 1.4, size=n_envs).astype(np.float32)
        robot.set_scale(np.repeat(scales[:, None], 3, axis=1))

    def apply_randomize_pose():
        """Re-sample each environment's joint-position target within the joint limits."""
        robot.control_dofs_position(rng.uniform(lower, upper, size=(n_envs, robot.n_dofs)).astype(np.float32))

    # Keybind callbacks fire on the viewer thread and must not launch GPU kernels concurrently with the main
    # loop's scene.step(); each only raises a request flag that the loop drains between steps (see run_pool).
    requested = {"objects": False, "size": False, "pose": False}
    is_running = [True]

    scene.viewer.register_keybinds(
        Keybind("randomize_objects", Key.R, KeyAction.PRESS, callback=lambda: requested.__setitem__("objects", True)),
        Keybind("randomize_size", Key.T, KeyAction.PRESS, callback=lambda: requested.__setitem__("size", True)),
        Keybind("randomize_pose", Key.Y, KeyAction.PRESS, callback=lambda: requested.__setitem__("pose", True)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=lambda: is_running.__setitem__(0, False)),
        overwrite=True,
    )

    if "PYTEST_VERSION" not in os.environ:
        apply_randomize_pose()

    print("\nArticulated controls:")
    print("R         - randomize each environment's robot")
    print("T         - randomize each environment's size")
    print("Y         - randomize each environment's joint pose")
    print("left-drag - grab and drag a link")
    print("ESC       - quit\n")

    while is_running[0] and scene.viewer.is_alive():
        if requested["objects"]:
            requested["objects"] = False
            apply_randomize_objects()
        if requested["size"]:
            requested["size"] = False
            apply_randomize_size()
        if requested["pose"]:
            requested["pose"] = False
            apply_randomize_pose()
        scene.step()
        if "PYTEST_VERSION" in os.environ:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_envs", type=int, default=9)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--articulated", action="store_true", default=False)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    if args.articulated:
        scene = gs.Scene(
            # Disable self-collision: random joint poses would otherwise drive an arm into self-penetration
            # (the neutral-config self-collision filter does not cover arbitrary poses) and blow up.
            rigid_options=gs.options.RigidOptions(enable_geom_scaling=True, enable_self_collision=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.4),
                enable_default_keybinds=False,
            ),
            show_viewer=True,
        )
        run_articulated(scene, args, n_envs)
    else:
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(enable_geom_scaling=True),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(4.0, -4.0, 3.0),
                camera_lookat=(0.0, 0.0, 0.2),
                enable_default_keybinds=False,
            ),
            show_viewer=True,
        )
        scene.add_entity(gs.morphs.Plane())
        run_pool(scene, args, n_envs)


if __name__ == "__main__":
    main()
