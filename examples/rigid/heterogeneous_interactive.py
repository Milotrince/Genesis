"""
Interactive Heterogeneous Simulation
====================================

Each parallel environment simulates a different geometry variant of the same entity, and the active variant
of every environment can be re-randomized live with the ``R`` key. This showcases ``set_active_variant``, the
runtime rebind that swaps an environment's collision geometry and inertial in place while the scene keeps
running (the joint configuration is preserved; here we also respawn the object so the new shape drops fresh).

Usage:
    python heterogeneous_interactive.py            # 9 environments, GPU
    python heterogeneous_interactive.py -n 16      # 16 environments
    python heterogeneous_interactive.py --cpu      # run on CPU

Controls:
    R   - randomize the active object variant of every environment
    ESC - quit
    (plus the usual viewer camera controls)
"""

import argparse

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

SPAWN_POS = (0.0, 0.0, 0.35)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_envs", type=int, default=9)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, -4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    scene.add_entity(gs.morphs.Plane())

    # The object is built with a list of morphs: every environment simulates one of these variants, and
    # set_active_variant can rebind any environment to any of them at runtime.
    variants = [
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=SPAWN_POS),
        gs.morphs.Sphere(radius=0.07, pos=SPAWN_POS),
        gs.morphs.Cylinder(radius=0.06, height=0.16, pos=SPAWN_POS),
        gs.morphs.Box(size=(0.07, 0.16, 0.07), pos=SPAWN_POS),
    ]
    obj = scene.add_entity(morph=variants)

    ########################## build ##########################
    scene.build(n_envs=n_envs, env_spacing=(0.6, 0.6))

    n_variants = len(variants)
    rng = np.random.default_rng(args.seed)
    spawn_pos = np.tile(np.array(SPAWN_POS, dtype=np.float32), (n_envs, 1))

    def randomize():
        """Rebind every environment to a random variant and drop it fresh from the spawn height."""
        variant_idx = rng.integers(0, n_variants, size=n_envs)
        obj.set_active_variant(variant_idx)
        obj.set_pos(spawn_pos, zero_velocity=True)
        gs.logger.info(f"Randomized active variants: {variant_idx.tolist()}")

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    # Start with a varied assignment, then let R re-randomize live. R overwrites the default record-video bind.
    randomize()
    scene.viewer.register_keybinds(
        Keybind("randomize_variants", Key.R, KeyAction.PRESS, callback=randomize, allow_overload=False),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    print("\nHeterogeneous controls:")
    print("R   - randomize the active object variant of every environment")
    print("ESC - quit\n")

    while is_running and scene.viewer.is_alive():
        scene.step()


if __name__ == "__main__":
    main()
