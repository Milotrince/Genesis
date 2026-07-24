"""Interactive demo of runtime heterogeneous variant switching.

A single entity is built from a list of morphs -- randomly sized boxes/spheres/cylinders, or (with --articulated)
2-link pendulum chains -- and each environment shows one of them. Press SPACE to randomize which variant every
environment shows, live, via `entity.set_entity_variant`; both the physics and the rendered geometry follow.
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.assets.procedural import build_articulated_chain
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_envs", type=int, default=4)
    parser.add_argument("-v", "--n_variants", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-f", "--use_force", action="store_true", help="Drag objects with a spring force instead of setting position"
    )
    parser.add_argument(
        "-a", "--articulated", action="store_true", help="Use 2-link pendulum chains instead of primitive shapes"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    rng = np.random.default_rng(args.seed)

    # A box, sphere and cylinder in rotation (or 2-link chains under --articulated), each at a random size.
    variants = []
    for i in range(args.n_variants):
        if args.articulated:
            # Fixed-base 2-link pendulum. Only the link radius varies across variants: the heterogeneous rebind
            # swaps geometry and inertial but not the joint anchors, so a different link length would leave the
            # second link attached at the primary variant's position.
            variants.append(
                gs.morphs.MJCF(
                    file=build_articulated_chain(
                        n_links=2,
                        link_radius=rng.uniform(0.015, 0.07),
                        link_length=0.25,
                    ),
                    pos=(0.0, 0.0, 0.8),
                )
            )
        elif i % 3 == 0:
            side = rng.uniform(0.12, 0.28)
            variants.append(gs.morphs.Box(size=(side, side, side), pos=(0.0, 0.0, 0.3)))
        elif i % 3 == 1:
            variants.append(gs.morphs.Sphere(radius=rng.uniform(0.08, 0.16), pos=(0.0, 0.0, 0.3)))
        else:
            variants.append(
                gs.morphs.Cylinder(radius=rng.uniform(0.07, 0.13), height=rng.uniform(0.15, 0.35), pos=(0.0, 0.0, 0.3))
            )

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.4 if args.articulated else 0.1),
        ),
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    het = scene.add_entity(
        morph=variants,
    )
    # Mouse-drag to sanity-check the switched-in geometry's collision and mass.
    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=args.use_force,
            color=(0.1, 0.6, 0.8, 0.6),
        )
    )
    # Every variant needs an env at build to get a render node.
    n_envs = max(args.n_envs, len(variants))
    scene.build(
        n_envs=n_envs,
        env_spacing=(1.0, 1.0),
    )

    n_variants = len(variants)
    is_running = True
    pending_randomize = True  # start from a random arrangement

    def randomize():
        nonlocal pending_randomize
        pending_randomize = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("randomize_variants", Key.SPACE, KeyAction.PRESS, callback=randomize),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )
    print("\nSPACE randomizes each environment's variant, drag objects with the mouse, ESC to quit.\n")

    while is_running:
        # Keybind callbacks fire on the viewer thread; mutate the scene here on the stepping thread.
        if pending_randomize:
            pending_randomize = False
            het.set_entity_variant(rng.integers(0, n_variants, size=n_envs))
            if args.articulated:
                # Kick the hinges so the fresh chain swings.
                het.set_dofs_velocity(rng.uniform(-5.0, 5.0, size=het.n_dofs))
            else:
                # Drop the fresh geometry so the switch reads clearly.
                het.set_pos((0.0, 0.0, 0.3))
                het.set_dofs_velocity(np.zeros(6))
        scene.step()

        if "PYTEST_VERSION" in os.environ:
            break


if __name__ == "__main__":
    main()
