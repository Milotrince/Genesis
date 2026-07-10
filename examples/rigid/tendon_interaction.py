"""
Interactive Tendon Arm
======================

A 2-link arm driven by a MuJoCo-style SPATIAL tendon that routes from a fixed anchor, wraps around a cylinder
"pulley", and attaches to the forearm tip; a FIXED tendon couples the two joints, and a position actuator drives the
spatial tendon's length. Grab the arm with the mouse to feel the tendon resist, and tighten/loosen the flexor tendon
live. See genesis/assets/xml/tendon_arm.xml.

Usage:
    python tendon_interaction.py

Controls:
    up / down   - tighten / loosen the flexor tendon
    left-drag   - grab and drag the arm (MouseInteraction plugin)
    ESC         - quit
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position", "-p", action="store_true", help="Drag by setting position instead of applying spring forces"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -1.6, 1.6),
            camera_lookat=(0.2, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    arm = scene.add_entity(
        gs.morphs.MJCF(file="xml/tendon_arm.xml"),
    )

    # Drag the arm with the mouse; spring-force mode (default) lets the tendon visibly resist your pull.
    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=not args.position,
            color=(0.9, 0.4, 0.2, 0.6),
        ),
    )

    scene.build()

    # Start in a pose where the flexor tendon drapes over the pulley, so the wrap is visible from the first frame.
    arm.set_qpos(np.array([0.75, -0.5]))

    flexor_idx = arm.get_tendon("flexor").idx

    # The viewer runs in its own thread, so keybind callbacks fire there. They only nudge this plain float; the GPU
    # tendon-control call runs on the stepping thread below, keeping every kernel launch on one thread.
    target_length = float(arm.solver.get_tendons_length(tendons_idx=[flexor_idx])[0])
    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    def tighten():
        nonlocal target_length
        target_length = max(0.2, target_length - 0.05)

    def loosen():
        nonlocal target_length
        target_length += 0.05

    scene.viewer.register_keybinds(
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        Keybind("tighten_flexor", Key.UP, KeyAction.PRESS, callback=tighten),
        Keybind("loosen_flexor", Key.DOWN, KeyAction.PRESS, callback=loosen),
        overwrite=True,
    )

    print("\nTendon-arm controls:")
    print("up / down - tighten / loosen the flexor tendon")
    print("left-drag - grab and drag the arm")
    print("ESC       - quit\n")

    tendon_nodes = []
    while is_running and scene.viewer.is_alive():
        arm.solver.control_tendons_position(np.array([target_length]), tendons_idx=[flexor_idx])
        scene.step()

        # Visualize the spatial tendon's routed/wrapped path (clear only our own lines so the drag overlay stays).
        for node in tendon_nodes:
            scene.clear_debug_object(node)
        tendon_nodes = arm.draw_debug_tendons(color=(0.9, 0.2, 0.2, 1.0), radius=0.006)

        if "PYTEST_VERSION" in os.environ:
            break


if __name__ == "__main__":
    main()
