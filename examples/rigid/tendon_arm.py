import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(precision="32", logging_level="info")

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.2, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    ########################## a self-authored tendon-driven arm ##########################
    # A spatial tendon ("flexor") is routed from a fixed anchor, wraps around a cylinder pulley, and attaches to the
    # forearm tip. A position actuator drives the tendon length to flex the arm. A fixed tendon ("couple") links the
    # two joints. See genesis/assets/xml/tendon_arm.xml.
    arm = scene.add_entity(gs.morphs.MJCF(file="xml/tendon_arm.xml"))

    scene.build()

    print(f"tendons: {[t.name for t in arm.tendons]}")
    flexor = arm.get_tendon("flexor")

    ########################## pull on the flexor tendon to flex the arm ##########################
    for i in range(1000):
        # Oscillate the target tendon length between slack and taut.
        phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / 300.0))
        target = 0.6 + 0.5 * phase
        arm.solver.control_tendons_position(np.array([target]), tendons_idx=[flexor.idx])
        scene.step()


if __name__ == "__main__":
    main()
