"""
Heterogeneous per-environment topology
======================================

Each parallel environment can simulate an articulated entity with a DIFFERENT kinematic structure - a
different number of links, joints, or degrees of freedom - from a single `add_entity(morph=[variant, ...])`
call. The variants share one statically-shaped device layout sized to the per-slot maximum over the variants
(the "superset"); a narrower variant leaves its unused link/DOF slots inert (massless, frozen), so every
environment steps in the same batched kernels. This is the backbone for cross-embodiment RL (one policy over a
fleet of different robots in parallel) and morphology domain randomization.

Here one entity is a 1-, 2-, or 3-link pendulum depending on the environment; all swing under gravity, each as
its own chain length.

Usage:
    python heterogeneous_topology.py            # 6 environments, GPU
    python heterogeneous_topology.py -n 12
    python heterogeneous_topology.py --cpu
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs


def pendulum_mjcf(n_links):
    """An n-link planar pendulum hanging from the world, each link a capsule on a hinge about y."""
    mjcf = ET.Element("mujoco", model=f"pendulum_{n_links}")
    ET.SubElement(mjcf, "option", gravity="0 0 -9.81")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = worldbody
    for i_link in range(n_links):
        body = ET.SubElement(parent, "body", name=f"l{i_link}", pos=("0 0 0.6" if i_link == 0 else "0.2 0 0"))
        ET.SubElement(body, "joint", name=f"j{i_link}", type="hinge", axis="0 1 0")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", mass="0.1")
        parent = body
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=6)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=args.vis,
    )

    # The first morph (most links) defines the skeleton; the shorter variants inert their unused link slots.
    pendulum = scene.add_entity(
        morph=(
            gs.morphs.MJCF(file=pendulum_mjcf(3)),
            gs.morphs.MJCF(file=pendulum_mjcf(2)),
            gs.morphs.MJCF(file=pendulum_mjcf(1)),
        ),
    )

    scene.build(n_envs=n_envs)
    gs.logger.info(f"Entity superset: {pendulum.n_links} links, {pendulum.n_dofs} DOFs (each env uses a subset).")

    # Give the pendulums an initial deflection so they swing visibly.
    pendulum.set_dofs_position(np.full((n_envs, pendulum.n_dofs), 0.5, dtype=np.float32))

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 1
    for _ in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
