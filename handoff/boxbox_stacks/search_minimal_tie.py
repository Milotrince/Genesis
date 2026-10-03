"""Search simple plate-on-cube configurations for the box-box tie kick, one configuration per environment.

    PYTHONPATH=$PWD python handoff/boxbox_stacks/search_minimal_tie.py --scale 0.4 --pile-x -3.0

A plate rests on a fixed base on one of the faces the stacks test uses, and a cube rests on the plate, with a grid of
yaws, relative yaws and rolls in degrees. Each environment is simulated for 50 steps of the test's settings, and the
configurations whose mechanical energy rises above 1e-3 m g scale are printed. None of about 2,100 configurations
tried so far (scales 0.1, 0.4 and 2.0, pile at the origin and at x = -3) kicked on the unfixed engine.
"""

import argparse
from itertools import product

import numpy as np

import genesis as gs
import genesis.utils.geom as gu

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--pile-x", type=float, default=-3.0)
args = parser.parse_args()

BOX_EULER_ROTS = np.array(((0, 0, 0), (180, 0, 0), (90, 0, 0), (-90, 0, 0), (0, -90, 0), (0, 90, 0)), dtype=float)
# plate face, cube face, plate yaw, cube yaw relative to the plate, cube roll
grid = np.array(list(product((0, 1), (0, 2, 4), (0.0, 30.0, 45.0), (0.5, 1.0, 2.0, 3.0), (0.01, 0.05, 0.1, 0.5))))
n_envs = len(grid)
scale = args.scale

gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=0)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=2, gravity=(0.0, 0.0, -9.81)),
    rigid_options=gs.options.RigidOptions(box_box_detection=True, use_hibernation=False),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Box(pos=(args.pile_x, 0.0, 0.5 * scale), size=scale * np.array((2.0, 2.0, 1.0)), fixed=True))
plate = scene.add_entity(gs.morphs.Box(pos=(args.pile_x, 0.0, 2.0 * scale), size=scale * np.array((1.0, 0.6, 0.02))))
cube = scene.add_entity(gs.morphs.Box(pos=(args.pile_x, 0.0, 3.0 * scale), size=scale * np.array((0.3, 0.3, 0.3))))
scene.build(n_envs=n_envs)

zeros = np.zeros(n_envs)
plate_quat = gu.transform_quat_by_quat(
    gu.euler_to_quat(BOX_EULER_ROTS[grid[:, 0].astype(int)]), gu.euler_to_quat(np.stack((zeros, zeros, grid[:, 2]), -1))
)
cube_quat = gu.transform_quat_by_quat(
    gu.euler_to_quat(BOX_EULER_ROTS[grid[:, 1].astype(int)]),
    gu.euler_to_quat(np.stack((grid[:, 4], zeros, grid[:, 2] + grid[:, 3]), -1)),
)
plate.set_pos(np.tile((args.pile_x, 0.0, 1.01 * scale), (n_envs, 1)))
plate.set_quat(plate_quat)
cube.set_pos(np.tile((args.pile_x, 0.0, 1.17 * scale), (n_envs, 1)))
cube.set_quat(cube_quat)

energy_unit = float(plate.get_mass() + cube.get_mass()) * 9.81 * scale
energy_0 = scene.rigid_solver.get_kinetic_energy() + scene.rigid_solver.get_potential_energy()
energy_rise = np.zeros(n_envs)
for _ in range(50):
    scene.step()
    energy = scene.rigid_solver.get_kinetic_energy() + scene.rigid_solver.get_potential_energy()
    energy_rise = np.maximum(energy_rise, ((energy - energy_0) / energy_unit).cpu().numpy())
envs_kicked = np.flatnonzero(energy_rise > 1e-3)
print(f"{envs_kicked.size} of {n_envs} configurations kicked (plate face, cube face, plate yaw, rel. yaw, roll):")
for i_env in envs_kicked:
    print(f"  {grid[i_env].tolist()}  energy rise {energy_rise[i_env]:.3f}")
