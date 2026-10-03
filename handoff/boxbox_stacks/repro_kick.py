"""Reproduce the phantom box-box contact that kicks a resting stack, from the state in kick_state.json.

Run from the repository root with PYTHONPATH pointing at it, so this checkout's engine is imported:

    PYTHONPATH=$PWD python handoff/boxbox_stacks/repro_kick.py replay   # 100 substeps from the recorded start
    PYTHONPATH=$PWD python handoff/boxbox_stacks/repro_kick.py detect   # one collision pass on the pre-kick state

On an engine without the tolerance fix in func_box_box_contact, 'replay' shows the energy jumping by about 3.3 J at
substep 87 and 'detect' lists a plate-cube (box1-box3) contact with a 0.1445 m penetration.
"""

import json
import os
import sys

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

MODE = sys.argv[1] if len(sys.argv) > 1 else "replay"
state = json.load(open(os.path.join(os.path.dirname(__file__), "kick_state.json")))

gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=0)
# One substep per step, so that every collision pass is observable (the test uses dt=0.01 with 2 substeps)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=1, gravity=(0.0, 0.0, -9.81)),
    rigid_options=gs.options.RigidOptions(box_box_detection=True, use_hibernation=False),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Box(pos=state["base_pos"], size=state["base_size"], fixed=True))
boxes = [
    scene.add_entity(gs.morphs.Box(pos=pos, quat=quat, size=size))
    for pos, quat, size in zip(state["start_pos"], state["start_quat"], state["boxes_size"])
]
scene.build()
geoms_name = {0: "base", **{box.geoms[0].idx: f"box{i}" for i, box in enumerate(boxes)}}


def print_contacts():
    contacts = {k: tensor_to_array(v) for k, v in scene.rigid_solver.collider.get_contacts().items()}
    for i_ga, i_gb, pos, pen in zip(
        contacts["geom_a"], contacts["geom_b"], contacts["position"], contacts["penetration"]
    ):
        print(f"  {geoms_name[i_ga]:>4}-{geoms_name[i_gb]:<4} pos {np.round(pos, 4)} penetration {pen:.4e}")


if MODE == "detect":
    for box, pos, quat in zip(boxes, state["substep86_pos"], state["substep86_quat"]):
        box.set_pos(pos)
        box.set_quat(quat)
    scene.step()
    print("Contacts detected on the pre-kick state:")
    print_contacts()
else:
    energy = []
    for i_step in range(101):
        if i_step > 0:
            scene.step()
        energy.append(float(scene.rigid_solver.get_kinetic_energy() + scene.rigid_solver.get_potential_energy()))
    energy_jump = np.diff(energy)
    i_jump = int(np.argmax(energy_jump))
    print(f"Largest energy jump: {energy_jump[i_jump]:.4f} J at substep {i_jump + 1}")
    print(f"Energy rise over the run: {max(energy) - energy[0]:.4f} J")
