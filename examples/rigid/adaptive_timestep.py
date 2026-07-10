import argparse
import os

import numpy as np

import genesis as gs
import genesis.vis.keybindings as kb
from genesis.utils.misc import qd_to_numpy
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE

##### Per-island adaptive timestep demo #####
# A pile of stacked boxes plus objects scattered across the plane, with a small sphere dropped onto the pile. With
# adaptive timestep on, each contact island integrates at macro_dt / rate, chosen automatically from a geometry CFL
# (no per-scene tuning): the resting scattered objects stay at the macro timestep while the fast-falling sphere (and
# the pile it hits) sub-step. With --vis the per-island timesteps are read back through the solver's internal
# `dofs_rate` field (deliberately jank - nothing is added to the engine to expose them) and streamed to a live line
# plot; drag objects around with the mouse (MouseInteractionPlugin) and watch the timesteps react, Esc to quit.


def main():
    parser = argparse.ArgumentParser(description="Per-island adaptive timestep demo with a live timestep plot.")
    parser.add_argument("--vis", "-v", action="store_true", help="Show the interactive 3D viewer + live timestep plot.")
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-island adaptive timestep (use --no-adaptive for uniform stepping).",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        rigid_options=gs.options.RigidOptions(
            use_adaptive_timestep=args.adaptive,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )

    # A pile: boxes stacked in contact settle into a single island that shares one rate.
    box_size = 0.2
    pile = [
        scene.add_entity(
            gs.morphs.Box(
                size=(box_size, box_size, box_size),
                pos=(0.0, 0.0, box_size * (i + 0.5)),
            ),
        )
        for i in range(4)
    ]

    # Objects scattered across the plane, each resting in its own island (should stay at the macro timestep).
    scattered = [
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.4, 0.4, 0.1))),
        scene.add_entity(gs.morphs.Sphere(radius=0.12, pos=(-1.1, 0.9, 0.12))),
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.7, -1.3, 0.1))),
        scene.add_entity(gs.morphs.Sphere(radius=0.12, pos=(-1.4, -0.7, 0.12))),
    ]

    # A small sphere dropped from high up: it falls fast relative to its size, so the geometry CFL sub-steps its island
    # during the fall and impact, then relaxes it back to the macro timestep once it settles.
    faller = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.06, 0.0, 2.0),
        ),
    )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
                color=(0.1, 0.6, 0.8, 0.6),
            )
        )

    # Track one representative object per island (the pile's boxes share an island, so one stands in for it). The live
    # plot has a fixed set of lines, so the number of tracked objects must stay constant over the run.
    tracked = [("faller", faller), ("pile", pile[0])]
    tracked += [(f"scattered_{i}", entity) for i, entity in enumerate(scattered)]
    labels = tuple(name for name, _ in tracked)

    # A mutable holder the run loop refreshes after each step; the recorder samples it inside scene.step(). Seeded with
    # the macro timestep (dt / substeps) so the plot's line count is fixed from the first sample, before the loop runs.
    latest_dts = [np.full(len(tracked), 0.01)]

    def plot_data():
        return latest_dts[0]

    recording = False
    if args.vis:
        plot_kwargs = dict(
            title=f"Per-island timestep ({'adaptive' if args.adaptive else 'uniform'})",
            labels=labels,
            x_label="step",
            y_label="island timestep dt (s)",
            hz=30.0,
            history_length=2000,
        )
        if IS_PYQTGRAPH_AVAILABLE:
            scene.start_recording(plot_data, gs.recorders.PyQtLinePlot(**plot_kwargs))
            recording = True
        elif IS_MATPLOTLIB_AVAILABLE:
            scene.start_recording(plot_data, gs.recorders.MPLLinePlot(**plot_kwargs))
            recording = True
        else:
            print("matplotlib or pyqtgraph not found, skipping the live timestep plot.")

    scene.build(n_envs=1)

    solver = scene.sim.rigid_solver
    macro_dt = solver._substep_dt
    island_state = solver.constraint_solver.island_state
    tracked_dofs = [entity.dof_start for _, entity in tracked]

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.RELEASE, callback=stop),
        )

    horizon = 400
    step = 0
    try:
        while is_running and (args.vis or step < horizon):
            scene.step()
            step += 1

            if recording:
                # Jank: read the per-DOF rate straight out of the solver's island state to recover each island's dt.
                # dofs_rate is only allocated when adaptive timestep is on; without it every island runs at the macro dt.
                if args.adaptive:
                    dofs_rate = qd_to_numpy(island_state.dofs_rate, transpose=True)  # [B, n_dofs]
                    latest_dts[0] = np.array(
                        [macro_dt / max(int(dofs_rate[0, dof]), 1) for dof in tracked_dofs],
                        dtype=gs.np_float,
                    )
                else:
                    latest_dts[0] = np.full(len(tracked_dofs), macro_dt)

            if "PYTEST_VERSION" in os.environ:
                break

            if args.vis and not scene.viewer.is_alive():
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        if recording:
            scene.stop_recording()


if __name__ == "__main__":
    main()
