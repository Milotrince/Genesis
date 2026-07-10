import argparse
import os

import numpy as np

import genesis as gs
import genesis.vis.keybindings as kb
from genesis.utils.misc import qd_to_numpy

##### Per-island adaptive timestep demo #####
# A pile of stacked boxes plus objects scattered across the plane, with a sphere dropped onto the pile. With adaptive
# timestep on, each contact island integrates at macro_dt / rate: the resting scattered objects stay at the macro
# timestep while the falling sphere (and the pile it hits) sub-step. The per-island timesteps are read back through the
# solver's internal `dofs_rate` field (deliberately jank - nothing is added to the engine to expose them) and plotted
# over time. Enable the viewer with --vis to drag objects around with the mouse (MouseInteractionPlugin) and watch the
# timesteps react; press Esc to quit.


def main():
    parser = argparse.ArgumentParser(description="Per-island adaptive timestep demo with timestep plotting.")
    parser.add_argument("--vis", "-v", action="store_true", help="Show the interactive viewer (mouse-draggable).")
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
            adaptive_timestep_max_rate=8,
            adaptive_timestep_ref_speed=0.5,
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

    # A sphere dropped onto the pile: fast while falling and on impact (small dt), settling back to the macro dt.
    faller = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.12,
            pos=(0.06, 0.0, 1.8),
        ),
    )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
                color=(0.1, 0.6, 0.8, 0.6),
            )
        )

    scene.build(n_envs=1)

    # Track one representative DOF per logical object; the pile's boxes share an island, so one stands in for it.
    tracked = {"faller": faller.dof_start, "pile": pile[0].dof_start}
    for i, entity in enumerate(scattered):
        tracked[f"scattered_{i}"] = entity.dof_start

    solver = scene.sim.rigid_solver
    macro_dt = solver._substep_dt
    island_state = solver.constraint_solver.island_state

    dt_series = {name: [] for name in tracked}

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

            # Jank: read the per-DOF rate straight out of the solver's island state. dofs_rate is only allocated when
            # adaptive timestep is on; without it every island runs at the macro timestep.
            if args.adaptive:
                dofs_rate = qd_to_numpy(island_state.dofs_rate, transpose=True)  # [B, n_dofs]
                for name, dof in tracked.items():
                    dt_series[name].append(macro_dt / max(int(dofs_rate[0, dof]), 1))
            else:
                for name in tracked:
                    dt_series[name].append(macro_dt)

            if "PYTEST_VERSION" in os.environ:
                break

            if args.vis and not scene.viewer.is_alive():
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")

    if "PYTEST_VERSION" not in os.environ:
        _plot_timesteps(dt_series, macro_dt, args.adaptive)


def _plot_timesteps(dt_series, macro_dt, adaptive):
    import matplotlib.pyplot as plt

    n = len(next(iter(dt_series.values())))
    times = np.arange(n) * macro_dt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, series in dt_series.items():
        width = 2.4 if name in ("faller", "pile") else 1.2
        ax.step(times, series, where="post", label=name, linewidth=width)
    ax.axhline(macro_dt, color="0.6", linestyle="--", linewidth=1.0, label="macro dt")
    ax.set_yscale("log")
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("island timestep dt (s)")
    ax.set_title(f"Per-island adaptive timestep ({'ON' if adaptive else 'OFF'})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()

    out_path = os.path.abspath("adaptive_timestep_dts.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    gs.logger.info(f"Saved timestep plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
