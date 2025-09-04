#!/usr/bin/env python3
"""
LiDAR/Depth Camera Visualization and Keyboard Teleoperation

- LiDAR: shows point clouds as debug spheres
- Depth camera: shows live depth image (H x W)
"""

import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
import torch
from pynput import keyboard

import genesis as gs
from genesis.sensors.raycaster.camera_pattern import DepthCameraPattern
from genesis.sensors.raycaster.lidar_pattern import (
    LivoxPattern,
    SphericalPattern,
    SpinningLidarPattern,
)
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import tensor_to_array


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)


def build_scene(show_viewer: bool = True) -> gs.Scene:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            gravity=(0.0, 0.0, -9.81),
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # A ring of obstacles to visualize raycaster sensor hits
    inner_radius = 3.0
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Cylinder(height=1.5, radius=0.3, pos=(x, y, 0.75), fixed=True))

    outer_radius = 5.0
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, 2.0), pos=(x, y, 1.0), fixed=True))

    return scene


def create_robot_with_lidar(scene, args):
    """
    Create fixed-base robot with a LiDAR or Depth Camera sensor attached.

    Parameters
    ----------
    scene : gs.Scene
        The scene to create the robot in.
    args : argparse.Namespace
        The arguments to create the robot with.

    Returns
    -------
    robot : gs.engine.entities.RigidEntity
        The robot entity.
    sensor : gs.sensors.Raycaster
        The LiDAR or Depth Camera sensor.
    """

    robot_kwargs = dict(
        pos=(0.0, 0.0, 0.35),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed=True,
    )

    if args.use_box:
        robot = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), **robot_kwargs))
    else:
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", **robot_kwargs))

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, -0.06),
        euler_offset=(180.0, 15.0, 0.0),
        return_world_frame=True,
    )

    if args.pattern == "depth":
        width = int(args.dc_width)
        height = int(args.dc_height)
        pattern_cfg = DepthCameraPattern(width=width, height=height)
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=pattern_cfg, **sensor_kwargs))
        return robot, sensor

    elif args.pattern == "livox":
        pattern_cfg = LivoxPattern(sensor_type=args.sensor_type)
    elif args.pattern == "spinning":
        pattern_cfg = SpinningLidarPattern(sensor_type=args.sensor_type)
    else:
        pattern_cfg = SphericalPattern(n_scan_lines=16, n_points_per_line=64, fov_vertical=30.0, fov_horizontal=360.0)
    sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))
    return robot, sensor


# ------------------------- Teleop + Visualization -------------------------
COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # red-ish
    (0.2, 1.0, 0.2, 1.0),  # green-ish
    (0.2, 0.6, 1.0, 1.0),  # blue-ish
    (1.0, 1.0, 0.2, 1.0),  # yellow-ish
]


def run(scene: gs.Scene, robot, sensor: gs.sensors.Lidar, n_envs: int, kb: KeyboardDevice, is_depth: bool = False):
    # Build scene with environments
    scene.build(n_envs=n_envs)

    print("\nKeyboard Controls:")
    print("↑/↓/←/→: Move XY, n/m: Up/Down, u/o: Roll CCW/CW, i/k: Pitch Up/Down, j/l: Yaw CCW/CW, r: Reset, esc: Quit")

    # Initial pose
    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    # For clearing previous visualization
    point_nodes: list[object | None] = [None] * n_envs

    # Depth image viewer
    depth_im = None
    fig = ax = None
    if is_depth:
        plt.ion()
        fig, ax = plt.subplots(num="Depth Image")
        ax.set_title("Depth (m)")

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        # Set the same pose for each environment instance
        pos_t = torch.tensor(pos_np, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        quat_t = torch.tensor(quat_np, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        for env_idx in range(n_envs):
            robot.set_pos(pos_t, envs_idx=[env_idx], zero_velocity=False)
            robot.set_quat(quat_t, envs_idx=[env_idx], zero_velocity=False)

    # Reset once at start
    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    # Main loop
    sphere_radius = 0.02
    lidar_interval = 2  # steps
    step = 0
    target_pos[2] += 0.2
    try:
        while True:
            # Handle keyboard
            pressed = kb.pressed_keys.copy()
            if keyboard.Key.esc in pressed:
                break
            if keyboard.KeyCode.from_char("r") in pressed:
                target_pos[:] = init_pos
                target_euler[:] = init_euler

            # Motion increments
            dpos = 0.03
            dangle = 0.04
            if keyboard.Key.up in pressed:
                target_pos[0] += dpos
            if keyboard.Key.down in pressed:
                target_pos[0] -= dpos
            if keyboard.Key.right in pressed:
                target_pos[1] -= dpos
            if keyboard.Key.left in pressed:
                target_pos[1] += dpos
            if keyboard.KeyCode.from_char("n") in pressed:
                target_pos[2] += dpos
            if keyboard.KeyCode.from_char("m") in pressed:
                target_pos[2] -= dpos

            # Orientation increments
            if keyboard.KeyCode.from_char("u") in pressed:
                target_euler[0] += dangle  # roll CCW around +X
            if keyboard.KeyCode.from_char("o") in pressed:
                target_euler[0] -= dangle  # roll CW around +X
            if keyboard.KeyCode.from_char("i") in pressed:
                target_euler[1] += dangle  # pitch up around +Y
            if keyboard.KeyCode.from_char("k") in pressed:
                target_euler[1] -= dangle  # pitch down around +Y
            if keyboard.KeyCode.from_char("j") in pressed:
                target_euler[2] += dangle  # yaw CCW around +Z
            if keyboard.KeyCode.from_char("l") in pressed:
                target_euler[2] -= dangle  # yaw CW around +Z

            # apply pose
            quat = euler_to_quat(target_euler)
            apply_pose_to_all_envs(target_pos, quat)

            # Step physics
            scene.step()

            # Update visualization periodically
            if step % lidar_interval == 0:

                # Draw point cloud only for LiDAR patterns
                if not is_depth:
                    hits = sensor.read()
                    hit_points = tensor_to_array(hits["hit_points"])
                    hit_distances = tensor_to_array(hits["hit_ranges"])
                    for env_idx in range(n_envs):
                        valid = hit_distances[env_idx] < sensor._options.max_range
                        if np.any(valid):
                            pts = hit_points[env_idx][valid]
                            if point_nodes[env_idx] is not None:
                                scene.clear_debug_object(point_nodes[env_idx])
                            color = COLORS[env_idx % len(COLORS)]
                            point_nodes[env_idx] = scene.draw_debug_spheres(pts, radius=sphere_radius, color=color)

                # Show depth image for depth camera
                else:
                    img_data = sensor.read()[0]  # first env
                    if depth_im is None:
                        depth_im = ax.imshow(
                            img_data,
                            vmin=0.0,
                            vmax=sensor._options.max_range,
                            cmap="plasma",
                            origin="upper",
                            aspect="auto",
                        )
                        fig.colorbar(depth_im, ax=ax)
                    else:
                        depth_im.set_data(img_data)
                    ax.set_xlabel("width (W)")
                    ax.set_ylabel("height/scan (H/S)")
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        for node in point_nodes:
            if node is not None:
                scene.clear_debug_object(node)
        if is_depth and plt.get_fignums():
            plt.close(fig)


# ------------------------- Main -------------------------


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Visualization with Keyboard Teleop")
    parser.add_argument("--n-envs", type=int, default=2, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--use-box", action="store_true", help="Use Go2 instead of Box")
    parser.add_argument(
        "--pattern",
        type=str,
        default="livox",
        choices=["spherical", "livox", "spinning", "depth"],
        help="Sensor pattern type",
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        default="horizon",
        choices=[
            # Livox LiDAR
            "avia",
            "HAP",
            "horizon",
            "mid40",
            "mid70",
            "mid360",
            "tele",
            # Spinning LiDAR
            "hdl64",
            "vlp32",
            "os128",
        ],
        help="Sensor model (depends on --pattern)",
    )
    parser.add_argument("--max-range", type=float, default=20.0, help="Max range (m)")

    # Spinning-specific optional overrides
    parser.add_argument("--f-rot", type=float, default=None, help="Spinning lidar rotation frequency (Hz)")
    parser.add_argument("--sample-rate", type=float, default=None, help="Spinning lidar sample rate (samples/sec)")
    parser.add_argument("--n-channels", type=int, default=None, help="Spinning lidar channel count")

    # Depth camera options
    parser.add_argument("--dc-width", type=int, default=100, help="Depth camera image width")
    parser.add_argument("--dc-height", type=int, default=100, help="Depth camera image height")
    parser.add_argument("--dc-fx", type=float, default=None, help="Depth camera fx (pixels)")
    parser.add_argument("--dc-fy", type=float, default=None, help="Depth camera fy (pixels)")
    parser.add_argument("--dc-cx", type=float, default=None, help="Depth camera cx (pixels)")
    parser.add_argument("--dc-cy", type=float, default=None, help="Depth camera cy (pixels)")
    parser.add_argument("--dc-fov-h", type=float, default=None, help="Depth camera horizontal FOV (deg)")
    parser.add_argument("--dc-fov-v", type=float, default=None, help="Depth camera vertical FOV (deg)")

    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    kb = KeyboardDevice()
    kb.start()

    scene = build_scene(show_viewer=True)
    robot, lidar = create_robot_with_lidar(scene, args)

    run(scene, robot, lidar, n_envs=args.n_envs, kb=kb, is_depth=args.pattern == "depth")


if __name__ == "__main__":
    main()
