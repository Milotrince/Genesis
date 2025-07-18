import argparse
import os
import pickle
import shutil
from importlib import metadata
from math import pi

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
import genesis as gs
from hand_env import InHandRotateEnv
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,  # Reduced learning rate for stability
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
            "min_std": 0.01,  # Minimum standard deviation to prevent collapse
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": False,  # Disable normalization that might cause issues
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 16,  # should match dofs in the hand (length of joint_names)
        # hand configuration
        "friction": 1.0,
        "tactile_sensors": {  # link name : tactile grid size
            "index_3_tip": (2, 2, 2),
            "middle_3_tip": (2, 2, 2),
            "ring_3_tip": (2, 2, 2),
            "thumb_3_tip": (2, 2, 2),
        },
        "joint_names": [  # Joint names in order for Allegro hand
            "index_roll",
            "middle_roll",
            "ring_roll",
            "thumb_bend0",
            "index_bend0",
            "middle_bend0",
            "ring_bend0",
            "thumb_roll",
            "index_bend1",
            "middle_bend1",
            "ring_bend1",
            "thumb_bend1",
            "index_bend2",
            "middle_bend2",
            "ring_bend2",
            "thumb_bend2",
        ],
        "default_joint_angles": [
            0.1,
            0.0,
            -0.1,
            0.7,
            0.6,
            0.6,
            0.6,
            1.0,
            0.65,
            0.65,
            0.65,
            1.0,
            0.6,
            0.6,
            0.6,
            0.7,
        ],
        # PD control parameters
        "kp": 40.0,
        "kd": 1.0,
        # termination conditions
        "max_obj_distance": 0.16,  # terminate if object moves too far from hand
        # initial poses
        "hand_init_pos": [0.0, 0.0, 0.5],
        "hand_init_euler": [0.0, -90.0, 0.0],
        "obj_init_pos": [-0.02, 0.0, 0.62],
        "obj_init_euler": [0.0, 0.0, 0.0],  # roll, pitch, yaw in degrees
        "obj_size": [0.09, 0.12, 0.03],
        # episode settings
        "episode_length_s": 10.0,
        "resampling_time_s": 2.0,
        "action_scale": 0.5,
        "clip_actions": 10.0,
        # visualization
        "max_visualize_FPS": 60,
    }

    # Calculate observation dimensions
    tactile_dim = 0
    for _, grid_size in env_cfg["tactile_sensors"].items():
        tactile_dim += grid_size[0] * grid_size[1] * grid_size[2] * 3  # 3 force components (fx, fy, fz)

    obs_cfg = {
        # obj_pos (3) + obj_quat (4) + obj_ang_vel (3) + commands (2) + dof_pos (16) + dof_vel (16) + actions (16)
        "num_obs": 3 + 4 + 3 + 2 + 16 + 16 + 16 + tactile_dim,
        "obs_scales": {
            "obj_pos": 1.0,
            "obj_quat": 1.0,
            "ang_vel": 1.0 / 10.0,  # normalize angular velocity
            "dof_pos": 1.0,
            "dof_vel": 0.1,
            "tactile": 1.0,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "target_obj_rot": 1.0,
        }  # Empty - you can add your own rewards
    }

    command_cfg = {
        "num_commands": 2,
        "target_obj_rot_radians": [0, pi],
        "target_obj_rot_axis": [2],  # [0, 1, 2],  # 0: x-axis, 1: y-axis, 2: z-axis
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="allegro-rotation")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = InHandRotateEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.git_status_repos = []

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/sensors/in_hand_rotate/allegro_train.py
"""
