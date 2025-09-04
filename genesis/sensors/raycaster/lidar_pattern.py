"""
Taichi-based LiDAR pattern generation for Genesis.

This module provides various ray patterns for LiDAR sensors, implemented using Taichi
for efficient computation. It mirrors the functionality of the Warp-based patterns
but uses Taichi for computation.
"""

import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import genesis as gs

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern


@dataclass
class GridPattern(RaycastPattern):
    """Configuration for the grid pattern for ray-casting.

    Defines a 2D grid of rays in the coordinates of the sensor.
    """

    resolution: float = 0.1  # Grid resolution (in meters)
    size: tuple[float, float] = (2.0, 2.0)  # Grid size (length, width) (in meters)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)  # Ray direction
    ordering: str = "xy"  # Ordering of points: "xy" or "yx"

    def get_return_shape(self) -> tuple[int, ...]:
        x_coords = np.arange(-self.size[0] / 2, self.size[0] / 2 + 1e-9, self.resolution)
        y_coords = np.arange(-self.size[1] / 2, self.size[1] / 2 + 1e-9, self.resolution)
        n_rays = len(x_coords) * len(y_coords)
        return (1, n_rays, 3)


@dataclass
class LidarPattern(RaycastPattern):
    """Configuration for the LiDAR pattern for ray-casting."""

    channels: int = 32  # Number of vertical channels (beams)
    vertical_fov_range: tuple[float, float] = (-15.0, 15.0)  # Vertical FOV in degrees
    horizontal_fov_range: tuple[float, float] = (-180.0, 180.0)  # Horizontal FOV in degrees
    horizontal_res: float = 1.0  # Horizontal resolution in degrees

    def get_return_shape(self) -> tuple[int, ...]:
        # Handle 360-degree horizontal FOV (exclude last point to avoid overlap)
        h_range = self.horizontal_fov_range[1] - self.horizontal_fov_range[0]
        num_horizontal_angles = math.ceil(h_range / self.horizontal_res)
        if abs(abs(h_range) - 360.0) < 1e-6:
            num_horizontal_angles -= 1
        return (self.channels, num_horizontal_angles, 3)


@dataclass
class BpearlPattern(RaycastPattern):
    """Configuration for the Bpearl pattern for ray-casting."""

    horizontal_fov: float = 360.0  # Horizontal field of view (in degrees)
    horizontal_res: float = 10.0  # Horizontal resolution (in degrees)
    vertical_ray_angles: list[float] = field(
        default_factory=lambda: [
            89.5,
            86.6875,
            83.875,
            81.0625,
            78.25,
            75.4375,
            72.625,
            69.8125,
            67.0,
            64.1875,
            61.375,
            58.5625,
            55.75,
            52.9375,
            50.125,
            47.3125,
            44.5,
            41.6875,
            38.875,
            36.0625,
            33.25,
            30.4375,
            27.625,
            24.8125,
            22,
            19.1875,
            16.375,
            13.5625,
            10.75,
            7.9375,
            5.125,
            2.3125,
        ]
    )

    def get_return_shape(self) -> tuple[int, ...]:
        h_angles = np.arange(-self.horizontal_fov / 2, self.horizontal_fov / 2, self.horizontal_res)
        return (len(self.vertical_ray_angles), len(h_angles), 3)


@dataclass
class SphericalPattern(RaycastPattern):
    """Configuration for spherical uniform pattern for ray-casting."""

    n_scan_lines: int = 32  # Number of vertical scan lines
    n_points_per_line: int = 64  # Number of horizontal points per scan line
    fov_vertical: float = 30.0  # Vertical field of view in degrees
    fov_horizontal: float = 360.0  # Horizontal field of view in degrees

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.n_scan_lines, self.n_points_per_line, 3)


@dataclass
class LivoxPattern(RaycastPattern):
    """Configuration for Livox LiDAR pattern for ray-casting."""

    sensor_type: str = "avia"  # Type of Livox sensor
    samples: int = 24000  # Number of ray samples per scan frame
    downsample: int = 1  # Downsampling factor for ray patterns
    use_simple_grid: bool = False  # Whether to use simple grid pattern instead
    rolling_window_start: int = 0  # Starting index for rolling window sampling

    # Simple grid parameters (used when use_simple_grid=True)
    horizontal_line_num: int = 80
    vertical_line_num: int = 50
    horizontal_fov_deg_min: float = -180
    horizontal_fov_deg_max: float = 180
    vertical_fov_deg_min: float = -2
    vertical_fov_deg_max: float = 57

    # Dynamic pattern parameters
    enable_dynamic_pattern: bool = True  # Enable dynamic ray updates
    pattern_rotation_speed: float = 0.1  # Rotation speed for dynamic patterns

    _is_dynamic: bool = True

    def get_return_shape(self) -> tuple[int, ...]:
        # Livox sensor parameters
        LIVOX_PARAMS = {
            "avia": {"samples": 24000},
            "HAP": {"samples": 45300},
            "horizon": {"samples": 24000},
            "mid40": {"samples": 24000},
            "mid70": {"samples": 10000},
            "mid360": {"samples": 20000},
            "tele": {"samples": 24000},
        }

        if self.use_simple_grid:
            n_rays = self.vertical_line_num * self.horizontal_line_num
        else:
            # For dynamic patterns, use max samples (before downsampling)
            params = LIVOX_PARAMS.get(self.sensor_type, {})
            n_rays = min(self.samples, params.get("samples", self.samples))
            if self.downsample > 1:
                n_rays = n_rays // self.downsample
        return (1, n_rays, 3)


@dataclass
class SpinningLidarPattern(RaycastPattern):
    """Configuration for traditional spinning lidars (HDL64, VLP32, OS128)."""

    sensor_type: str = "hdl64"  # one of {"hdl64", "vlp32", "os128"}
    f_rot: float = 10.0  # rotation frequency (Hz)
    sample_rate: float = 2.2e6  # samples per second (defaults for HDL64)
    n_channels: int = 64  # number of channels (64/32/128)
    phi_fov: tuple[float, float] = (-24.9, 2.0)  # deg, used for HDL64 when no custom table

    def get_return_shape(self) -> tuple[int, ...]:
        VLP32_ANGLES_DEG = np.array(
            [
                -25.0,
                -22.5,
                -20.0,
                -15.0,
                -13.0,
                -10.0,
                -5.0,
                -3.0,
                -2.333,
                -1.0,
                -0.667,
                -0.333,
                0.0,
                0.0,
                0.333,
                0.667,
                1.0,
                1.333,
                1.667,
                2.0,
                2.333,
                2.667,
                3.0,
                3.333,
                3.667,
                4.0,
                5.0,
                7.0,
                10.0,
                15.0,
                17.0,
                20.0,
            ],
            dtype=np.float32,
        )

        sensor = self.sensor_type.lower()
        if sensor == "hdl64":
            n_channels = self.n_channels if self.n_channels is not None else 64
            f_rot = self.f_rot
            sample_rate = self.sample_rate if self.sample_rate is not None else 2.2e6
        elif sensor == "vlp32":
            n_channels = len(VLP32_ANGLES_DEG)
            f_rot = self.f_rot
            sample_rate = self.sample_rate if self.sample_rate is not None else 1.2e6
        else:  # os128
            n_channels = self.n_channels if self.n_channels is not None else 128
            f_rot = self.f_rot if self.f_rot is not None else 20.0
            sample_rate = self.sample_rate if self.sample_rate is not None else 5.2e6

        # Calculate number of time steps
        n_time_steps = int(sample_rate / (f_rot * n_channels))
        n_rays = n_time_steps * n_channels
        return (1, n_rays, 3)


@register_pattern(GridPattern)
class GridPatternGenerator(RaycastPatternGenerator):
    """Grid pattern generator using Taichi."""

    def __init__(self, cfg: GridPattern):
        super().__init__(cfg)
        # Pre-compute n_rays from config
        self.x_coords = np.arange(-cfg.size[0] / 2, cfg.size[0] / 2 + 1e-9, cfg.resolution)
        self.y_coords = np.arange(-cfg.size[1] / 2, cfg.size[1] / 2 + 1e-9, cfg.resolution)

    def get_ray_directions(self) -> torch.Tensor:
        dirs = torch.zeros(self.cfg.get_return_shape(), dtype=gs.tc_float, device=gs.device)
        dirs[0, :, :] = torch.tensor(self.cfg.direction, dtype=gs.tc_float, device=gs.device)
        return dirs

    def get_ray_starts(self) -> torch.Tensor:
        if self.cfg.ordering not in ["xy", "yx"]:
            raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{self.cfg.ordering}'.")
        if self.cfg.resolution <= 0:
            raise ValueError(f"Resolution must be greater than 0. Received: '{self.cfg.resolution}'.")
        if self.cfg.ordering == "xy":
            grid_x, grid_y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        else:
            grid_x, grid_y = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")
        starts = torch.zeros(self.cfg.get_return_shape(), dtype=gs.tc_float, device=gs.device)
        starts[0, :, 0] = grid_x.flatten()
        starts[0, :, 1] = grid_y.flatten()

        return starts


@register_pattern(LidarPattern)
class LidarPatternGenerator(RaycastPatternGenerator):
    """LiDAR pattern generator using Taichi."""

    def __init__(self, cfg: LidarPattern):
        super().__init__(cfg)
        # Handle 360-degree horizontal FOV (exclude last point to avoid overlap)
        self.h_range = cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]
        self.num_horizontal_angles = math.ceil(self.h_range / cfg.horizontal_res)
        if abs(abs(self.h_range) - 360.0) < 1e-6:
            self.num_horizontal_angles -= 1

    def get_ray_directions(self) -> torch.Tensor:
        """Generate LiDAR pattern."""
        vertical_angles = np.linspace(self.cfg.vertical_fov_range[0], self.cfg.vertical_fov_range[1], self.cfg.channels)

        horizontal_angles = np.linspace(
            self.cfg.horizontal_fov_range[0], self.cfg.horizontal_fov_range[1], self.num_horizontal_angles
        )

        v_rad = np.deg2rad(vertical_angles)
        h_rad = np.deg2rad(horizontal_angles)

        v_angles, h_angles = np.meshgrid(v_rad, h_rad, indexing="ij")

        # Spherical to Cartesian conversion (Z is up)
        x = np.cos(v_angles) * np.cos(h_angles)
        y = np.cos(v_angles) * np.sin(h_angles)
        z = np.sin(v_angles)

        # Stack and reshape to [n_scan_lines, n_points_per_line, 3]
        ray_directions = np.stack([x, y, z], axis=-1).astype(np.float32)

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)


@register_pattern(BpearlPattern)
class BpearlPatternGenerator(RaycastPatternGenerator):
    """Bpearl pattern generator using Taichi."""

    def __init__(self, cfg: BpearlPattern):
        super().__init__(cfg)
        # Pre-compute n_rays from config
        self.h_angles = np.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate Bpearl pattern."""
        # Vertical angles (predefined for Bpearl)
        v_angles = np.array(self.cfg.vertical_ray_angles, dtype=np.float32)

        # Create meshgrid
        pitch, yaw = np.meshgrid(v_angles, self.h_angles, indexing="xy")
        pitch_rad = np.deg2rad(pitch.flatten()) + np.pi / 2
        yaw_rad = np.deg2rad(yaw.flatten())

        # Spherical to Cartesian
        x = np.sin(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(pitch_rad) * np.sin(yaw_rad)
        z = np.cos(pitch_rad)

        # Bpearl uses negative direction convention
        ray_directions = -np.stack([x, y, z], axis=1).astype(np.float32)

        # Reshape to [n_scan_lines, n_points_per_line, 3]
        ray_directions = ray_directions.reshape(*self.cfg.get_return_shape())

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)


@register_pattern(SphericalPattern)
class SphericalPatternGenerator(RaycastPatternGenerator):
    """Spherical uniform pattern generator using Taichi."""

    def __init__(self, cfg: SphericalPattern):
        super().__init__(cfg)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spherical uniform pattern."""
        # Create angular grids
        vertical_angles = np.linspace(-self.cfg.fov_vertical / 2, self.cfg.fov_vertical / 2, self.cfg.n_scan_lines)
        horizontal_angles = np.linspace(
            -self.cfg.fov_horizontal / 2, self.cfg.fov_horizontal / 2, self.cfg.n_points_per_line
        )

        # Generate ray vectors in spherical coordinates
        ray_vectors = np.zeros((self.cfg.n_scan_lines, self.cfg.n_points_per_line, 3), dtype=np.float32)

        for i, v_angle in enumerate(vertical_angles):
            for j, h_angle in enumerate(horizontal_angles):
                v_rad = np.deg2rad(v_angle)
                h_rad = np.deg2rad(h_angle)

                # Convert spherical to cartesian (x=forward, y=left, z=up)
                ray_vectors[i, j, 0] = np.cos(v_rad) * np.cos(h_rad)  # x (forward)
                ray_vectors[i, j, 1] = np.cos(v_rad) * np.sin(h_rad)  # y (left)
                ray_vectors[i, j, 2] = np.sin(v_rad)  # z (up)

        return torch.from_numpy(ray_vectors).to(device=gs.device, dtype=gs.tc_float)


@register_pattern(LivoxPattern)
class LivoxPatternGenerator(RaycastPatternGenerator):
    """Livox LiDAR pattern generator with caching (prefers precomputed .npy scan patterns)."""

    LIVOX_PARAMS = {
        "avia": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 77.2,
            "samples": 24000,
        },
        "HAP": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 45300,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
        },
        "horizon": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid40": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid70": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 70.4,
            "samples": 10000,
        },
        "mid360": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 360.0,
            "vertical_fov": 59.0,
            "samples": 20000,
        },
        "tele": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 14.5,
            "vertical_fov": 16.1,
            "samples": 24000,
        },
    }

    _pattern_cache: dict[str, np.ndarray] = {}

    def __init__(self, cfg: LivoxPattern):
        super().__init__(cfg)
        self.current_start_index = 0
        self.generated_patterns = {}
        self._last_update_tick = None

    def get_actual_ray_count(self) -> int:
        """Get the actual number of rays that will be returned (after downsampling)."""
        return self.cfg.get_return_shape()[1]

    def get_ray_directions(self) -> torch.Tensor:
        """Generate Livox pattern with caching."""
        if self.cfg.use_simple_grid:
            ray_directions = self._generate_simple_grid_pattern(self.cfg)
        else:
            ray_directions = self._generate_livox_scan_pattern(self.cfg)

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)

    def _generate_simple_grid_pattern(self, cfg: LivoxPattern) -> np.ndarray:
        """Generate simple grid pattern for Livox sensor."""
        # Convert FOV to radians
        h_fov_min = math.radians(cfg.horizontal_fov_deg_min)
        h_fov_max = math.radians(cfg.horizontal_fov_deg_max)
        v_fov_min = math.radians(cfg.vertical_fov_deg_min)
        v_fov_max = math.radians(cfg.vertical_fov_deg_max)

        # Generate grid pattern
        ray_directions = np.zeros((cfg.vertical_line_num, cfg.horizontal_line_num, 3), dtype=np.float32)

        for i in range(cfg.vertical_line_num):
            for j in range(cfg.horizontal_line_num):
                # Calculate angles
                if cfg.vertical_line_num > 1:
                    v_angle = v_fov_min + (v_fov_max - v_fov_min) * i / (cfg.vertical_line_num - 1)
                else:
                    v_angle = (v_fov_min + v_fov_max) / 2

                if cfg.horizontal_line_num > 1:
                    h_angle = h_fov_min + (h_fov_max - h_fov_min) * j / (cfg.horizontal_line_num - 1)
                else:
                    h_angle = (h_fov_min + h_fov_max) / 2

                # Convert to Cartesian (x=forward, y=left, z=up)
                cos_h = math.cos(h_angle)
                sin_h = math.sin(h_angle)
                cos_v = math.cos(v_angle)
                sin_v = math.sin(v_angle)

                ray_directions[i, j, 0] = cos_h * cos_v  # x (forward)
                ray_directions[i, j, 1] = sin_h * cos_v  # y (left)
                ray_directions[i, j, 2] = sin_v  # z (up)

        return ray_directions

    def _generate_livox_scan_pattern(self, cfg: LivoxPattern) -> np.ndarray:
        """Generate realistic Livox scan pattern using NumPy RNG."""
        if cfg.sensor_type not in self.LIVOX_PARAMS:
            raise ValueError(f"Unsupported Livox sensor type: {cfg.sensor_type}")

        params = self.LIVOX_PARAMS[cfg.sensor_type]

        # Create cache key
        cache_key = self._create_cache_key(cfg, params)

        # Check if pattern is already cached
        if cache_key in self._pattern_cache:
            full_pattern = self._pattern_cache[cache_key]
        else:
            # Generate new pattern using Taichi
            full_pattern = self._generate_taichi_pattern(cfg, params)
            self._pattern_cache[cache_key] = full_pattern

        # Store pattern for this instance
        self.generated_patterns[cfg.sensor_type] = full_pattern

        # Return sampled pattern (first frame)
        return self._sample_pattern(full_pattern, cfg)

    def _create_cache_key(self, cfg: LivoxPattern, params: dict) -> str:
        """Create a unique cache key for the pattern configuration."""
        key_data = {
            "sensor_type": cfg.sensor_type,
            "horizontal_fov": params.get("horizontal_fov", 360.0),
            "vertical_fov": params.get("vertical_fov", 90.0),
            "total_samples": params["samples"] * 10,  # Generate enough for temporal sampling
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _generate_taichi_pattern(self, cfg: LivoxPattern, params: dict) -> np.ndarray:
        """Load Livox pattern angles from precomputed .npy files when available.
        Falls back to NumPy RNG when files are missing.
        Returns array of shape (N, 2) with columns [theta, phi] in radians.
        """

        # Map sensor type to pattern filename (note HAP is upper-case)
        pattern_files = {
            "avia": "avia.npy",
            "horizon": "horizon.npy",
            "HAP": "HAP.npy",
            "mid360": "mid360.npy",
            "mid40": "mid40.npy",
            "mid70": "mid70.npy",
            "tele": "tele.npy",
        }
        pattern_file = pattern_files.get(cfg.sensor_type)
        pattern_angles: np.ndarray | None = None
        if pattern_file is not None:
            # Local scan_patterns directory relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(script_dir, "patterns", pattern_file)
            pattern_path = local_path
            if not os.path.exists(pattern_path):
                # Optional unified path fallback (kept for compatibility, may not exist)
                omniperc_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))))
                )
                unified_dir = os.path.join(
                    omniperc_root,
                    "LidarSensor",
                    "LidarSensor",
                    "sensor_pattern",
                    "sensor_lidar",
                    "scan_mode",
                )
                unified_path = os.path.join(unified_dir, pattern_file)
                if os.path.exists(unified_path):
                    pattern_path = unified_path
            if os.path.exists(pattern_path):
                data = np.load(pattern_path)
                # Expect shape (N, 2): [theta, phi]
                if isinstance(data, np.lib.npyio.NpzFile):
                    # If accidentally using .npz, try common keys
                    if "angles" in data:
                        data = data["angles"]
                    elif "theta" in data and "phi" in data:
                        data = np.stack([data["theta"], data["phi"]], axis=-1)
                    else:
                        # Fallback: try first 2 columns of the first array
                        first_key = list(data.files)[0]
                        data = data[first_key]
                if data.ndim == 2 and data.shape[1] >= 2:
                    pattern_angles = data[:, :2].astype(np.float32)
        # Fallback to RNG if files missing or invalid
        if pattern_angles is None:
            total_samples = params["samples"] * 10
            h_fov = math.radians(params.get("horizontal_fov", 360.0))
            v_fov = math.radians(params.get("vertical_fov", 90.0))
            rng = np.random.default_rng(seed=abs(hash(cfg.sensor_type)) % (2**32))
            pattern_angles = np.empty((total_samples, 2), dtype=np.float32)
            pattern_angles[:, 0] = rng.uniform(-0.5 * h_fov, 0.5 * h_fov, size=total_samples)  # theta
            pattern_angles[:, 1] = rng.uniform(-0.5 * v_fov, 0.5 * v_fov, size=total_samples)  # phi
        return pattern_angles

    def _sample_pattern(
        self, full_pattern: np.ndarray, cfg: LivoxPattern, start_index: int | None = None
    ) -> np.ndarray:
        """Sample a subset of rays from the full pattern.
        If start_index is provided, sampling starts from there; otherwise uses cfg.rolling_window_start.
        """
        total_rays = full_pattern.shape[0]
        samples = min(cfg.samples, total_rays)

        # Rolling window sampling start
        if start_index is None:
            start_idx = cfg.rolling_window_start % total_rays
        else:
            start_idx = start_index % total_rays

        if start_idx + samples <= total_rays:
            selected_angles = full_pattern[start_idx : start_idx + samples]
        else:
            # Wraparound case
            end_samples = total_rays - start_idx
            begin_samples = samples - end_samples
            selected_angles = np.vstack([full_pattern[start_idx:], full_pattern[:begin_samples]])

        # Apply downsampling if requested
        if cfg.downsample > 1:
            selected_angles = selected_angles[:: cfg.downsample]

        # Convert angles to Cartesian coordinates
        theta = selected_angles[:, 0]  # horizontal angles
        phi = selected_angles[:, 1]  # vertical angles

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Convert to ray directions (x=forward, y=left, z=up)
        x = cos_theta * cos_phi  # forward component
        y = sin_theta * cos_phi  # left component
        z = sin_phi  # up component

        ray_directions = np.stack([x, y, z], axis=1).astype(np.float32)

        # Normalize directions
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / norms

        # Return as flat array for compatibility with grid patterns
        return ray_directions.reshape(1, -1, 3)

    def update_dynamic_pattern(self, cfg: LivoxPattern, time_step: float) -> np.ndarray | None:
        """Update pattern for dynamic Livox sensors by advancing an internal rolling index.
        time_step is treated as the current simulation time in seconds. We update once per tick
        determined by pattern_update_rate (updates per second). No new cfg is created.
        """
        if not cfg.enable_dynamic_pattern or cfg.sensor_type not in self.generated_patterns:
            return None

        # Determine whether to update this call. Default: 10 updates/sec (every 0.1s).
        # You can map pattern_rotation_speed to a rate if desired; keep simple for now.
        pattern_update_rate = 10  # Hz
        current_tick = int(time_step * pattern_update_rate + 1e-6)
        if self._last_update_tick is not None and current_tick == self._last_update_tick:
            return None  # Not time to update yet

        # Time to update
        self._last_update_tick = current_tick
        full_pattern = self.generated_patterns[cfg.sensor_type]
        total_rays = full_pattern.shape[0]

        # Advance rolling window by one frame worth of samples
        self.current_start_index = (self.current_start_index + cfg.samples) % total_rays

        # Sample using the updated start index
        return self._sample_pattern(full_pattern, cfg, start_index=self.current_start_index)


@register_pattern(SpinningLidarPattern)
class SpinningLidarPatternGenerator(RaycastPatternGenerator):
    """Spinning lidar pattern generator (Velodyne HDL64/VLP32, Ouster OS128)."""

    VLP32_ANGLES_DEG = np.array(
        [
            -25.0,
            -22.5,
            -20.0,
            -15.0,
            -13.0,
            -10.0,
            -5.0,
            -3.0,
            -2.333,
            -1.0,
            -0.667,
            -0.333,
            0.0,
            0.0,
            0.333,
            0.667,
            1.0,
            1.333,
            1.667,
            2.0,
            2.333,
            2.667,
            3.0,
            3.333,
            3.667,
            4.0,
            5.0,
            7.0,
            10.0,
            15.0,
            17.0,
            20.0,
        ],
        dtype=np.float32,
    )

    def __init__(self, cfg: SpinningLidarPattern):
        super().__init__(cfg)

    def get_ray_directions(self) -> torch.Tensor:
        sensor = self.cfg.sensor_type.lower()
        if sensor not in {"hdl64", "vlp32", "os128"}:
            raise ValueError(f"Unsupported spinning lidar type: {self.cfg.sensor_type}")

        # Determine vertical angles (phi) and channel count
        if sensor == "hdl64":
            n_channels = self.cfg.n_channels if self.cfg.n_channels is not None else 64
            phi_min, phi_max = np.deg2rad(self.cfg.phi_fov)
            phi = np.linspace(phi_min, phi_max, n_channels, dtype=np.float32)
            f_rot = self.cfg.f_rot
            sample_rate = self.cfg.sample_rate if self.cfg.sample_rate is not None else 2.2e6
        elif sensor == "vlp32":
            phi = np.deg2rad(self.VLP32_ANGLES_DEG)
            n_channels = phi.shape[0]
            f_rot = self.cfg.f_rot
            sample_rate = self.cfg.sample_rate if self.cfg.sample_rate is not None else 1.2e6
        else:  # os128
            n_channels = self.cfg.n_channels if self.cfg.n_channels is not None else 128
            phi = np.deg2rad(np.linspace(-22.5, 22.5, n_channels, dtype=np.float32))
            f_rot = self.cfg.f_rot if self.cfg.f_rot is not None else 20.0
            sample_rate = self.cfg.sample_rate if self.cfg.sample_rate is not None else 5.2e6

        # Time sequence over one rotation
        t = np.arange(0.0, 1.0 / f_rot, n_channels / sample_rate, dtype=np.float32)[:, None]
        # Horizontal angles (theta)
        theta = (2.0 * np.pi * f_rot * t) % (2.0 * np.pi)

        # Broadcast to grids
        theta_grid = theta + np.zeros((1, n_channels), dtype=np.float32)
        phi_grid = np.zeros_like(theta, dtype=np.float32) + phi

        # Flatten
        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)

        # Convert to directions (x=forward, y=left, z=up)
        cos_theta = np.cos(theta_flat)
        sin_theta = np.sin(theta_flat)
        cos_phi = np.cos(phi_flat)
        sin_phi = np.sin(phi_flat)
        x = cos_theta * cos_phi
        y = sin_theta * cos_phi
        z = sin_phi
        dirs = np.stack([x, y, z], axis=1).astype(np.float32)
        # Normalize (safety)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        return torch.from_numpy(dirs.reshape(1, -1, 3)).to(device=gs.device, dtype=gs.tc_float)
