"""
Depth camera (pinhole) ray pattern generation for Genesis.

Follows the same style as lidar_pattern: define a config, a generator, and helpers
that return local-frame ray starts and directions of shape [H, W, 3].

Coordinate convention (robotics camera frame):
- x: forward, y: left, z: up
- Derived from standard camera frame (x right, y down, z forward) via:
  [x_r, y_r, z_r] = [z_c, -x_c, -y_c]
"""

from dataclasses import dataclass
import math
import numpy as np
import torch
from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern
import genesis as gs


@dataclass
class DepthCameraPattern(RaycastPattern):
    """Pinhole depth camera pattern configuration.

    You can provide intrinsics (fx, fy, cx, cy). If missing, they will be computed
    from image size and FOVs when provided.
    """

    width: int = 640
    height: int = 480
    # Intrinsics (in pixels)
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    # Alternative specification via FOV (degrees)
    fov_horizontal: float | None = 90.0
    fov_vertical: float | None = None

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.height, self.width, 3)


@register_pattern(DepthCameraPattern)
class DepthCameraPatternGenerator(RaycastPatternGenerator):
    """Generate ray directions for a pinhole camera."""

    def __init__(self, cfg: DepthCameraPattern):
        super().__init__(cfg)

    def get_ray_directions(self) -> torch.Tensor:
        W, H = int(self.cfg.width), int(self.cfg.height)

        if W <= 0 or H <= 0:
            raise ValueError("width and height must be positive")

        # Derive intrinsics if needed
        fx, fy, cx, cy = self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy
        fh, fv = self.cfg.fov_horizontal, self.cfg.fov_vertical

        if fx is None or fy is None:
            if fh is None and fv is None:
                # Default FOVs if nothing provided
                fh = 90.0
            if fh is not None and fv is None:
                # preserve aspect ratio
                fh_rad = math.radians(fh)
                fv_rad = 2.0 * math.atan((H / W) * math.tan(fh_rad / 2.0))
            elif fv is not None and fh is None:
                fv_rad = math.radians(fv)
                fh_rad = 2.0 * math.atan((W / H) * math.tan(fv_rad / 2.0))
            else:
                fh_rad = math.radians(fh)
                fv_rad = math.radians(fv)
            fx = W / (2.0 * math.tan(fh_rad / 2.0))
            fy = H / (2.0 * math.tan(fv_rad / 2.0))
        if cx is None:
            cx = W * 0.5
        if cy is None:
            cy = H * 0.5

        # Pixel centers
        u = np.arange(0, W, dtype=np.float32) + 0.5  # shape (W,)
        v = np.arange(0, H, dtype=np.float32) + 0.5  # shape (H,)
        uu, vv = np.meshgrid(u, v, indexing="xy")  # (H, W)

        # Camera frame (x right, y down, z forward)
        x_c = (uu - cx) / float(fx)
        y_c = (vv - cy) / float(fy)
        z_c = np.ones_like(x_c, dtype=np.float32)

        # Robotics camera frame (x forward, y left, z up): [z, -x, -y]
        x_r = z_c
        y_r = -x_c
        z_r = -y_c
        dirs = np.stack([x_r, y_r, z_r], axis=-1).astype(np.float32)  # (H, W, 3)

        # Normalize
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        return torch.from_numpy(dirs).to(device=gs.device, dtype=gs.tc_float)  # (H, W, 3)
